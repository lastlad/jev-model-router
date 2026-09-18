# Jev-powered model router: research findings and implementation plan

Status: proposal, revised 2026-09-18 after decisions below. Nothing in this
repo is built yet.

## 0. Decisions taken

| Question | Decision |
|---|---|
| Front door | OpenAI-compatible `/v1/chat/completions`, served by **LiteLLM proxy**; the router is a LiteLLM `CustomLogger` plugin |
| Candidate providers | Anthropic and OpenAI from phase 1; LiteLLM owns the format translation |
| Jev access | Keys are available; `typesafe-sdk` against `jev-latest` from day one, LLM-backed adapter kept only for CI |
| Objective | Quality first, cost second |
| Conversation identity | `x-litellm-session-id` / `x-litellm-trace-id` header when present, otherwise derived by the router |
| Privacy | Redacted, minimized transcripts may be sent to TypeSafe |
| Language | Python 3.12 (LiteLLM plugins are Python; the Jev SDK is Python-first) |

## 1. What we are building

A routing layer that sits between end-user applications and LLM providers.
Applications call a LiteLLM proxy with `model: "jev-auto"`. Each incoming turn
is inspected, TypeSafe's **Jev** model judges what the turn needs, and
deterministic code turns that judgment plus the conversation's routing history
and prompt-cache state into a concrete `(model, effort)` choice. LiteLLM then
serves the turn with that model and returns an OpenAI-format response.

The three requirements from the brief, restated as design constraints:

1. **Whole-conversation awareness.** The decision must consider the full
   transcript, tool definitions, and in-flight tool calls, not just the newest
   user message.
2. **Routing memory.** The router keeps its own per-conversation ledger of
   prior decisions and uses it on the next turn.
3. **Cache-aware economics.** Staying on the incumbent model is worth real
   money when the prompt prefix is cached. Switching must be priced against
   that, not decided on complexity alone.

## 2. Research findings

### 2.1 Jev, in one paragraph

Jev is the first "System One" model from TypeSafe AI (founded by Diogo
Almeida, ex-OpenAI), announced 2026-09-15. It does not generate text. You send
it a `state` (string, JSON object, or array) plus a map of typed questions, and
it returns typed answers with calibrated probabilities in one parallel pass.
TypeSafe reports 70–500 ms end-to-end latency, `$0.042` per million input
tokens, free output, and trains for calibration with an RL objective they call
RLCD. It is in early access behind a waitlist, and also reachable through
OpenRouter (`~typesafe/jev-latest`) and Vercel AI Gateway.

### 2.2 The API surface (verified from the official SDK source)

Endpoint: `POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer <key>`.

Request:

```json
{
  "model": "jev-latest",
  "state": "<string | object | array>",
  "questions": {
    "<key>": { "type": "choice", "instructions": "...", "criteria": { "label": "description or null" } },
    "<key>": { "type": "score",  "instructions": "...", "criteria": [ "level 0 desc", "level 1 desc", "..." ] },
    "<key>": { "type": "noul",   "instructions": "...", "criteria": { "true": "...", "false": "..." } }
  }
}
```

Response:

```json
{
  "model": "jev-1.13.0",
  "usage": { "input_tokens": 1234, "output_tokens": 0 },
  "answers": {
    "<key>": { "type": "choice", "choice": "label", "confidence": 0.81, "probabilities": { "label": 0.81, "...": 0.19 } },
    "<key>": { "type": "score",  "score": 1.7, "confidence": 0.62, "legend": { "0": "...", "1": "..." }, "probabilities": { "0": 0.1, "1": 0.3, "2": 0.6 } },
    "<key>": { "type": "noul",   "noul": 0.93 }
  }
}
```

Rules that matter for us:

| Fact | Consequence for the router |
|---|---|
| `choice` takes 1–255 labels; `score` takes 2–10 ordered levels; `noul` needs `instructions` or `criteria` | Tier ladder fits comfortably in a `score`; candidate list fits in a `choice` |
| `state` cannot be `null` (422); empty string/object is fine | Always send an object |
| State plus the longest single question must fit in ~32k tokens; state plus all questions in ~64k | We must build a minimized state, never send the raw transcript |
| Questions are evaluated independently and in parallel | Ask every question we might need in one call ("speculative fan-out") |
| Accuracy drops as unrelated content grows in the state | Retrieve and filter in code first; send only what the questions need |
| Text only | Images are reduced to signals (`has_image`, count) in the state |
| Rate limits for jev-1.13: 250k tokens/s, 1,200 requests/min; 429 on overrun | Fine for a test product; SDK retries with backoff |
| No rationale is returned | Store full probability distributions for audit and calibration |
| Calibration is a property of groups of predictions, not any single answer | Confidence-gate every automated action; fall back on low confidence |
| Python SDK `typesafe-sdk` (py ≥ 3.10) exposes `TypeSafeClient` / `AsyncTypeSafeClient`, `system_one(state=, questions=)`, `Choice`, `Score`, `Noul`; JS SDK `@typesafe-ai/sdk` (Node 20+) | Python is the natural first language |
| `system-one-adapter-python` is an official drop-in `TypeSafeClient` replacement backed by OpenAI or Anthropic | Lets us develop and run CI before the early-access key arrives |

TypeSafe's own design guidance, which this plan follows: *"Code calculates.
Jev judges. Reasoning models reason."* Questions should be atomic, pass the
"five-second expert" test, include a catch-all option, and thresholds should be
set per action rather than globally.

### 2.3 Prompt-cache economics per provider (what "cache savings" actually are)

| Provider | Mechanism | Read discount | Write premium | Minimum prefix | Scope | Usage field |
|---|---|---|---|---|---|---|
| Anthropic | Explicit `cache_control` breakpoints; prefix match over `tools → system → messages` | ~0.1× input (0.025× on Fable 5.1) | 1.25× (5-min TTL) or 2× (1-hour TTL) | 512 (Opus 5, Fable 5.x), 1024 (Sonnet 5, Opus 4.8), 4096 (Haiku 4.5, Opus 4.6) | **Per model** | `cache_read_input_tokens`, `cache_creation_input_tokens` |
| OpenAI | Automatic on repeated prefixes ≥1024 tokens, 128-token steps | ~0.1× on GPT-5.x | 1.25× on GPT-5.6+ | 1024 | Per model | `prompt_tokens_details.cached_tokens` |
| Gemini | Implicit caching, on by default | 0.1× on 2.5+ | none | 1024 (Flash) to 2048–4096 (Pro) | Per model | `cachedContentTokenCount` |

The decisive fact: **caches are model-scoped on every provider.** Switching
models mid-conversation re-processes the entire history at full price and pays
the write premium again. On Anthropic the 5-minute TTL refreshes on every read,
so a conversation with turns under five minutes apart keeps the cache warm
indefinitely as long as the model stays fixed.

Two more Anthropic-specific facts shape the design:

- **Thinking blocks are bound to the producing model.** Fable 5.1's blocks are
  dropped by other models (unbilled) and vice versa. A model switch therefore
  also throws away the model's own reasoning context. That is a quality cost of
  switching, separate from the cache cost.
- **Effort can change without a model switch.** On Claude Opus 5 and Fable 5.1
  a per-message effort system message (beta
  `mid-conversation-output-config-2026-07-01`) changes effort without
  invalidating the messages cache. So the cheapest "downgrade" is often the same
  model at lower effort, not a different model. The router's unit of choice is
  therefore `(model, effort)`, not just `model`. Through LiteLLM that beta is
  not reachable (section 2.5), so on the chosen path an effort change keeps
  only the tools-plus-system cache; it is still cheaper than a model switch.

### 2.4 Prior art

- **prismhq/jev-router** (MIT, Python, LiteLLM pre-call hook). Sends the last
  8 messages truncated to 2,000 chars plus `has_image` / `has_tools` /
  `message_count`, asks one `choice` question over the candidate list, and takes
  the answer. Rules baseline picks cheapest-eligible. It does **not** use
  confidence, routing history, tool-loop state, or cache state, and it asks Jev
  to do the cost trade-off in prose ("prefer the cheapest model that clears the
  quality bar"), which is exactly the arithmetic a System One model is not built
  for. Useful as a reference for eligibility filtering and the YAML candidate
  schema; not a base to fork.
- **Bifrost / 9router / OmniRoute feature proposals** converge on the same
  shape: a `choice` over models, a `noul` for tools, a `score` for complexity,
  fail-open on timeout, 1–2 s budget. None address multi-turn stickiness or
  caching.
- **RouteLLM** (LMSYS/Berkeley) is the academic baseline: trained classifiers
  predicting whether a weak model suffices, reporting up to 85% cost reduction
  at 95% of strong-model quality on MT-Bench. It routes single prompts.
- **arXiv 2604.02367** benchmarks 1–4B self-hosted SLMs as front-door routers:
  best ~0.79 exact-match at ~1 s median, and none met the ≥0.85 / ≤2 s P95
  viability gate. Jev's claimed 70–500 ms and calibrated output is the
  interesting delta to test.
- Anthropic's own guidance on cost optimisation: measure "one strong model at
  lower effort" before building a multi-model cascade, because a cascade
  forfeits cache reuse across models. Our design adopts this as a first-class
  candidate rather than a competitor.

### 2.5 LiteLLM proxy as the host (verified against LiteLLM docs and issues)

LiteLLM already solves the parts of this product that are not about routing:
OpenAI-compatible ingress, translation to Anthropic and OpenAI, streaming,
retries, spend logging, and a Redis-backed cache. The existing Jev router
(prismhq/jev-router) is a LiteLLM plugin for the same reason. What LiteLLM
gives us, and where its edges are:

| Capability | Status | Consequence |
|---|---|---|
| `async_pre_call_hook(user_api_key_dict, cache, data, call_type)` on `/v1/chat/completions` | Works; may rewrite `data["model"]`, `data["reasoning_effort"]`, messages, metadata, or reject | This is the router's entry point |
| Same hook on LiteLLM's Anthropic-format `/v1/messages` | **Bypassed** (open issues #30469, #27518) | Only the OpenAI-format endpoint is supported; documented as a constraint |
| `cache_control` in OpenAI-format content blocks forwarded to Anthropic, Bedrock, Gemini | Works | The hook can place explicit breakpoints |
| `cache_control_injection_points` per deployment (`location: message`, `role`, `index`) | Works | Zero-code default: system block plus `index: -1` |
| Usage normalization: `prompt_tokens_details.cached_tokens` and `cache_creation_input_tokens` on chat completions, for Anthropic and OpenAI | Works | One ledger update path for both providers |
| Streaming cache accounting | Historic cost bug (#11789) closed; `/v1/messages`-to-OpenAI-upstream bug (#41067) open but not our path | Verify on the pinned LiteLLM version in phase 0; the ledger reconciles from real usage either way |
| `reasoning_effort` mapped to Anthropic `output_config.effort` (low/medium/high/xhigh/max, model-gated) | Works for Claude 4.6+; model gating has lagged new releases before (#25957) | Pin a LiteLLM version and test each tier's effort value in phase 0 |
| Anthropic's cache-preserving per-message effort change (`role: system`, `content: []`, `output_config`) | Not expressible in OpenAI format; LiteLLM folds `system` messages into top-level `system` | On this path an effort change **costs the messages cache**, keeps tools and system cache. The scorer models this (section 3.1) |
| `x-litellm-trace-id` / `x-litellm-session-id` headers, `x-<vendor>-session-id` fallback | Works; lands in metadata and spend logs | Conversation identity for free when apps send it |
| `cache: DualCache` argument to the hook (in-memory plus Redis) | Works | Home for the per-conversation ledger |
| Built-in beta `auto_router/complexity_router` with a custom `async classify(context)` plugin | Works, but the plugin returns a tier only and the classifier context is the last human ask plus 3 turns at 200 chars each | Too narrow for history-, cache-, and ledger-aware routing; the pre-call hook is the right seam. Its heuristic scorer is a free baseline for the eval |
| Provider switching mid-conversation | LiteLLM translates tool-call formats; `reasoning_content` / thinking blocks from one provider do not transfer to another | Cross-provider switch penalty in the scorer |

## 3. Architecture

```
client app ──► LiteLLM proxy  POST /v1/chat/completions  {model: "jev-auto", messages, tools, stream}
                     │
                     ▼
        JevRouterPlugin.async_pre_call_hook            (the router; all logic lives in a
                     │                                  host-agnostic `jev_router.core` package)
     ┌───────────────┼──────────────────────────┐
     ▼               ▼                          ▼
 Fast-path rules   Jev judge (1 call,        Ledger in LiteLLM DualCache
 (tool-result      ~6 questions,             (incumbent, cache state,
  continuation…)   minimized state)           routing history)
     │               │                          │
     └──────► Deterministic scorer ◄────────────┘
              utility(model, effort) = −cost − quality risk − switch cost
                     │
                     ▼
        rewrites data["model"], data["reasoning_effort"], adds cache_control,
        stamps data["metadata"]["jev_router"] (decision, distributions, predictions)
                     │
                     ▼
        LiteLLM Router ──► Anthropic / OpenAI ──► streamed OpenAI-format response
                     │
                     ▼
        JevRouterPlugin.async_log_success_event   (final usage, streaming or not)
        → ledger update (observed cached tokens, timestamp) → decision log row
```

### 3.1 Components

**`RequestNormalizer`** turns the incoming OpenAI-format request (`data` from
the hook) into an internal `Turn` with: message list, tool definitions, system
prompt, token estimate, and structural signals (last assistant message carried
`tool_calls` and the tail is `role: tool` messages? images present? which tools
were called recently?). It also resolves the conversation id: the
`x-litellm-session-id` / `x-litellm-trace-id` value from metadata when present,
otherwise a hash of the system prompt plus the first user message plus the
LiteLLM `user` field, which is stable across turns because the served history
is append-only.

**`FastPath`** applies hard rules before Jev is called. These are cases where a
semantic judgment is not needed or not allowed:

| Rule | Decision | Why |
|---|---|---|
| The turn is delivering `role: tool` results for pending `tool_calls` | Incumbent model, same effort, Jev skipped | Mid-loop switch breaks thinking-block continuity and the cache; adds no value |
| No incumbent (first turn) and the request is over the small model's context | Filter candidates by context | Capability, not judgment |
| Conversation has no ledger entry and the message is under a configurable size | Default tier from config, Jev still consulted | Cold start |
| Jev unreachable, 429, or timeout (budget 1.5 s) | Incumbent if present, else configured default | Fail open, never block on Jev |

**`StateBuilder`** produces Jev's `state` under the 32k-token budget. It is a
JSON object, not a transcript dump:

```json
{
  "system_prompt": { "summary": "<first 600 chars>", "chars": 5400, "hash": "…" },
  "tools": { "names": ["search", "run_sql", "…"], "count": 12 },
  "history": {
    "turn_count": 23,
    "recent": [
      { "role": "user", "text": "<last 1,500 chars>", "chars": 4200 },
      { "role": "assistant", "text": "<last 800 chars>", "tool_calls": ["run_sql"], "chars": 9100 },
      "..."
    ],
    "older_summary": "<router-maintained rolling summary of turns before `recent`>"
  },
  "current_message": "<full text up to 6,000 chars, tail-truncated with marker>",
  "signals": { "has_image": false, "code_blocks_in_message": 1, "pending_tool_use": false },
  "routing_history": [
    { "turn": 22, "model": "claude-sonnet-5", "effort": "medium", "required_tier": 1.4, "confidence": 0.71, "cache_read_tokens": 18200, "outcome": "ok" },
    { "turn": 21, "model": "claude-sonnet-5", "effort": "medium", "required_tier": 1.2, "confidence": 0.80, "cache_read_tokens": 16900, "outcome": "ok" }
  ]
}
```

The rolling `older_summary` is maintained by the router for Jev's benefit
only. **The message history sent to the serving model is never rewritten**;
Fable 5.1's preserved-thinking check rejects edited history, and any rewrite
would also invalidate the cache. Redaction hooks run here because the state
leaves our boundary for TypeSafe's API.

**`JevJudge`** issues one `system_one` call with all questions. Draft question
set (final wording is tuned in the eval loop, section 5):

| Key | Type | Question | Used for |
|---|---|---|---|
| `required_tier` | score, 4 levels | 0 "trivial: greeting, acknowledgement, lookup a fact just stated"; 1 "routine: clear single-step task a competent assistant handles reliably"; 2 "demanding: multi-step reasoning, non-trivial code, ambiguity to resolve"; 3 "frontier: hard reasoning, long agentic work, high cost of error" | Quality-risk term; the expected score and the full distribution are both used |
| `task_type` | choice | chit_chat, factual_qa, writing, code, math_or_logic, data_analysis, agentic_multistep, other | Per-model strength adjustments; analytics |
| `continues_task` | noul | "The current message continues the task worked on in the recent turns rather than starting a new one." | Stickiness weight: high value makes a downgrade riskier |
| `needs_history` | noul | "Answering this message well requires details from earlier in the conversation." | Also stickiness; a low value plus new task is where a downgrade is cheapest in quality terms |
| `quality_complaint` | noul | "The user is expressing that the previous answer was wrong, incomplete, or low quality." | Forces at least one tier above the incumbent |
| `expected_output` | score, 3 levels | short / medium / long | Output-token cost estimate |

Each answer's distribution and confidence is stored, not just the winner.

**`CacheLedger`** lives in the `DualCache` LiteLLM hands to the hook (Redis
when configured, in-memory otherwise) under `jev_router:<conversation_id>`
with a TTL a little over the provider cache TTL. It stores: incumbent model and
effort, the timestamp of the last request start, the cached-prefix token count
from the last response, the split of that prefix into tools-plus-system versus
messages (estimated from token counts), the provider's TTL, and the last known
total prompt size. It predicts `cached_tokens(m, e)` for each candidate:

| Candidate relative to incumbent | Predicted cached tokens |
|---|---|
| Same model, same effort, within TTL | Full last cached prefix |
| Same model, different effort, within TTL | Tools-plus-system portion only (messages cache is invalidated by an effort change on this path) |
| Different model, any provider | 0 |
| Anything past TTL | 0 |

It is corrected from the real `usage` in `async_log_success_event` after
every call, so drift (an upstream prompt change, a lost cache, an expired TTL)
is caught within one turn, and predicted-versus-observed is logged.

**`Scorer`** is pure code. For each eligible candidate `(m, e)`:

```
input_cost   = uncached_tokens(m) × p_in(m) + cached_tokens(m) × p_cached(m)
             + write_tokens(m) × p_write(m)
output_cost  = expected_output_tokens × p_out(m)
quality_risk = Σ_t P(required_tier = t) × penalty(t − tier(m, e))     # 0 when tier(m,e) ≥ t
switch_cost  = 0 if (m, e) == incumbent
             else base_switch + continuity_weight × P(continues_task) × P(needs_history)
                  + thinking_loss(m ≠ incumbent model)
utility      = −λ_cost × (input_cost + output_cost) − λ_quality × quality_risk − switch_cost
```

`switch_cost` also adds a cross-provider constant when `m` is on a different
provider than the incumbent (tool-call format translation and total loss of
reasoning context). Overrides applied after scoring: `quality_complaint` above
threshold raises the floor to incumbent tier + 1; confidence on
`required_tier` below a threshold collapses the choice to the incumbent (or
default). `λ_cost`, `λ_quality`, the penalty table, and thresholds live in
config and are what the eval loop tunes. Effort-only moves within the
incumbent model keep the tools-plus-system cache and carry no thinking loss,
so they still beat a model switch whenever the tier gap is small.

**Execution** is LiteLLM's. The hook sets `data["model"]` to the tier's
`model_name`, `data["reasoning_effort"]` to the tier's effort (Anthropic tiers
only; OpenAI reasoning models take their own `reasoning_effort` values from
config), and places `cache_control` on the last system content block and on
the last user message so both providers' prefix caches see a stable prefix
and a moving tail. Anthropic thinking stays adaptive and is pinned per tier so
it never varies within a route. Client-supplied `model` values other than the
router alias pass through untouched.

**`JevRouterPlugin`** is the thin LiteLLM `CustomLogger` subclass: it owns no
logic, only wiring. `async_pre_call_hook` calls `core.decide(turn, ledger)`
and applies the result; `async_log_success_event` (which LiteLLM calls once
per request with the assembled response for streaming and non-streaming
alike) calls `core.observe(usage)`; `async_post_call_failure_hook` records
failures so a provider error does not leave a stale incumbent. A shadow flag
makes the hook compute and log the decision but leave `data["model"]` on the
configured default. The same `core` package is importable by the CLI and the
eval harness with no LiteLLM present.

**`DecisionLog`** writes one row per turn: state hash, Jev answers, candidate
utilities, chosen route, predicted vs. observed cache tokens, cost, latency of
Jev and of the model. This is the dataset the eval loop and the dashboard read.

### 3.2 Configuration (phase 1)

LiteLLM's `config.yaml` owns credentials, deployments, and the default cache
injection points; `router.yaml` owns everything the scorer needs. Tier names
in `router.yaml` must match `model_name` entries in `config.yaml`.

```yaml
# config.yaml (LiteLLM)
model_list:
  - model_name: haiku
    litellm_params: { model: anthropic/claude-haiku-4-5, api_key: os.environ/ANTHROPIC_API_KEY }
  - model_name: sonnet
    litellm_params: { model: anthropic/claude-sonnet-5, api_key: os.environ/ANTHROPIC_API_KEY }
  - model_name: opus
    litellm_params: { model: anthropic/claude-opus-5, api_key: os.environ/ANTHROPIC_API_KEY }
  - model_name: gpt-small
    litellm_params: { model: openai/<small model>, api_key: os.environ/OPENAI_API_KEY }
  - model_name: gpt-large
    litellm_params: { model: openai/<frontier model>, api_key: os.environ/OPENAI_API_KEY }
  - model_name: jev-auto           # the alias apps call; the hook rewrites it
    litellm_params: { model: anthropic/claude-sonnet-5, api_key: os.environ/ANTHROPIC_API_KEY }
litellm_settings:
  callbacks: jev_router.litellm_plugin.proxy_handler_instance
  cache: true
  cache_params: { type: redis }    # DualCache backing for the ledger
```

```yaml
# router.yaml (jev_router)
alias: jev-auto
objective: { lambda_cost: 1.0, lambda_quality: 3.0 }     # quality first
default_route: { tier: sonnet-medium }
jev: { model: jev-latest, timeout_ms: 1500, min_confidence: 0.55 }
cache_ttl_seconds: { anthropic: 300, openai: 300 }
switch_cost: { base: 0.002, cross_provider: 0.01, continuity_weight: 0.02, thinking_loss: 0.005 }   # in dollars-equivalent
tiers:                              # ordered by capability; index is the tier number
  - { name: haiku,         model: haiku,     provider: anthropic, effort: null,   price_in: 1.00, price_out: 5.00,  cache_read: 0.10, cache_write: 1.25, min_cache_prefix: 4096, context: 200000 }
  - { name: gpt-small,     model: gpt-small, provider: openai,    effort: low,    price_in: TBD,  price_out: TBD,   cache_read: 0.10, cache_write: 1.00, min_cache_prefix: 1024, context: TBD }
  - { name: sonnet-low,    model: sonnet,    provider: anthropic, effort: low,    price_in: 2.00, price_out: 10.00, cache_read: 0.10, cache_write: 1.25, min_cache_prefix: 1024, context: 1000000 }
  - { name: sonnet-medium, model: sonnet,    provider: anthropic, effort: medium, price_in: 2.00, price_out: 10.00, cache_read: 0.10, cache_write: 1.25, min_cache_prefix: 1024, context: 1000000 }
  - { name: sonnet-high,   model: sonnet,    provider: anthropic, effort: high,   price_in: 2.00, price_out: 10.00, cache_read: 0.10, cache_write: 1.25, min_cache_prefix: 1024, context: 1000000 }
  - { name: gpt-large,     model: gpt-large, provider: openai,    effort: high,   price_in: TBD,  price_out: TBD,   cache_read: 0.10, cache_write: 1.25, min_cache_prefix: 1024, context: TBD }
  - { name: opus-medium,   model: opus,      provider: anthropic, effort: medium, price_in: 5.00, price_out: 25.00, cache_read: 0.10, cache_write: 1.25, min_cache_prefix: 512,  context: 1000000 }
  - { name: opus-xhigh,    model: opus,      provider: anthropic, effort: xhigh,  price_in: 5.00, price_out: 25.00, cache_read: 0.10, cache_write: 1.25, min_cache_prefix: 512,  context: 1000000 }
```

Anthropic prices are first-party rates as of 2026-06. OpenAI entries are
marked TBD: which two OpenAI models to include and their current prices are
filled from OpenAI's price list at implementation time (open question 1). The
tier order across providers is a hypothesis the phase 2 eval checks; Jev's
`required_tier` is defined on the four semantic levels, and the mapping from
those levels to this ladder is config.

### 3.3 Why Jev does not pick the model directly

The brief says "use Jev as the decision maker", and it is: every semantic
judgment (how hard, does it continue, is the user unhappy, how long an answer)
comes from Jev, and the routing history is in Jev's state so it can weigh
continuity itself. What Jev is not asked to do is the arithmetic. Cache savings
are `cached_tokens × (p_in − p_cached)` against a concrete price table; a
System One model has no reasoning chain and TypeSafe explicitly scopes it to
judgments. Putting the numbers in code also makes every decision reproducible
and auditable from the log, which a prose instruction like "prefer the cheapest
adequate model" never is. If the eval shows the composed score beats a direct
`choice` over models, we keep it; the direct form stays available as a
configurable alternative judge for the comparison.

## 4. Delivery phases

### Phase 0: scaffolding and LiteLLM spike (about 2 days)

- `uv`-managed Python 3.12 project, `ruff`, `pyright`, `pytest`; pinned
  `litellm[proxy]` version recorded in the README.
- `Judge` protocol with three implementations: `JevJudge` (typesafe-sdk),
  `AdapterJudge` (system-one-adapter-python, for CI without network),
  `RecordedJudge` (fixtures).
- Spike that de-risks the host: a no-op plugin that rewrites `model` and sets
  `reasoning_effort` on the pinned LiteLLM version, and a script that makes
  two identical requests per tier and asserts `cached_tokens > 0` on the
  second, streaming and non-streaming, for Anthropic and OpenAI. Any tier
  whose effort value LiteLLM rejects is fixed or dropped here.
- Config loader for `router.yaml`, token estimator (LiteLLM's
  `token_counter` for the ledger split, chars/4 for the Jev state budget).

### Phase 1: routing core inside the proxy (about 1 week)

- `jev_router.core`: `RequestNormalizer`, `FastPath`, `StateBuilder` (budget
  enforcement, rolling summary, redaction hook), `JevJudge` with the six
  questions, `CacheLedger`, `Scorer`, `DecisionLog` (SQLite plus JSONL).
- `jev_router.litellm_plugin`: the `CustomLogger` wiring described above,
  shadow flag, and decision stamped into `metadata` so it appears in LiteLLM
  spend logs and any configured observability sink.
- CLI that replays a saved conversation turn by turn against `core` (no
  proxy needed) and prints the decision table: Jev answers, per-tier utility,
  cache prediction versus observed.
- Unit tests for the scorer (cache makes the incumbent win on a routine
  follow-up; quality complaint forces an upgrade; tool-result turns never
  switch; effort-only downgrade beats a model switch for adjacent tiers;
  cross-provider switch needs a larger tier gap than same-provider).
- Integration tests against the running proxy: the two-request cache check
  from phase 0 now through the `jev-auto` alias; a forced model switch shows
  cached tokens drop to zero and the ledger records it.

### Phase 2: evaluation and tuning (about 1 week)

- Eval set of multi-turn conversations: synthetic scripts covering each
  `task_type` × tier, tool-loop transcripts, and adversarial cases (long
  irrelevant history, mid-task topic switch, "that's wrong, try again").
  Label each turn's minimum acceptable tier with a strong-model judge and
  spot-check by hand.
- Shadow mode on the proxy to collect real decisions without acting on them.
- Metrics: routing accuracy against labels, cost per completed conversation
  versus an all-Opus baseline, cache hit rate, switch rate per conversation,
  cross-provider switch rate, Jev p50 and p95 latency, calibration curve of
  `required_tier` confidence.
- Baselines to beat: always-Opus, always-Sonnet, LiteLLM's built-in
  heuristic `complexity_router`, cheapest-eligible rules (jev-router style),
  and the single-`choice`-over-models judge.
- Tune `λ`, penalty table, switch costs, thresholds, and question wording.

### Phase 3: hardening (after phase 2 results)

- Postgres-backed decision log if the proxy already runs one; dashboard over
  the log.
- Gemini tier if wanted (LiteLLM already normalizes its cache usage).
- Optional upstream contribution: a LiteLLM passthrough for Anthropic's
  per-message effort change, which would make effort-only moves free of
  messages-cache loss on Opus 5 and Fable 5.1.

## 5. Proposed repository layout

```
jev_router/
  core/
    config.py        # router.yaml schema, tiers, prices
    normalize.py     # RequestNormalizer, Turn, conversation id derivation
    fastpath.py
    state.py         # StateBuilder, budget, redaction, rolling summary
    judge/
      base.py        # Judge protocol, Judgment dataclass
      jev.py         # typesafe-sdk implementation, the question set
      adapter.py     # system-one-adapter-python fallback (CI)
      recorded.py
    ledger.py        # CacheLedger over a small KV protocol (DualCache or dict)
    scorer.py        # pure functions, fully unit-tested
    log.py           # DecisionLog
    router.py        # decide() / observe() entry points used by every host
  litellm_plugin.py  # CustomLogger wiring; proxy_handler_instance
  cli.py             # replay + explain
deploy/
  config.yaml        # LiteLLM proxy config (models, callbacks, redis)
  router.yaml
eval/
  conversations/     # JSONL eval set
  label.py           # strong-model labeling
  run.py             # metrics + baselines
tests/
docs/
```

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Jev is three days old and gated; API may change or the key may not arrive | `Judge` protocol; develop on the official LLM-backed adapter; OpenRouter and Vercel AI Gateway as alternate access paths |
| Calibration is TypeSafe's claim, not independently measured | Phase 2 calibration curve; confidence gating from day one |
| 32k state budget on long conversations | Rolling summary for Jev only; never touch the served history |
| Conversation content leaves our boundary to TypeSafe | Redaction hook in `StateBuilder`; document it; shadow mode can run on synthetic traffic first |
| Added latency (70–500 ms plus network) on every non-fast-path turn | Fast path skips Jev on tool loops; timeout 1.5 s; fail open |
| Cache prediction drifts from reality (prompt changed upstream, TTL expired) | Ledger corrected from every response's `usage`; predicted vs. observed logged |
| Router accidentally rewrites history (breaks cache, rejected by Fable 5.1) | Served history is append-only by construction; test asserts byte-identical prefix between turns |
| LiteLLM version drift: effort gating, cache accounting, hook behaviour have all regressed before | Pin the version; phase 0 spike is the regression test; upgrade only with it green |
| Apps use LiteLLM's Anthropic-format `/v1/messages`, where the hook is bypassed | Documented as unsupported; requests there get the alias's default deployment with no routing |
| Effort change loses the messages cache on this path | Modelled in the ledger; measured in phase 2; upstream passthrough is the phase 3 fix |
| Ledger lost on proxy restart without Redis | Redis configured in `config.yaml`; in-memory fallback degrades to cold-start routing, never to errors |

## 7. Remaining open questions

Defaults will be built if unanswered.

1. **Which OpenAI models.** Default: one small and one frontier model from the
   current GPT-5.x line, prices taken from OpenAI's price list when the
   config is written. Name them if you have preferences.
2. **Existing LiteLLM deployment.** Is there a proxy already running with a
   `config.yaml` and Redis, and which version? Default: a fresh pinned
   deployment under `deploy/`.
3. **Traffic profile.** Typical gap between turns decides the TTL strategy
   (under 5 minutes keeps the default cache warm on both providers). Default:
   assume interactive traffic, 5-minute TTL, no keep-alives.

## 8. Sources

- TypeSafe launch coverage: [DataCamp](https://www.datacamp.com/blog/system-one-models-jev), [The Rundown](https://www.therundown.ai/news/typesafe-jev-ai-decisions-software), [KuCoin news](https://www.kucoin.com/news/flash/ex-openai-researcher-launches-typesafe-ai-with-non-text-model-jev), [Latent Space AINews](https://www.latent.space/p/ainews-jev-a-system-one-model-that), [Kingy AI review](https://kingy.ai/blog/typesafe-jev-review-the-ai-model-that-doesnt-generate-text/), [Anthony Maio](https://anthonymaio.substack.com/p/jev-the-language-model-that-wont), [Flavio Copes](https://flaviocopes.com/jev/), [Mohammed Shehu guide](https://mohammedshehu.com/jev-typesafe-ai/), [DEV guide](https://dev.to/valyuai/how-to-use-jev-a-practical-guide-to-typesafes-system-one-model-g5e)
- TypeSafe official: [blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [API reference](https://docs.typesafe.ai/api), [quick start](https://docs.typesafe.ai/introduction/quickstart), [jev-1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13), [Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python), [JS SDK](https://github.com/typesafe-ai/typesafe-sdk-js) and [issue #6 on request shapes](https://github.com/typesafe-ai/typesafe-sdk-js/issues/6), [system-one-adapter-python](https://github.com/typesafe-ai/system-one-adapter-python), [skills](https://github.com/typesafe-ai/skills)
- Access paths: [OpenRouter jev-1.13](https://openrouter.ai/typesafe/jev-1.13), [OpenRouter jev-latest](https://openrouter.ai/~typesafe/jev-latest), [Vercel AI Gateway](https://vercel.com/changelog/typesafe-ai-jev-now-available-on-ai-gateway), [LiteLLM pass-through](https://docs.litellm.ai/docs/pass_through/typesafe)
- Patterns and limits: [awesome-jev-by-typesafe](https://github.com/24601/awesome-jev-by-typesafe), [Jev project reference gist](https://gist.github.com/pjburnhill/adf8d28efcad9df037bfdece178ef965), [actionbox review](https://actionbox.cloud/blog/typesafe-ai-jev-review/), [orcarouter](https://www.orcarouter.ai/blog/jev-typesafe-system-one-what-we-know)
- Prior art: [prismhq/jev-router](https://github.com/prismhq/jev-router), [Bifrost proposal](https://github.com/maximhq/bifrost/issues/7278), [9router proposal](https://github.com/decolua/9router/issues/4126), [OmniRoute proposal](https://github.com/diegosouzapw/OmniRoute/issues/13987), [RouteLLM overview](https://neuraltrust.ai/blog/llm-model-routing), [routing survey arXiv 2603.04445](https://arxiv.org/pdf/2603.04445), [SLM front-door routing arXiv 2604.02367](https://arxiv.org/abs/2604.02367), [2026 router comparisons](https://dev.to/artem42/top-llm-routing-tools-in-2026-architectures-benchmarks-and-production-trade-offs-ife)
- LiteLLM: [call hooks](https://docs.litellm.ai/docs/proxy/call_hooks), [prompt caching](https://docs.litellm.ai/docs/completion/prompt_caching), [auto-inject cache checkpoints](https://docs.litellm.ai/docs/tutorials/prompt_caching), [Anthropic effort](https://docs.litellm.ai/docs/providers/anthropic_effort), [request headers](https://docs.litellm.ai/docs/proxy/request_headers), [beta auto routing](https://docs.litellm.ai/docs/proxy/auto_routing), [custom_logger.py](https://github.com/BerriAI/litellm/blob/main/litellm/integrations/custom_logger.py); issues [#30469](https://github.com/BerriAI/litellm/issues/30469) and [#27518](https://github.com/BerriAI/litellm/issues/27518) (hook bypassed on `/v1/messages`), [#11789](https://github.com/BerriAI/litellm/issues/11789) (streaming cache cost, closed), [#41067](https://github.com/BerriAI/litellm/issues/41067) (bridged streaming cache accounting, open), [#25957](https://github.com/BerriAI/litellm/issues/25957) (effort gating lag)
- Caching: [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching), [OpenAI Prompt Caching 201](https://developers.openai.com/cookbook/examples/prompt_caching_201), [Gemini context caching](https://ai.google.dev/gemini-api/docs/caching), [Gemini implicit caching](https://developers.googleblog.com/gemini-2-5-models-now-support-implicit-caching/); Anthropic caching, thinking-block, and effort facts from the Claude API reference bundled with this session
