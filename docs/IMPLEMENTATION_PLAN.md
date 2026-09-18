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
| OpenAI models | The GPT-5.6 stack: `gpt-5.6-luna` (small), `gpt-5.6-terra` (mid), `gpt-5.6-sol` (frontier); all priced in LiteLLM's registry |
| Deployment | No proxy exists yet; this repo ships a `deploy/docker-compose.yml` with LiteLLM proxy plus Redis, pinned to the version the plan was verified against (`litellm` 1.101.0) |
| Turn gap | Unknown and use-case dependent, so nothing in the design assumes one; the ledger measures it and the phase 2 report decides TTL strategy per use case |
| Scope rule | **LiteLLM owns every transaction.** Ingress formats, provider translation, streaming, retries, fallbacks, caching directives, pricing, token counting, cost logging. The router adds only what LiteLLM cannot do: ask Jev, remember past decisions per conversation, and pick the `(model, effort)` |

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
| `system-one-adapter-python` is an official drop-in `TypeSafeClient` replacement backed by OpenAI or Anthropic | Not needed (keys are in hand); noted as an escape hatch if Jev access ever lapses |

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
| `litellm.cost_per_token(model, prompt_tokens, completion_tokens, cache_read_input_tokens=, cache_creation_input_tokens=)` | Works; cache-aware, both providers | The scorer prices candidates with it; **no price table in our config** |
| `litellm.get_model_info(model)` → `input_cost_per_token`, `cache_read_input_token_cost`, `cache_creation_input_token_cost`, `max_input_tokens`, `supports_function_calling`, `supports_vision`, `supports_prompt_caching`, `prompt_cache_min_tokens` | Works; registry already lists Opus 5, Sonnet 5, Haiku 4.5 and the GPT-5.6 stack (luna, terra, sol) with prices and cache rates | Eligibility filtering and cache minimums come from the registry; **no capability table in our config** |
| `litellm.token_counter(model=, messages=, tools=)` | Works | Used for the Jev state budget and the stable-prefix estimate |
| `response_cost` in the success-callback `kwargs`; `async_log_success_event` fires once per request with the assembled response, streaming included | Works | The router never computes cost for logging |
| `data["litellm_session_id"]` populated before hooks from `x-litellm-session-id` / `x-litellm-trace-id`; a random UUID when absent | Works | Use when the client supplied it; otherwise derive a stable id ourselves |
| Responses API `/v1/responses` | Goes through the same shared request processor as chat completions (call type `aresponses`) | Expected to work; confirmed in the phase 0 spike |
| Requests to the `jev-auto` alias whose chosen deployment fails | LiteLLM Router `fallbacks` in `config.yaml` | The ledger records the model that actually served, from the success callback |

## 3. Architecture

```
client ──► LiteLLM proxy ──► /v1/chat/completions or /v1/responses   {model: "jev-auto", ...}
                │
                ▼
   JevRouterPlugin.async_pre_call_hook(user_api_key_dict, cache, data, call_type)
                │
                │   reads   data["messages"], data["tools"], data["litellm_session_id"], data["metadata"]
                │   ledger  cache.get("jev_router:<conversation_id>")
                │   asks    Jev (one call, ~6 questions, minimized state)      ← skipped on tool-result turns
                │   prices  litellm.cost_per_token(...) per tier, with predicted cached tokens
                │   writes  data["model"], data["reasoning_effort"], data["metadata"]["jev_router"]
                ▼
   LiteLLM does everything else: translation, cache_control injection, streaming,
   retries, fallbacks, spend logging, provider call
                │
                ▼
   JevRouterPlugin.async_log_success_event(kwargs, response_obj, start_time, end_time)
                │
                │   reads   response_obj.model, response_obj.usage (cached_tokens,
                │           cache_creation_input_tokens), kwargs["response_cost"]
                │   writes  ledger: incumbent (model, effort), observed cached tokens, timestamp
                ▼
```

The plugin changes exactly three fields on the way in and reads one usage
object on the way out. It never edits `messages` or `tools`.

### 3.1 What LiteLLM owns versus what the router owns

| Concern | Owner | How |
|---|---|---|
| Ingress formats (chat completions, Responses API), provider translation, streaming, retries, fallbacks | LiteLLM | Standard proxy config; nothing in the router |
| Cache breakpoints for Anthropic (and Gemini if added) | LiteLLM | `cache_control_injection_points` on each deployment: system block plus last message |
| Prices, cache read and write rates, context windows, tool and vision support | LiteLLM | `get_model_info` and `cost_per_token` |
| Token counts | LiteLLM | `token_counter` |
| Cost per request in logs | LiteLLM | `response_cost` in the success callback |
| Conversation id when the client sends a session header | LiteLLM | `data["litellm_session_id"]` |
| Conversation id when the client sends nothing | Router | SHA-256 of system prompt plus first user message plus the `user` field; stable because served history is append-only |
| Reading the request for signals | Router | Pure functions over `data["messages"]` and `data["tools"]` |
| Building Jev's state and asking Jev | Router | `StateBuilder`, `JevJudge` |
| Remembering the incumbent and cache state per conversation | Router | `CacheLedger` over LiteLLM's `DualCache` |
| Choosing `(model, effort)` | Router | `Scorer` |
| Decision record | Router writes, LiteLLM stores | Stamped into `data["metadata"]["jev_router"]`, which LiteLLM carries into spend logs and any configured logging integration; optional JSONL for local runs |

### 3.2 Router components

**`FastPath`.** Hard rules before Jev:

| Rule | Decision |
|---|---|
| The tail of `messages` is `role: tool` results for pending `tool_calls` | Incumbent model and effort; Jev skipped |
| Prompt tokens exceed a tier's `max_input_tokens` | That tier is ineligible |
| Jev unreachable, 429, or over the 1.5 s budget | Incumbent if present, else the default tier; fail open |
| No ledger entry (first turn) | Jev consulted; no incumbent, so no cache term |

**`StateBuilder`.** Produces Jev's `state` under the 32k-token budget as a
JSON object: system prompt summary and length, tool names and count, the
last few turns with per-turn truncation, a router-maintained rolling summary
of older turns, the full current message up to a cap, structural signals,
and the last five ledger entries. Redaction runs here, since this object
leaves our boundary for TypeSafe. The rolling summary exists only for Jev;
the messages LiteLLM forwards to the model are untouched.

**`JevJudge`.** One `system_one` call with six questions, evaluated in parallel:

| Key | Type | Question | Used for |
|---|---|---|---|
| `required_tier` | score, 4 levels | 0 trivial · 1 routine · 2 demanding · 3 frontier | Quality-risk term; full distribution kept |
| `task_type` | choice | chit_chat, factual_qa, writing, code, math_or_logic, data_analysis, agentic_multistep, other | Per-tier strength adjustments; analytics |
| `continues_task` | noul | Continues the task from recent turns rather than starting a new one | Stickiness weight |
| `needs_history` | noul | Answering well requires details from earlier in the conversation | Stickiness weight |
| `quality_complaint` | noul | User says the previous answer was wrong, incomplete, or poor | Forces a tier above the incumbent |
| `expected_output` | score, 3 levels | short / medium / long | Output-token estimate for pricing |

**`CacheLedger`.** Key `jev_router:<conversation_id>` in the `DualCache`
LiteLLM passes to the hook, TTL slightly over the provider cache TTL. Fields:
incumbent model and effort, last request start time, observed cached tokens
from the last response, estimated stable-prefix tokens (system plus tools,
via `token_counter`), and the last five decisions. Prediction per candidate:

| Candidate relative to incumbent | Predicted cached tokens |
|---|---|
| Same model and effort, within TTL | Last observed cached tokens |
| Same model, different effort, within TTL | `min(observed, stable_prefix)`; an effort change invalidates the messages cache on this path |
| Different model, or past TTL | 0 |

Corrected from the real usage after every response, so a wrong prediction
lasts one turn and is logged as predicted-versus-observed.

**`Scorer`.** Pure code over the tier list. For each eligible tier:

```
input_cost, output_cost = litellm.cost_per_token(
    model, prompt_tokens - cached, expected_output_tokens,
    cache_read_input_tokens=cached, cache_creation_input_tokens=prompt_tokens - cached)
quality_risk = Σ_t P(required_tier = t) × penalty[t − rank(tier)]      # 0 when rank ≥ t
switch_cost  = 0 if tier == incumbent
             else base + continuity × P(continues_task) × P(needs_history)
                  + thinking_loss × [model ≠ incumbent model]
                  + cross_provider × [provider ≠ incumbent provider]
utility      = −λ_cost × (input_cost + output_cost) − λ_quality × quality_risk − switch_cost
```

Overrides: `quality_complaint` above threshold raises the floor to the
incumbent's rank plus one; `required_tier` confidence below threshold
collapses to the incumbent or default. Quality first means `λ_quality` is
several times `λ_cost`.

**`JevRouterPlugin`.** The `CustomLogger` subclass; wiring only.
`async_pre_call_hook` returns early unless `data["model"]` is the alias,
then calls `core.decide` and applies the result. `async_log_success_event`
calls `core.observe`. `async_post_call_failure_hook` clears the incumbent so
a provider error never leaves a stale entry. A `shadow: true` config flag
logs the decision without changing `data["model"]`.

### 3.3 Configuration

```yaml
# deploy/config.yaml (LiteLLM)
model_list:
  - model_name: haiku
    litellm_params: { model: anthropic/claude-haiku-4-5, api_key: os.environ/ANTHROPIC_API_KEY }
    cache_control_injection_points: [{ location: message, role: system }, { location: message, index: -1 }]
  - model_name: sonnet
    litellm_params: { model: anthropic/claude-sonnet-5, api_key: os.environ/ANTHROPIC_API_KEY }
    cache_control_injection_points: [{ location: message, role: system }, { location: message, index: -1 }]
  - model_name: opus
    litellm_params: { model: anthropic/claude-opus-5, api_key: os.environ/ANTHROPIC_API_KEY }
    cache_control_injection_points: [{ location: message, role: system }, { location: message, index: -1 }]
  - model_name: luna
    litellm_params: { model: openai/gpt-5.6-luna, api_key: os.environ/OPENAI_API_KEY }
  - model_name: terra
    litellm_params: { model: openai/gpt-5.6-terra, api_key: os.environ/OPENAI_API_KEY }
  - model_name: sol
    litellm_params: { model: openai/gpt-5.6-sol, api_key: os.environ/OPENAI_API_KEY }
  - model_name: jev-auto                       # alias apps call; the hook rewrites it
    litellm_params: { model: anthropic/claude-sonnet-5, api_key: os.environ/ANTHROPIC_API_KEY }
router_settings:
  fallbacks: [{ opus: [sonnet] }, { sol: [sonnet] }, { terra: [sonnet] }, { luna: [haiku] }]
litellm_settings:
  callbacks: jev_router.litellm_plugin.proxy_handler_instance
  cache: true
  cache_params: { type: redis }
```

```yaml
# deploy/router.yaml (the only router-specific config)
alias: jev-auto
shadow: false
objective: { lambda_cost: 1.0, lambda_quality: 3.0 }
jev: { model: jev-latest, timeout_ms: 1500, min_confidence: 0.55 }
switch_cost: { base: 0.002, continuity: 0.02, thinking_loss: 0.005, cross_provider: 0.01 }   # dollar-equivalent
tier_penalty: [0, 0.05, 0.20, 0.60]        # cost of being 0, 1, 2, 3 ranks below what Jev thinks is needed
tiers:                                     # ordered by capability; everything else comes from litellm.get_model_info
  - { name: haiku,         model: haiku,  effort: null }
  - { name: luna,          model: luna,   effort: low }
  - { name: sonnet-low,    model: sonnet, effort: low }
  - { name: sonnet-medium, model: sonnet, effort: medium }
  - { name: terra,         model: terra,  effort: medium }
  - { name: sonnet-high,   model: sonnet, effort: high }
  - { name: sol,           model: sol,    effort: high }
  - { name: opus-medium,   model: opus,   effort: medium }
  - { name: opus-xhigh,    model: opus,   effort: xhigh }
default: sonnet-medium
```

Registry prices as of `litellm` 1.101.0, per million tokens, input / output /
cached input:

| Tier model | Input | Output | Cached input |
|---|---|---|---|
| claude-haiku-4-5 | 1.00 | 5.00 | 0.10 |
| gpt-5.6-luna | 0.20 | 1.20 | 0.02 |
| claude-sonnet-5 | 2.00 | 10.00 | 0.20 |
| gpt-5.6-terra | 2.00 | 12.00 | 0.20 |
| gpt-5.6-sol | 4.00 | 20.00 | 0.40 |
| claude-opus-5 | 5.00 | 25.00 | 0.50 |

The cross-provider order in the ladder is a hypothesis the phase 2 eval
checks; the scorer only needs the ranks to be monotone in capability.

### 3.3.1 Turn gaps and cache TTLs

The gap between turns is unknown and will differ by use case, so the design
does not assume one. Both providers' caches expire about five minutes after
the last use, and the ledger stores the last request start time, so a
candidate is predicted to have cached tokens only when the gap is inside the
TTL. After a longer gap there is no cache to protect and the stickiness term
drops to its base value, which is the right behaviour: a fresh decision on
quality and price alone. The ledger also records the observed gap for every
turn, and the phase 2 report includes the gap distribution per traffic
source. If a use case turns out to sit mostly in the 5–60 minute band, the
per-deployment fix is Anthropic's 1-hour cache TTL on the injected
`cache_control` for that route, which is a `config.yaml` change and no
router code.

### 3.4 Why Jev does not pick the model directly

Every semantic judgment comes from Jev, and the routing history is in Jev's
state so it can weigh continuity itself. Jev is not asked to do arithmetic:
cache savings are token counts times LiteLLM's price table, and a System One
model has no reasoning chain. Keeping the numbers in code makes every
decision reproducible from the log. A direct `choice` over tiers stays
available as an alternative judge for the eval comparison.

## 4. Delivery phases

### Phase 0: LiteLLM spike (about 1 day)

- `uv` project, `litellm[proxy]` pinned to 1.101.0, `typesafe-sdk`, `ruff`, `pyright`, `pytest`.
- `deploy/docker-compose.yml`: LiteLLM proxy (same pinned image tag) with
  `deploy/config.yaml` mounted, Redis, and an env file for the three API
  keys. `docker compose up` is the whole deployment.
- A no-op plugin that rewrites `model` and sets `reasoning_effort` for each
  tier, run against the pinned proxy on both `/v1/chat/completions` and
  `/v1/responses`, streaming and not.
- A script that sends two identical requests per tier and asserts
  `prompt_tokens_details.cached_tokens > 0` on the second, for Anthropic and
  OpenAI, and that `response_cost` arrives in the success callback. Any tier
  whose effort value LiteLLM rejects is fixed or dropped here.

### Phase 1: routing core (about 4 days)

- `jev_router.core`: `fastpath`, `state`, `judge` (`JevJudge`,
  `RecordedJudge` for tests), `ledger`, `scorer`, `router.decide` and
  `router.observe`.
- `jev_router.litellm_plugin` with the shadow flag and metadata stamping.
- CLI that replays a saved request through `decide` with a recorded judge
  and prints the per-tier utilities and cache prediction.
- Unit tests on the scorer: incumbent wins a routine follow-up when cached;
  quality complaint forces an upgrade; tool-result turns never switch;
  effort-only downgrade beats a model switch for adjacent tiers;
  cross-provider needs a larger gap than same-provider.
- Integration test through the proxy: the phase 0 cache check via the
  `jev-auto` alias, plus a forced switch showing cached tokens drop to zero
  and the ledger recording it.

### Phase 2: evaluation and tuning (about 1 week)

- Multi-turn eval set: each `task_type` × tier, tool-loop transcripts,
  adversarial cases (long irrelevant history, mid-task topic switch, "that's
  wrong, try again"); minimum acceptable tier labelled by a strong model and
  spot-checked.
- Shadow mode on the proxy for real decisions.
- Metrics: routing accuracy, cost per completed conversation versus
  all-Opus, cache hit rate, switch rate, cross-provider switch rate, Jev p50
  and p95 latency, calibration of `required_tier` confidence.
- Baselines: always-Opus, always-Sonnet, LiteLLM's built-in heuristic
  `complexity_router`, cheapest-eligible, and the direct-choice judge.
- Tune `λ`, `tier_penalty`, `switch_cost`, thresholds, question wording.

### Phase 3: only if phase 2 asks for it

- Postgres spend logs as the decision store, dashboard over them.
- Gemini tier (LiteLLM already normalizes its cache usage).
- Upstream LiteLLM contributions if they matter: pre-call hooks on
  `/v1/messages`; Anthropic's per-message effort change.

## 5. Repository layout

```
jev_router/
  core/
    config.py        # router.yaml schema
    fastpath.py
    state.py         # StateBuilder, budget, redaction, rolling summary
    judge.py         # Judge protocol, JevJudge, RecordedJudge, the question set
    ledger.py        # CacheLedger over a get/set protocol (DualCache or dict)
    scorer.py        # pure functions; calls litellm.cost_per_token / get_model_info
    router.py        # decide() / observe()
  litellm_plugin.py  # CustomLogger wiring; proxy_handler_instance
  cli.py             # replay + explain
deploy/
  docker-compose.yml # LiteLLM proxy + Redis, pinned
  config.yaml        # LiteLLM proxy config
  router.yaml
  .env.example       # ANTHROPIC_API_KEY, OPENAI_API_KEY, TYPESAFE_API_KEY
eval/
  conversations/     # JSONL eval set
  label.py
  run.py
tests/
docs/
```

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Jev is days old and gated; API may change | `Judge` protocol; `RecordedJudge` keeps tests independent of the network |
| Calibration is TypeSafe's claim | Phase 2 calibration curve; confidence gating from day one |
| 32k state budget on long conversations | Rolling summary for Jev only; served messages untouched |
| Conversation content leaves our boundary to TypeSafe | Redaction hook in `StateBuilder`; shadow mode on synthetic traffic first |
| Added latency on every non-fast-path turn | Fast path skips Jev on tool loops; 1.5 s timeout; fail open |
| Cache prediction drifts | Ledger corrected from every response's usage; predicted-versus-observed logged |
| LiteLLM version drift (effort gating, cache accounting, hook coverage have regressed before) | Pinned version; the phase 0 spike is the regression test; upgrade only with it green |
| Apps call LiteLLM's Anthropic-format `/v1/messages`, where hooks are bypassed upstream | Documented as unrouted; the alias's default deployment serves them; fix belongs in LiteLLM, not here |
| Effort change loses the messages cache on this path | Modelled in the ledger; measured in phase 2 |
| No Redis | In-memory `DualCache` still works per process; cold-start routing after restart, never errors |
| Turn gaps longer than the cache TTL make caching moot for that traffic | Expected and handled: no cache term, fresh decision; gap distribution reported in phase 2 to pick TTLs per route |

## 7. Open questions

None outstanding. Everything the plan depends on is recorded in section 0.
The first decisions the build itself will surface are the eval results in
phase 2: the cross-provider tier order, the `λ` weights, and whether any
route needs the 1-hour cache TTL.

## 8. Sources

- TypeSafe launch coverage: [DataCamp](https://www.datacamp.com/blog/system-one-models-jev), [The Rundown](https://www.therundown.ai/news/typesafe-jev-ai-decisions-software), [KuCoin news](https://www.kucoin.com/news/flash/ex-openai-researcher-launches-typesafe-ai-with-non-text-model-jev), [Latent Space AINews](https://www.latent.space/p/ainews-jev-a-system-one-model-that), [Kingy AI review](https://kingy.ai/blog/typesafe-jev-review-the-ai-model-that-doesnt-generate-text/), [Anthony Maio](https://anthonymaio.substack.com/p/jev-the-language-model-that-wont), [Flavio Copes](https://flaviocopes.com/jev/), [Mohammed Shehu guide](https://mohammedshehu.com/jev-typesafe-ai/), [DEV guide](https://dev.to/valyuai/how-to-use-jev-a-practical-guide-to-typesafes-system-one-model-g5e)
- TypeSafe official: [blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [API reference](https://docs.typesafe.ai/api), [quick start](https://docs.typesafe.ai/introduction/quickstart), [jev-1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13), [Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python), [JS SDK](https://github.com/typesafe-ai/typesafe-sdk-js) and [issue #6 on request shapes](https://github.com/typesafe-ai/typesafe-sdk-js/issues/6), [system-one-adapter-python](https://github.com/typesafe-ai/system-one-adapter-python), [skills](https://github.com/typesafe-ai/skills)
- Access paths: [OpenRouter jev-1.13](https://openrouter.ai/typesafe/jev-1.13), [OpenRouter jev-latest](https://openrouter.ai/~typesafe/jev-latest), [Vercel AI Gateway](https://vercel.com/changelog/typesafe-ai-jev-now-available-on-ai-gateway), [LiteLLM pass-through](https://docs.litellm.ai/docs/pass_through/typesafe)
- Patterns and limits: [awesome-jev-by-typesafe](https://github.com/24601/awesome-jev-by-typesafe), [Jev project reference gist](https://gist.github.com/pjburnhill/adf8d28efcad9df037bfdece178ef965), [actionbox review](https://actionbox.cloud/blog/typesafe-ai-jev-review/), [orcarouter](https://www.orcarouter.ai/blog/jev-typesafe-system-one-what-we-know)
- Prior art: [prismhq/jev-router](https://github.com/prismhq/jev-router), [Bifrost proposal](https://github.com/maximhq/bifrost/issues/7278), [9router proposal](https://github.com/decolua/9router/issues/4126), [OmniRoute proposal](https://github.com/diegosouzapw/OmniRoute/issues/13987), [RouteLLM overview](https://neuraltrust.ai/blog/llm-model-routing), [routing survey arXiv 2603.04445](https://arxiv.org/pdf/2603.04445), [SLM front-door routing arXiv 2604.02367](https://arxiv.org/abs/2604.02367), [2026 router comparisons](https://dev.to/artem42/top-llm-routing-tools-in-2026-architectures-benchmarks-and-production-trade-offs-ife)
- LiteLLM: [call hooks](https://docs.litellm.ai/docs/proxy/call_hooks), [prompt caching](https://docs.litellm.ai/docs/completion/prompt_caching), [auto-inject cache checkpoints](https://docs.litellm.ai/docs/tutorials/prompt_caching), [Anthropic effort](https://docs.litellm.ai/docs/providers/anthropic_effort), [request headers](https://docs.litellm.ai/docs/proxy/request_headers), [beta auto routing](https://docs.litellm.ai/docs/proxy/auto_routing), [custom_logger.py](https://github.com/BerriAI/litellm/blob/main/litellm/integrations/custom_logger.py); issues [#30469](https://github.com/BerriAI/litellm/issues/30469) and [#27518](https://github.com/BerriAI/litellm/issues/27518) (hook bypassed on `/v1/messages`), [#11789](https://github.com/BerriAI/litellm/issues/11789) (streaming cache cost, closed), [#41067](https://github.com/BerriAI/litellm/issues/41067) (bridged streaming cache accounting, open), [#25957](https://github.com/BerriAI/litellm/issues/25957) (effort gating lag)
- Caching: [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching), [OpenAI Prompt Caching 201](https://developers.openai.com/cookbook/examples/prompt_caching_201), [Gemini context caching](https://ai.google.dev/gemini-api/docs/caching), [Gemini implicit caching](https://developers.googleblog.com/gemini-2-5-models-now-support-implicit-caching/); Anthropic caching, thinking-block, and effort facts from the Claude API reference bundled with this session
