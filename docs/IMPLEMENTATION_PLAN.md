# Jev-powered model router: research findings and implementation plan

Status: proposal, 2026-09-18. Nothing in this repo is built yet.

## 1. What we are building

A routing layer that sits between end-user applications and LLM providers.
Each incoming turn of a conversation is inspected, TypeSafe's **Jev** model
judges what the turn needs, and deterministic code turns that judgment plus the
conversation's routing history and prompt-cache state into a concrete
`(model, effort)` choice. The chosen model then serves the turn.

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
  therefore `(model, effort)`, not just `model`.

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

## 3. Architecture

```
client app ──► /v1/messages (model: "auto") ──► Router
                                                  │
                     ┌────────────────────────────┼─────────────────────────────┐
                     ▼                            ▼                             ▼
             Fast-path rules             Jev judge (1 call,            Cache ledger +
             (tool-result continuation,  ~6 questions, minimized       routing history
              context overflow, ...)     state)                        (per conversation)
                     │                            │                             │
                     └──────────────► Deterministic scorer ◄────────────────────┘
                                      utility(model, effort) = −cost − quality risk − switch cost
                                                  │
                                                  ▼
                                    Provider adapter (Anthropic first)
                                    appends cache_control, executes, records usage
                                                  │
                                                  ▼
                                    Ledger update + decision log
```

### 3.1 Components

**`RequestNormalizer`** turns the incoming request (Anthropic Messages shape in
phase 1) into an internal `Turn` with: message list, tool definitions, system
prompt, token estimate, and structural signals (last assistant message ended in
`tool_use`? images present? which tools were called recently?).

**`FastPath`** applies hard rules before Jev is called. These are cases where a
semantic judgment is not needed or not allowed:

| Rule | Decision | Why |
|---|---|---|
| The turn is delivering `tool_result` blocks for a pending `tool_use` | Incumbent model, same effort, Jev skipped | Mid-loop switch breaks thinking-block continuity and the cache; adds no value |
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

**`CacheLedger`** stores per conversation: incumbent model and effort, the
timestamp of the last request start, the cached-prefix token count reported by
the last response, the provider's TTL, and the last known total prompt size. It
predicts, for each candidate: `cached_tokens(m)` = last cached prefix if
`m == incumbent` and `now − last_start < ttl`, else 0. It is corrected from
actual `usage` after every call, so drift (an upstream prompt change, a lost
cache) is caught within one turn.

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

Overrides applied after scoring: `quality_complaint` above threshold raises the
floor to incumbent tier + 1; confidence on `required_tier` below a threshold
collapses the choice to the incumbent (or default). `λ_cost`, `λ_quality`, the
penalty table, and thresholds live in config and are what the eval loop tunes.
Effort-only moves within the incumbent model carry no cache cost and no
thinking loss, so they win whenever the tier gap is small, which is the
behaviour Anthropic's cost guidance predicts.

**`ProviderAdapter`** executes the call. Phase 1 is Anthropic only via the
official SDK: streaming, adaptive thinking, one explicit `cache_control`
breakpoint on the last static system block plus top-level automatic caching
for the growing tail, effort pinned per route, `fallbacks: "default"` for
refusal handling on Fable 5.1 / Opus 5, and the effort-change system message
when the router changes effort on Opus 5 / Fable 5.1. The adapter returns the
response and its `usage`, which feeds the ledger.

**`DecisionLog`** writes one row per turn: state hash, Jev answers, candidate
utilities, chosen route, predicted vs. observed cache tokens, cost, latency of
Jev and of the model. This is the dataset the eval loop and the dashboard read.

### 3.2 Candidate configuration (phase 1)

```yaml
objective: { lambda_cost: 1.0, lambda_quality: 3.0 }
default_route: { model: claude-sonnet-5, effort: medium }
jev: { model: jev-latest, timeout_ms: 1500, min_confidence: 0.55 }
cache: { anthropic_ttl_seconds: 300 }
tiers:                       # ordered; index is the tier number
  - { model: claude-haiku-4-5, effort: null,    price_in: 1.00, price_out: 5.00,  cache_read_mult: 0.10, tools: true, vision: true, context: 200000 }
  - { model: claude-sonnet-5,  effort: low,     price_in: 2.00, price_out: 10.00, cache_read_mult: 0.10, tools: true, vision: true, context: 1000000 }
  - { model: claude-sonnet-5,  effort: high,    price_in: 2.00, price_out: 10.00, cache_read_mult: 0.10, tools: true, vision: true, context: 1000000 }
  - { model: claude-opus-5,    effort: medium,  price_in: 5.00, price_out: 25.00, cache_read_mult: 0.10, tools: true, vision: true, context: 1000000 }
  - { model: claude-opus-5,    effort: xhigh,   price_in: 5.00, price_out: 25.00, cache_read_mult: 0.10, tools: true, vision: true, context: 1000000 }
```

Prices are the first-party Anthropic rates as of 2026-06; the config is the
single place they live. Fable 5.1 can be appended as a top tier once the
account has 30-day retention (it is not served under zero-data-retention).

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

### Phase 0: scaffolding (about 1 day)

- `uv`-managed Python 3.12 project, `ruff`, `pyright`, `pytest`.
- `Judge` protocol with three implementations: `JevJudge` (typesafe-sdk),
  `AdapterJudge` (system-one-adapter-python backed by Claude, for development
  before the early-access key arrives), `RecordedJudge` (fixtures for tests).
- Config loader for the YAML above; price table; token estimator (Anthropic
  `count_tokens` for real numbers, a chars/4 heuristic for the Jev state budget).

### Phase 1: routing core and Anthropic execution (about 1 week)

- `RequestNormalizer`, `FastPath`, `StateBuilder` (with budget enforcement and
  redaction hook), `JevJudge` with the six questions, `CacheLedger` (SQLite),
  `Scorer`, Anthropic `ProviderAdapter`, `DecisionLog` (SQLite, JSONL export).
- FastAPI service exposing `POST /v1/messages` that accepts the Anthropic
  Messages request shape with `model: "auto"` and a required
  `conversation_id` (header or metadata), streams the response, and adds a
  `x-router-decision` header plus a `GET /decisions/{conversation_id}` endpoint.
- CLI that replays a saved conversation turn by turn and prints the decision
  table (Jev answers, utilities, cache prediction vs. observed).
- Unit tests for the scorer (cache makes incumbent win on a routine follow-up;
  a quality complaint forces an upgrade; tool-result turns never switch;
  effort-only downgrade preferred over model downgrade when tiers are adjacent).
- Integration test: two identical requests show `cache_read_input_tokens > 0`
  on the second, and a forced model switch shows it drop to zero.

### Phase 2: evaluation and tuning (about 1 week)

- Build an eval set of multi-turn conversations: synthetic scripts covering
  each `task_type` × tier, plus tool-loop transcripts, plus adversarial cases
  (long irrelevant history, mid-task topic switch, "that's wrong, try again").
  Label each turn's minimum acceptable tier with a strong-model judge and
  spot-check by hand.
- Shadow mode: the service routes with the default tier but logs what the Jev
  path would have chosen. Lets us collect real traffic decisions without risk.
- Metrics: routing accuracy against labels, cost per completed conversation vs.
  an all-Opus baseline, cache hit rate, switch rate per conversation, Jev p50
  and p95 latency, calibration curve of `required_tier` confidence.
- Baselines to beat: always-Opus, always-Sonnet, cheapest-eligible rules
  (jev-router style), and the single-`choice`-over-models judge.
- Tune `λ`, penalty table, thresholds, and question wording against the set.

### Phase 3: multi-provider (optional, after phase 2 results)

- OpenAI and Gemini adapters with their own cache predictors (automatic
  prefix caching, no breakpoints). Cross-provider switches carry an extra
  format-translation cost (tool-call shapes differ, thinking blocks do not
  transfer at all), which becomes a config constant in `switch_cost`.
- Per-provider price tables and TTL semantics in the ledger.

## 5. Proposed repository layout

```
jev_router/
  config.py          # YAML schema, price table, tiers
  normalize.py       # RequestNormalizer, Turn
  fastpath.py
  state.py           # StateBuilder, budget, redaction, rolling summary
  judge/
    base.py          # Judge protocol, Judgment dataclass
    jev.py           # typesafe-sdk implementation, the question set
    adapter.py       # system-one-adapter-python fallback
    recorded.py
  ledger.py          # CacheLedger + RoutingHistory (SQLite)
  scorer.py          # pure functions, fully unit-tested
  providers/
    base.py
    anthropic.py
  service.py         # FastAPI /v1/messages
  cli.py             # replay + explain
  log.py             # DecisionLog
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
| Effort-change beta unavailable on a route | Adapter falls back to top-level effort and the ledger records the messages-cache loss |

## 7. Open questions for you

Answers change the plan; defaults are what will be built if unanswered.

1. **Language and shape.** Default: Python 3.12 library plus a FastAPI service
   speaking the Anthropic Messages API with `model: "auto"`. Alternative: a
   TypeScript service, or an OpenAI-compatible `/v1/chat/completions` front so
   existing apps drop in without changes.
2. **Candidate models.** Default: Anthropic first-party only in phases 1–2
   (Haiku 4.5, Sonnet 5, Opus 5, optionally Fable 5.1). Do you want OpenAI or
   Gemini candidates from the start? That pulls phase 3 forward and adds
   format-translation work.
3. **Jev access.** Do you already have a `TYPESAFE_API_KEY`, or should phase 0
   run on the LLM-backed adapter and OpenRouter until it arrives?
4. **Objective.** Default is "quality first, cost second" (`λ_quality` = 3 ×
   `λ_cost`). If this is a cost-driven product, say so and the defaults flip.
5. **Conversation identity.** The router needs a stable `conversation_id` per
   thread to keep the ledger. Can the calling apps supply one, or should the
   router derive it from a hash of the first user message plus system prompt?
6. **Privacy.** Is sending a minimized, redacted transcript to TypeSafe
   acceptable for the test product, or must Jev see only structural signals?
7. **Traffic profile.** Typical gap between turns matters for the TTL choice
   (under 5 minutes keeps the default cache warm; 5–60 minutes justifies the
   1-hour TTL or a keep-alive). Any data on this?

## 8. Sources

- TypeSafe launch coverage: [DataCamp](https://www.datacamp.com/blog/system-one-models-jev), [The Rundown](https://www.therundown.ai/news/typesafe-jev-ai-decisions-software), [KuCoin news](https://www.kucoin.com/news/flash/ex-openai-researcher-launches-typesafe-ai-with-non-text-model-jev), [Latent Space AINews](https://www.latent.space/p/ainews-jev-a-system-one-model-that), [Kingy AI review](https://kingy.ai/blog/typesafe-jev-review-the-ai-model-that-doesnt-generate-text/), [Anthony Maio](https://anthonymaio.substack.com/p/jev-the-language-model-that-wont), [Flavio Copes](https://flaviocopes.com/jev/), [Mohammed Shehu guide](https://mohammedshehu.com/jev-typesafe-ai/), [DEV guide](https://dev.to/valyuai/how-to-use-jev-a-practical-guide-to-typesafes-system-one-model-g5e)
- TypeSafe official: [blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [API reference](https://docs.typesafe.ai/api), [quick start](https://docs.typesafe.ai/introduction/quickstart), [jev-1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13), [Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python), [JS SDK](https://github.com/typesafe-ai/typesafe-sdk-js) and [issue #6 on request shapes](https://github.com/typesafe-ai/typesafe-sdk-js/issues/6), [system-one-adapter-python](https://github.com/typesafe-ai/system-one-adapter-python), [skills](https://github.com/typesafe-ai/skills)
- Access paths: [OpenRouter jev-1.13](https://openrouter.ai/typesafe/jev-1.13), [OpenRouter jev-latest](https://openrouter.ai/~typesafe/jev-latest), [Vercel AI Gateway](https://vercel.com/changelog/typesafe-ai-jev-now-available-on-ai-gateway), [LiteLLM pass-through](https://docs.litellm.ai/docs/pass_through/typesafe)
- Patterns and limits: [awesome-jev-by-typesafe](https://github.com/24601/awesome-jev-by-typesafe), [Jev project reference gist](https://gist.github.com/pjburnhill/adf8d28efcad9df037bfdece178ef965), [actionbox review](https://actionbox.cloud/blog/typesafe-ai-jev-review/), [orcarouter](https://www.orcarouter.ai/blog/jev-typesafe-system-one-what-we-know)
- Prior art: [prismhq/jev-router](https://github.com/prismhq/jev-router), [Bifrost proposal](https://github.com/maximhq/bifrost/issues/7278), [9router proposal](https://github.com/decolua/9router/issues/4126), [OmniRoute proposal](https://github.com/diegosouzapw/OmniRoute/issues/13987), [RouteLLM overview](https://neuraltrust.ai/blog/llm-model-routing), [routing survey arXiv 2603.04445](https://arxiv.org/pdf/2603.04445), [SLM front-door routing arXiv 2604.02367](https://arxiv.org/abs/2604.02367), [2026 router comparisons](https://dev.to/artem42/top-llm-routing-tools-in-2026-architectures-benchmarks-and-production-trade-offs-ife)
- Caching: [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching), [OpenAI Prompt Caching 201](https://developers.openai.com/cookbook/examples/prompt_caching_201), [Gemini context caching](https://ai.google.dev/gemini-api/docs/caching), [Gemini implicit caching](https://developers.googleblog.com/gemini-2-5-models-now-support-implicit-caching/); Anthropic caching, thinking-block, and effort facts from the Claude API reference bundled with this session
