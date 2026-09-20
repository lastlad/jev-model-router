# Jev model router: design and implementation plan v0

**Status: completed.** Proposed 2026-09-18, implemented and evaluated by
2026-09-20. This is the design record: why the router looks the way it does,
and what the first evaluation changed. The README describes how to run it;
`deploy/routers/gpt.yaml` documents every tunable. Later changes (several
routers per proxy, Claude Code on a subscription) are not folded back into
this document.

## 1. What was built

A routing layer between applications and LLM providers. Applications call a
LiteLLM proxy with `model: "jev-auto"`. For each turn, TypeSafe's **Jev**
judges what the turn needs; deterministic code combines that judgment with
the conversation's routing history and prompt-cache state to pick a concrete
`(model, effort)`; LiteLLM serves the turn with it.

Three constraints from the brief shaped everything:

1. **Whole-conversation awareness** — the decision considers the transcript,
   tool definitions and in-flight tool calls, not just the newest message.
2. **Routing memory** — a per-conversation ledger of prior decisions feeds
   the next one.
3. **Cache-aware economics** — staying on the incumbent model is worth real
   money when its prefix is cached; switching is priced against that.

### Decisions

| Question | Decision |
|---|---|
| Host | LiteLLM proxy; the router is a `CustomLogger` plugin on the pre-call hook. Ingress: `/v1/chat/completions`, `/v1/responses`, Anthropic `/v1/messages` |
| Scope rule | **LiteLLM owns every transaction** — formats, translation, streaming, retries, fallbacks, cache directives, pricing, token counting, cost logging. The router only asks Jev, remembers decisions, and picks `(model, effort)` |
| Judge | `typesafe-sdk` against `jev-latest`; six questions in one call; tests use a recorded judge |
| Objective | Quality first, cost second; both in dollars per turn |
| Unit of choice | `(model, effort)`, because a same-model effort change is usually the cheapest downgrade |
| Conversation identity | Client session header when present; otherwise response-fingerprint chaining, then a content hash |
| Privacy | Only a minimized state goes to TypeSafe; no built-in redaction (it damages what Jev judges); `state.filter` hook for teams with a policy |
| Providers | Anthropic and OpenAI (GPT-5.6 luna / terra / sol). Shipped default is OpenAI-only; Claude tiers are kept as comments in the deploy configs |
| Deployment | `docker compose`: LiteLLM `v1.101.0` + Redis |

## 2. Research that mattered

**Jev.** A "System One" model: it returns typed, calibrated answers to
questions about a `state`, in one parallel pass, without generating text.
`score` questions give a full probability distribution over ordered levels;
`noul` gives a probability; `choice` a label. State plus questions must fit
~32k tokens and accuracy drops with irrelevant content, so the router sends a
minimized state, never the raw transcript. No rationale is returned, so full
distributions are logged for audit. TypeSafe's guidance, followed here:
*"Code calculates. Jev judges."* Jev is not asked to do the cost arithmetic.

**Prompt caches are model-scoped everywhere.** Anthropic (explicit
`cache_control`, ~0.1× read, 1.25× write, 5-minute TTL refreshed on read),
OpenAI (automatic on ≥1024-token prefixes in 128-token steps, ~0.1× read) and
Gemini all key the cache on the model. A model switch re-processes the whole
history at full price and pays the write premium again. On Anthropic,
thinking blocks are also bound to the producing model, so a switch discards
the model's own reasoning: a quality cost separate from the cache cost.

**LiteLLM covers everything non-routing.** Verified in the pinned version:
the pre-call hook fires on all three ingress paths (`/v1/messages` was
reported broken in earlier versions); `data["model"]` and
`reasoning_effort` may be rewritten; `cache_control_injection_points` places
Anthropic breakpoints with no code; usage is normalized to
`prompt_tokens_details.cached_tokens` for both providers; `cost_per_token`,
`get_model_info` and `token_counter` mean the router carries **no price or
capability table**; `x-litellm-session-id` and harness metadata (Claude Code,
Codex, opencode) resolve to `data["litellm_session_id"]` before the hook.

**Prior art** (prismhq/jev-router, RouteLLM, the SLM front-door routing
paper) routes single prompts or the last few messages, and asks the judge to
pick a model directly. None handle multi-turn stickiness or cache state, which
is exactly the gap this design fills.

## 3. Architecture

```
client ──► LiteLLM proxy   {model: "jev-auto"} on chat completions / responses / messages
              │
              ▼
   JevRouterPlugin.async_pre_call_hook
              │  reads   messages, tools, litellm_session_id, metadata
              │  ledger  cache.get("jev_router:conv:<conversation_id>")
              │  asks    Jev once, six questions, minimized state     ← skipped on tool-result turns
              │  prices  litellm.cost_per_token per tier with predicted cached tokens
              │  writes  data["model"], effort, metadata["jev_router"]
              ▼
   LiteLLM: translation, cache breakpoints, streaming, retries, fallbacks, provider call
              │
              ▼
   JevRouterPlugin.async_log_success_event
              │  reads   usage (cached tokens), response_cost
              │  writes  ledger: incumbent (model, effort), observed cache, timestamp, history
```

The plugin changes three request fields on the way in and reads one usage
object on the way out. It never edits `messages` or `tools`.

### Components (`jev_router/core`)

**FastPath.** Tool-result turns go to the incumbent without calling Jev.
Tiers whose context window the prompt exceeds are ineligible. Jev
unreachable or over budget: keep the incumbent, else the default (fail open).

**StateBuilder.** Jev's state is a deterministic zoom by recency, budgeted at
8k tokens: the current message in full (head + tail if long), the last six
messages truncated, one-line stubs for older messages, the system prompt
head, tool names, signals (images, code blocks, pending tool calls) and the
router's own last five decisions as a semantic trail. No model-written
summary; every cap is in `router.yaml`.

**JevJudge.** One call, six questions:

| Key | Type | Used for |
|---|---|---|
| `required_tier` | score, 0 trivial · 1 routine · 2 demanding · 3 frontier | quality-risk term (full distribution) |
| `task_type` | choice over eight kinds | analytics |
| `continues_task`, `needs_history` | noul | switch-cost weight |
| `quality_complaint` | noul | forces a tier above the incumbent |
| `expected_output` | score, short / medium / long | output-token estimate |

**Ledger.** Per conversation, in LiteLLM's `DualCache`: incumbent model and
effort, last start time, observed cached tokens, stable-prefix estimate,
recent decisions. Cache prediction per candidate: same model and effort
within TTL → last observed; different model or past TTL → 0; same model,
different effort → the stable prefix on Anthropic, **0 on OpenAI** (see §5).
Corrected from real usage after every response.

**Scorer.** Pure code over the tier list:

```
cost         = litellm.cost_per_token(model, prompt − cached, expected_output, cache_read=cached, cache_creation=prompt − cached)
quality_risk = Σ_t P(required_tier = t) × tier_penalty[t − tier.level]      for t > tier.level
switch_cost  = 0 if incumbent else base + continuity × P(continues) × P(needs_history)
                                        + thinking_loss × [model changed] + cross_provider × [provider changed]
utility      = −λ_cost × cost − λ_quality × quality_risk − switch_cost
```

Overrides: `quality_complaint` above threshold raises the floor to the next
tier in the ladder; `required_tier` confidence below threshold keeps the
incumbent. All three terms are dollars for this turn.

**Identity.** `session_id:thread_hash` when the client sent a session id
(the thread hash keeps a harness's subagents apart); else the fingerprint of
the last assistant message (`tool_use` ids, or a text hash) looked up in the
ledger; else a hash of the first user message. Compaction by a harness
starts a fresh thread, which is the truthful state.

## 4. Delivery

| Phase | Plan | Outcome |
|---|---|---|
| 0 — spike | Pinned proxy, plugin rewrites model + effort on all three ingress paths, cache check per tier | Done. The pinned image tag did not exist (`v1.101.0`, not `main-v1.101.0`) and the image ships no pip; both fixed in the Dockerfile |
| 1 — core | `core/` modules, plugin with shadow mode, replay CLI, unit tests | Done as planned |
| 2 — evaluation | Multi-turn eval set, shadow mode, metrics, tuning | Done as `jev-router eval`: a labelled golden dataset, live and simulate modes, JSON + Markdown reports with threshold gating. See `evals/README.md` |
| 3 — optional | Postgres decision store, Gemini tier, upstream LiteLLM work | Not needed yet |

## 5. What the first evaluation changed

Twelve labelled 5–10 turn conversations, run live against GPT-5.6:

- **Switch costs were in the wrong units.** `continuity: 0.02` exceeded the
  entire cost of most turns, so the router never routed down (0 downward
  moves in 78 turns). Scaled to `0.003`; the tuned run had 5–8 downward
  moves and the same upward ones.
- **Complaint threshold 0.6 escalated on ordinary follow-ups** ("Fine. Now a
  simple one…", "Fix the diagram so…") and ratcheted conversations to the
  top tier. Raised to `0.85`; genuine complaints score ≥ 0.92.
- **Jev rarely emits level 0** (3 of 75 judgments, all "thanks"), so
  level-0-only tiers were never chosen. `luna-medium` was moved to level 1.
- **OpenAI's prompt cache is keyed on `reasoning_effort`.** A prefix written
  at `medium` misses at `high` and `xhigh`. The ledger now predicts a full
  miss for same-model effort switches on OpenAI; the simulated cache in the
  evaluator agrees with observed hit ratios within two points.
- The proxy's Redis **response** cache served identical repeated requests
  without reaching the provider, which hid cache hits from the phase-0
  check; the check now opts out per request.

Still open, deliberately: with `λ_quality = 3` and `tier_penalty[1] = 0.05`
the router over-provisions fresh turns (a 15 % chance of a one-level miss
is worth ~$0.02, more than most turns cost) and never picks the lower-effort
variants of a model. Lowering `λ_quality` is the lever; it trades quality
for cost and is a product decision.

## 6. Risks and how they are covered

| Risk | Cover |
|---|---|
| Jev API changes or access lapses | `Judge` protocol; recorded judge in tests; `system-one-adapter-python` as a fallback |
| Calibration is TypeSafe's claim | Confidence gating from day one; distributions logged; eval reports the Jev level distribution |
| Added latency | Fast path on tool loops; 1.5 s budget; fail open. Observed p50 ≈ 200 ms, p95 ≈ 430 ms |
| Cache prediction drifts | Corrected from every response; predicted-vs-observed in the decision log |
| LiteLLM version drift | Pinned; `tests/integration` boots the real proxy with mock deployments and is the regression gate |
| No Redis | In-memory `DualCache` per process; cold-start routing, never errors |
| Shared session id across harness threads | Ledger keyed by `session_id:thread_hash` |
| Turn gaps longer than the cache TTL | No cache term, fresh decision; Anthropic 1-hour TTL is a `config.yaml` change if a route needs it |

## 7. Sources

- TypeSafe: [blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [API reference](https://docs.typesafe.ai/api), [quick start](https://docs.typesafe.ai/introduction/quickstart), [jev-1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13), [Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python), [system-one-adapter-python](https://github.com/typesafe-ai/system-one-adapter-python), [skills](https://github.com/typesafe-ai/skills); coverage: [DataCamp](https://www.datacamp.com/blog/system-one-models-jev), [Latent Space](https://www.latent.space/p/ainews-jev-a-system-one-model-that), [DEV guide](https://dev.to/valyuai/how-to-use-jev-a-practical-guide-to-typesafes-system-one-model-g5e)
- Prior art: [prismhq/jev-router](https://github.com/prismhq/jev-router), [RouteLLM overview](https://neuraltrust.ai/blog/llm-model-routing), [routing survey arXiv 2603.04445](https://arxiv.org/pdf/2603.04445), [SLM front-door routing arXiv 2604.02367](https://arxiv.org/abs/2604.02367)
- LiteLLM: [call hooks](https://docs.litellm.ai/docs/proxy/call_hooks), [prompt caching](https://docs.litellm.ai/docs/completion/prompt_caching), [auto-inject cache checkpoints](https://docs.litellm.ai/docs/tutorials/prompt_caching), [Anthropic effort](https://docs.litellm.ai/docs/providers/anthropic_effort), [request headers](https://docs.litellm.ai/docs/proxy/request_headers); issues [#30469](https://github.com/BerriAI/litellm/issues/30469), [#27518](https://github.com/BerriAI/litellm/issues/27518) (hook on `/v1/messages`), [#11789](https://github.com/BerriAI/litellm/issues/11789), [#41067](https://github.com/BerriAI/litellm/issues/41067) (streaming cache accounting), [#25957](https://github.com/BerriAI/litellm/issues/25957) (effort gating)
- Caching: [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching), [OpenAI Prompt Caching 201](https://developers.openai.com/cookbook/examples/prompt_caching_201), [Gemini implicit caching](https://developers.googleblog.com/gemini-2-5-models-now-support-implicit-caching/); Anthropic caching, thinking-block and effort facts from the Claude API reference
