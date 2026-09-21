# jev-model-router

A LiteLLM proxy plugin that uses TypeSafe's Jev (a "System One" decision
model) to judge each conversation turn, then routes it to the cheapest
`(model, effort)` that clears the quality bar, accounting for routing history
and prompt-cache savings on the incumbent model. Applications call the proxy
with a router alias as the model (`jev-auto-gpt`, `jev-auto-claude`,
`jev-auto-claude-code`; one proxy serves them all) and get an ordinary
OpenAI- or Anthropic-format response back.

```
app ──▶ LiteLLM proxy ──▶ jev_router plugin ──▶ Jev: "what does this turn need?"
                              │                        │
                              ▼                        ▼
                    pick (model, effort) ◀── cost, quality risk, switch cost, cache state
                              │
                              ▼
                    OpenAI / Anthropic deployment
```

## Quickstart

You need Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker, a Jev key
(`TYPESAFE_API_KEY`), and a provider key for the router you'll call: the
quickstart uses `jev-auto-gpt` (`OPENAI_API_KEY`); `jev-auto-claude` needs
`ANTHROPIC_API_KEY`; `jev-auto-claude-code` uses your claude.ai login. One
proxy serves all three — see the table in [deploy/README.md](deploy/README.md).

```sh
make setup                            # .venv with the package and dev tools
make test                             # unit + integration tests; boots a mock proxy, no keys needed

cp deploy/.env.example deploy/.env    # fill in TYPESAFE_API_KEY and OPENAI_API_KEY
make up                               # LiteLLM proxy + Redis on http://localhost:4000
make preflight                        # every tier accepts its effort and gets a cache hit

curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-change-me" -H "Content-Type: application/json" \
  -H "x-litellm-session-id: demo" \
  -d '{"model":"jev-auto-gpt","messages":[{"role":"user","content":"Prove that there are infinitely many primes."}]}'
make logs                             # the routing decision for that request, as JSON
```

Send the same `x-litellm-session-id` on each turn of a conversation so the
router can track its incumbent model and cache state; without it, turns are
chained by the previous reply's fingerprint.

To use the router from Claude Code — on your claude.ai subscription with Claude
tiers, or on the GPT-5.6 ladder — see [deploy/README.md](deploy/README.md#claude-code);
it also covers shadow mode and replaying saved requests.

## Tuning the router

Each file in [`deploy/routers/`](deploy/routers/) is one router: its alias, tier
ladder and levels, the cost/quality trade-off, switch penalties, complaint
handling, and what Jev is shown. [`gpt.yaml`](deploy/routers/gpt.yaml) is fully
annotated; `claude.yaml` and `claude-code.yaml` note only what differs.
Deployments the tiers point at are shared in [`deploy/config.yaml`](deploy/config.yaml).
After changing either:

```sh
make eval                                   # real Jev, no model calls, ~15 s: did the routing change the way you meant?
make eval ROUTER=deploy/routers/claude.yaml # any router file
make eval-live                              # against the running proxy with real providers: cache hits and cost
```

Both write a JSON report and a Markdown summary to `evals/reports/` and exit
non-zero when the dataset's thresholds fail. See [evals/README.md](evals/README.md)
for the dataset format and how to read a report.

## Layout

```
jev_router/core/        host-agnostic routing logic
  config.py               router file schema; load_configs() reads a file or a directory of them
  request.py              Turn extracted from the request LiteLLM hands the hook
  identity.py             conversation id: session header, response chaining, content hash
  state.py                the minimized state Jev sees
  judge.py                the six Jev questions; JevJudge and RecordedJudge
  ledger.py               per-conversation incumbent, cache state, routing history
  scorer.py               utility = -cost - quality risk - switch cost
  fastpath.py             rules that skip Jev (tool-result turns)
  router.py               decide() / observe()
jev_router/litellm_plugin.py   CustomLogger wiring for the proxy
jev_router/cli.py              `jev-router replay | state | eval`
jev_router/eval/               golden-dataset evaluation: runner (live / simulate), scoring, reports
deploy/                        docker-compose, LiteLLM config, routers/*.yaml (one router each)
evals/                         datasets and reports for evaluating a router config
tests/                         unit tests; tests/integration boots a real proxy with mock models
docs/DESIGN.md                 how the router works, with diagrams
docs/IMPLEMENTATION_PLAN_v0.md design record: rationale, decisions, and what the first eval changed
```

## Development

```sh
make test      # pytest; `pytest -m "not integration"` for unit tests only
make lint      # ruff + pyright (what CI runs)
make fmt       # apply ruff fixes and formatting
```

Two kinds of testing live side by side and should stay separate: `tests/`
answers "did this change break the router?" with no network and no keys;
`jev-router eval` answers "how well does this configuration route?" against a
labelled dataset.

## License

MIT — see [LICENSE](LICENSE).
