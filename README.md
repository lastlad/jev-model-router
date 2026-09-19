# jev-model-router

A LiteLLM proxy plugin that uses TypeSafe's Jev (a "System One" decision
model) to judge each conversation turn, then routes it to the cheapest
`(model, effort)` on Anthropic or OpenAI that clears the quality bar,
accounting for routing history and prompt-cache savings on the incumbent
model. Applications call the proxy with `model: "jev-auto"`.

Design and research: [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).
Running it: [deploy/README.md](deploy/README.md).

## Layout

```
jev_router/core/      host-agnostic routing logic
  config.py           router.yaml schema
  request.py          Turn extracted from the request LiteLLM hands the hook
  identity.py         conversation id: session header, response chaining, content hash
  state.py            the minimized state Jev sees
  judge.py            the six Jev questions; JevJudge and RecordedJudge
  ledger.py           per-conversation incumbent, cache state, routing history
  scorer.py           utility = -cost - quality risk - switch cost
  fastpath.py         rules that skip Jev (tool-result turns)
  router.py           decide() / observe()
jev_router/litellm_plugin.py   CustomLogger wiring for the proxy
jev_router/cli.py              replay and state inspection without a proxy
deploy/                        docker-compose, LiteLLM config, router.yaml
tests/
```

## Development

```sh
uv venv && uv pip install -e ".[dev]"
pytest
ruff check jev_router tests && ruff format --check jev_router tests
pyright jev_router
```
