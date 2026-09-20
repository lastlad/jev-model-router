# Running the router

```sh
cp deploy/.env.example deploy/.env   # fill in the keys
cd deploy && docker compose up --build
```

The proxy listens on port 4000. Call it with `model: "jev-auto"` on
`/v1/chat/completions`, `/v1/responses`, or Anthropic-format `/v1/messages`.

## Claude Code

Claude Code talks to the proxy in Anthropic Messages format on `/v1/messages`; the
router sees it as call type `anthropic_messages`. Two ways to run it:

**On your claude.ai subscription, Claude tiers** (`router.claude-code.yaml`):

```sh
ROUTER_CONFIG=router.claude-code.yaml make up     # or: make up-claude-code
source deploy/claude-code.env
claude
```

No `ANTHROPIC_AUTH_TOKEN`/`ANTHROPIC_API_KEY` is set, so Claude Code keeps your claude.ai
login as its credential. The proxy authenticates you through the `x-litellm-api-key` custom
header and forwards the login's OAuth bearer and `anthropic-beta` header to Anthropic for the
`haiku`/`sonnet`/`opus` deployments (`model_group_settings.forward_client_headers_to_llm_api`
in `config.yaml`). Usage counts against the subscription; the router's cost term is LiteLLM's
list price, which stands in for how much of the plan limit a turn burns. This is the setup
[Anthropic documents for gateways](https://code.claude.com/docs/en/llm-gateway#subscriptions-and-gateways)
and [LiteLLM documents for Claude Code](https://docs.litellm.ai/docs/tutorials/claude_code_max_subscription).

**On the GPT-5.6 ladder, billed per token** (`router.yaml`, verified end to end):

```sh
make up && source deploy/claude-code-openai.env && claude
```

Anthropic does not support routing Claude Code to non-Claude models, so agentic behaviour is
rougher there; it is the way to test the plumbing without Anthropic access.

In both modes:

- Claude Code sends `x-claude-code-session-id`, which LiteLLM resolves to the session id, so
  conversations are tracked without extra headers; subagents get their own thread by the
  first-message hash.
- Claude Code inserts per-turn `role: system` entries into `messages` (environment block,
  token counter); the router ignores them for identity, stable prefix, and the tool-result
  fast path.
- Claude Code warns that `jev-auto` "isn't described by this version's model catalog" and
  assumes a 200k context window. Harmless; map it with `modelOverrides` in Claude Code
  settings if you want the picker to show the model behind it.
- Watch decisions with `make logs`. Tool-result turns should show `reason=tool_result` in a
  few milliseconds with a near-total cache hit on the next observation.
- `jev-router eval preflight` cannot check the Claude tiers in subscription mode: it has no
  OAuth token to forward. The first real Claude Code turn is the check.

## Shadow mode

Set `shadow: true` in `router.yaml`. Decisions are computed and logged under
`metadata.jev_router` but the request goes to the `jev-auto` deployment.

## Check the tiers and evaluate routing

```sh
jev-router eval preflight                                   # every tier accepts its effort and caches
jev-router eval run evals/datasets/routing-golden.yaml      # see evals/README.md
```

Routing decisions are appended to `deploy/logs/decisions.jsonl` (`JEV_ROUTER_LOG_FILE` in
docker-compose.yml); the live evaluation reads them from there.

## Replay a saved request without the proxy

```sh
jev-router state request.json                       # what Jev would see
jev-router replay request.json                      # calls Jev, prints the decision table
jev-router replay request.json --judgment j.json    # recorded judgment, no network
```
