# Running the router

```sh
cp deploy/.env.example deploy/.env   # fill in the keys
cd deploy && docker compose up --build
```

The proxy listens on port 4000. Call it with `model: "jev-auto"` on
`/v1/chat/completions`, `/v1/responses`, or Anthropic-format `/v1/messages`.

## Claude Code

```sh
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_AUTH_TOKEN=$LITELLM_MASTER_KEY
export ANTHROPIC_MODEL=jev-auto
export ANTHROPIC_DEFAULT_HAIKU_MODEL=haiku   # background calls skip routing
```

Claude Code's session id is read from its request metadata by LiteLLM, so
conversations are tracked without extra headers. Confirm the variable names
against the Claude Code version in use.

## Shadow mode

Set `shadow: true` in `router.yaml`. Decisions are computed and logged under
`metadata.jev_router` but the request goes to the `jev-auto` deployment.

## Replay a saved request without the proxy

```sh
jev-router state request.json                       # what Jev would see
jev-router replay request.json                      # calls Jev, prints the decision table
jev-router replay request.json --judgment j.json    # recorded judgment, no network
```
