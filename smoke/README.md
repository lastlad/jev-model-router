# Smoke run without provider keys

Runs the real LiteLLM proxy with mocked deployments and a fake Jev endpoint,
then drives it through chat, follow-up, complaint, tool-result, streaming,
Anthropic Messages, Responses API, and header-less chaining.

```sh
python smoke/fake_jev.py &
TYPESAFE_API_KEY=fake TYPESAFE_BASE_URL=http://127.0.0.1:4010 \
JEV_ROUTER_CONFIG=deploy/router.yaml JEV_ROUTER_LOG_FILE=decisions.jsonl \
litellm --config smoke/config.yaml --port 4001 &
python smoke/drive.py
cat decisions.jsonl
```

Verifies plugin wiring only. Cache accounting, effort acceptance, and Jev's
real answers need the real deployment in `deploy/`.

## Phase 0 check against the real proxy

With `deploy/` running and keys set:

```sh
LITELLM_MASTER_KEY=... python smoke/cache_check.py
```

Sends two identical requests per tier with the tier's effort and prints the
cached-token count on the second. Every tier should show a cache hit and no
HTTP error; a rejected effort value or a missing hit is a phase 0 finding.
Set `JEV_ROUTER_LOG_LEVEL=DEBUG` on the proxy for verbose plugin logging.
