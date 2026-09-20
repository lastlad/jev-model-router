# Evaluating the router

Two things get tested in this repo, on purpose separately:

| | Regression (`pytest`) | Evaluation (`jev-router eval`) |
|---|---|---|
| Question | Did a code change break the router? | How well does *this router file* route? |
| Needs | nothing (fake Jev, mock models) | a Typesafe key; provider keys for live mode |
| Lives in | `tests/` (unit) + `tests/integration/` (real LiteLLM proxy, mock deployments) | `jev_router/eval/`, datasets in `evals/datasets/` |
| Output | pass/fail | JSON report + Markdown summary, exit code from thresholds |

```sh
pytest                        # unit + integration (~5 s; the proxy boots once per session)
pytest -m "not integration"   # unit only
```

## Running an evaluation

```sh
# Simulate (default): no proxy, no model calls. Real Jev judgments; provider cache and cost are simulated.
TYPESAFE_API_KEY=... jev-router eval run evals/datasets/routing-golden.yaml

# Live: real proxy in deploy/, real providers, real prompt caching and cost (~$0.60 for the golden set).
make up
LITELLM_MASTER_KEY=... jev-router eval run evals/datasets/routing-golden.yaml --mode live

# Options
  --config deploy/routers/gpt.yaml  router file under evaluation (its alias is the model in live mode)
  --only support --only sql          substring filter on conversation names
  --record evals/datasets/x.yaml     live only: write a copy of the dataset with replies pinned
  --out-dir evals/reports            where <dataset>-<mode>-<run_id>.{json,md} land
```

`eval preflight` checks, against a live proxy, that every tier in a router file accepts its effort
value and gets a prompt-cache hit on a repeated request. Run it after changing tiers or LiteLLM.

### Live vs simulate

Simulate drives the router core in-process with the real Jev judge and a simulated provider cache
(per session and model; per effort too on OpenAI, whose cache is keyed on `reasoning_effort`).
It is free and takes seconds, and it reproduces the routing decision exactly, but:

- assistant replies are placeholders unless the turn pins `assistant:`; Jev sees those in the
  history, so record a live run first (`--record`) when reply content matters;
- cache hit ratios and cost are the router's own model of the provider, not measurements.

Use live runs to validate the cache model and cost; use simulate to iterate on a router file.

## Datasets

A dataset is YAML: reusable `contexts`, then `conversations` of 5–10 `turns`. A turn is a `user`
message or a scripted `tool` result. Labels are optional per turn:

```yaml
- user: Prove that the sum of 1/p over primes diverges.
  expected_level: [2, 3]        # int, or an inclusive [lo, hi] range on the 0–3 Jev scale
- tool: { name: run_tests, args: { path: tests/ }, result: "1 failed" }
  expected_fastpath: true       # must route to the incumbent without calling Jev
- user: That is wrong. Think harder.
  expected_complaint: true      # must escalate with reason=quality_complaint
  assistant: (pinned reply)     # optional; used instead of the live reply
```

`thresholds` at the top level gate the exit code: `level_accuracy` and `fastpath_accuracy` are
minimums, `under_provision_rate`, `over_provision_rate`, `false_complaint_rate`, `cost_usd`,
`jev_p95_ms` and `errors` are maximums.

`routing-golden.yaml` has twelve conversations: steady trivial/routine/deep work, an agent tool
loop, genuine complaints, escalation, de-escalation, spiky difficulty and a long shared prefix.

## Reading the report

The Markdown summary has a scorecard (level accuracy, over/under-provisioning, level moves
up/down, switches, fast-path and complaint counts, cache hit ratio, cost, Jev latency, tier usage),
threshold results, one row per conversation (actual vs expected level trajectory), auto-generated
findings (under-provisioned turns, false complaints, tiers never selected, "never routed down"),
and every turn with Jev's judgment and the router's reason. The JSON has the same plus the full
judgment and score table per turn and a snapshot of the router config, so two reports can be
diffed to see what a config change did.
