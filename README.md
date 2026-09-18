# jev-model-router

A LiteLLM proxy plugin that uses TypeSafe's Jev (a "System One" decision
model) to judge each conversation turn, then routes it to the cheapest
`(model, effort)` on Anthropic or OpenAI that clears the quality bar,
accounting for routing history and prompt-cache savings on the incumbent
model. Applications call the proxy's OpenAI-compatible endpoint with
`model: "jev-auto"`.

Status: research and planning. See
[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).
