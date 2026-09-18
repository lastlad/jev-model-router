# jev-model-router

A model router that uses TypeSafe's Jev (a "System One" decision model) to
judge each conversation turn, then routes it to the cheapest `(model, effort)`
that clears the quality bar, accounting for routing history and prompt-cache
savings on the incumbent model.

Status: research and planning. See
[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).
