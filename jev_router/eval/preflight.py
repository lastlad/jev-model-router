"""Phase-0 check of a router config against a live proxy.

Every tier must accept its effort value and show a prompt-cache hit on a repeated request; a
rejected effort or a missing hit means the tier cannot be routed to as configured.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from ..core.config import load_config

SYSTEM = "You are a careful assistant. " + ("Answer precisely and concisely. " * 120)


@dataclass
class TierCheck:
    tier: str
    model: str
    effort: str | None
    ok: bool
    detail: str


def _usage(body: dict) -> tuple[int, int]:
    u = body.get("usage") or {}
    return u.get("prompt_tokens", 0), (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0


def preflight(router_config: str, base_url: str, api_key: str) -> list[TierCheck]:
    cfg = load_config(router_config)
    client = httpx.Client(base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=120)
    checks: list[TierCheck] = []
    for tier in cfg.tiers:
        body: dict = {
            "model": tier.model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Reply with one word: ready."},
            ],
            "max_tokens": 20,
            "cache": {"no-cache": True},  # bypass the proxy's response cache so the 2nd call reaches the provider
        }
        if tier.effort:
            body["reasoning_effort"] = tier.effort
        results = []
        error = None
        for _ in range(2):
            r = client.post("/v1/chat/completions", json=body)
            if r.status_code != 200:
                error = f"HTTP {r.status_code}: {r.text[:160]}"
                break
            results.append(_usage(r.json()))
        if error:
            checks.append(TierCheck(tier.name, tier.model, tier.effort, False, error))
            continue
        (p1, c1), (p2, c2) = results
        ok = c2 > 0
        checks.append(
            TierCheck(
                tier.name,
                tier.model,
                tier.effort,
                ok,
                f"prompt={p1} first cached={c1} second cached={c2}" + ("" if ok else " -> NO CACHE HIT"),
            )
        )
    return checks
