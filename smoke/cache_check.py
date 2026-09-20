"""Phase 0 spike against a real proxy: two identical requests per tier must show cached tokens on the second."""

import os
import sys

import httpx
import yaml

BASE = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
KEY = os.environ.get("LITELLM_MASTER_KEY", "sk-change-me")
SYSTEM = "You are a meticulous assistant. " + ("Follow the house style guide exactly. " * 150)


def usage(r: dict) -> tuple[int, int, int]:
    u = r.get("usage") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or u.get("cache_read_input_tokens") or 0
    return u.get("prompt_tokens") or u.get("input_tokens") or 0, cached, u.get("cache_creation_input_tokens") or 0


def main(router_yaml: str = "deploy/router.yaml") -> None:
    with open(router_yaml) as f:
        cfg = yaml.safe_load(f)
    client = httpx.Client(base_url=BASE, headers={"Authorization": f"Bearer {KEY}"}, timeout=120)
    ok = True
    for tier in cfg["tiers"]:
        body = {
            "model": tier["model"],
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Reply with one word: ready."},
            ],
            "max_tokens": 20,
            "cache": {"no-cache": True},  # bypass LiteLLM's response cache so the 2nd call reaches the provider
        }
        if tier.get("effort"):
            body["reasoning_effort"] = tier["effort"]
        results = []
        for _ in range(2):
            r = client.post("/v1/chat/completions", json=body)
            if r.status_code != 200:
                print(f"{tier['name']:14} HTTP {r.status_code}: {r.text[:160]}")
                ok = False
                break
            results.append(usage(r.json()))
        if len(results) == 2:
            (p1, c1, w1), (p2, c2, w2) = results
            status = "ok" if c2 > 0 else "NO CACHE HIT"
            ok &= c2 > 0
            print(f"{tier['name']:14} prompt={p1:5} first: cached={c1:5} written={w1:5}", end=" | ")
            print(f"second: cached={c2:5} -> {status}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(*sys.argv[1:])
