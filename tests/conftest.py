import pytest

from jev_router.core.config import RouterConfig
from jev_router.core.judge import Judgment
from jev_router.core.ledger import Ledger, LedgerEntry, MemoryKV

TIERS = [
    {
        "name": "haiku",
        "model": "haiku",
        "effort": None,
        "level": 0,
        "model_id": "anthropic/claude-haiku-4-5",
        "provider": "anthropic",
    },
    {
        "name": "luna",
        "model": "luna",
        "effort": "low",
        "level": 0,
        "model_id": "openai/gpt-5.6-luna",
        "provider": "openai",
    },
    {
        "name": "sonnet-low",
        "model": "sonnet",
        "effort": "low",
        "level": 1,
        "model_id": "anthropic/claude-sonnet-5",
        "provider": "anthropic",
    },
    {
        "name": "sonnet-medium",
        "model": "sonnet",
        "effort": "medium",
        "level": 1,
        "model_id": "anthropic/claude-sonnet-5",
        "provider": "anthropic",
    },
    {
        "name": "sonnet-high",
        "model": "sonnet",
        "effort": "high",
        "level": 2,
        "model_id": "anthropic/claude-sonnet-5",
        "provider": "anthropic",
    },
    {
        "name": "sol",
        "model": "sol",
        "effort": "high",
        "level": 2,
        "model_id": "openai/gpt-5.6-sol",
        "provider": "openai",
    },
    {
        "name": "opus-medium",
        "model": "opus",
        "effort": "medium",
        "level": 3,
        "model_id": "anthropic/claude-opus-5",
        "provider": "anthropic",
    },
]


@pytest.fixture
def cfg() -> RouterConfig:
    return RouterConfig(tiers=TIERS, default="sonnet-medium")


@pytest.fixture
def ledger(cfg) -> Ledger:
    return Ledger(MemoryKV(), cfg.cache_ttl_seconds)


def entry(tier="sonnet-medium", cached=20000, stable=3000, last_start=1000.0) -> LedgerEntry:
    return LedgerEntry(
        tier=tier,
        model=tier.split("-")[0],
        effort=tier.split("-")[1] if "-" in tier else None,
        provider="anthropic",
        last_start=last_start,
        cached_tokens=cached,
        stable_prefix_tokens=stable,
    )


def judgment(level=1, conf=0.9, **kw) -> Judgment:
    return Judgment(required_tier={level: 1.0}, required_confidence=conf, **kw)


def chat(
    n_turns=1,
    current="ok, now add a unit test for it",
    system="You are a coding assistant.",
    tools=None,
) -> dict:
    msgs = [{"role": "system", "content": system}]
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"user message {i} " + "x" * 400})
        msgs.append({"role": "assistant", "content": f"assistant reply {i} " + "y" * 800})
    msgs.append({"role": "user", "content": current})
    data = {"model": "jev-auto", "messages": msgs}
    if tools:
        data["tools"] = tools
    return data
