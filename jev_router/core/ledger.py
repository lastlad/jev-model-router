from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from .config import Tier

# Providers whose prompt cache is keyed on reasoning effort as well as the prefix:
# a prefix written at one effort misses at another, so an effort-only switch on these
# providers is a full cache miss.
EFFORT_KEYED_CACHE_PROVIDERS = frozenset({"openai"})


class KV(Protocol):
    async def get(self, key: str) -> Any | None: ...
    async def set(self, key: str, value: Any, ttl: int) -> None: ...


class MemoryKV:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    async def get(self, key: str) -> Any | None:
        return self.data.get(key)

    async def set(self, key: str, value: Any, ttl: int) -> None:
        self.data[key] = value


@dataclass
class HistoryItem:
    tier: str
    task_type: str
    required_tier: float
    confidence: float
    cached_tokens: int
    gap_s: float


@dataclass
class LedgerEntry:
    tier: str
    model: str
    effort: str | None
    provider: str | None
    last_start: float
    cached_tokens: int = 0
    stable_prefix_tokens: int = 0
    history: list[HistoryItem] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LedgerEntry":
        d = dict(d)
        d["history"] = [HistoryItem(**h) for h in d.get("history", [])]
        return cls(**d)


class Ledger:
    def __init__(self, kv: KV, ttl_seconds: int, namespace: str = "") -> None:
        self.kv = kv
        self.ttl = ttl_seconds
        self.entry_ttl = ttl_seconds * 12  # keep routing history well past the cache TTL
        # Routers sharing one cache keep separate ledgers: a conversation's incumbent under one
        # ladder says nothing about another.
        self.prefix = f"jev_router:{namespace}:" if namespace else "jev_router:"

    async def get(self, conversation_id: str) -> LedgerEntry | None:
        raw = await self.kv.get(f"{self.prefix}conv:{conversation_id}")
        return LedgerEntry.from_dict(raw) if raw else None

    async def put(self, conversation_id: str, entry: LedgerEntry) -> None:
        await self.kv.set(f"{self.prefix}conv:{conversation_id}", asdict(entry), self.entry_ttl)

    async def link_response(self, fingerprint: str, conversation_id: str) -> None:
        await self.kv.set(f"{self.prefix}resp:{fingerprint}", conversation_id, self.entry_ttl)

    async def lookup_response(self, fingerprint: str) -> str | None:
        return await self.kv.get(f"{self.prefix}resp:{fingerprint}")

    def predict_cached(self, entry: LedgerEntry | None, tier: Tier, now: float) -> int:
        if entry is None or now - entry.last_start > self.ttl or tier.model != entry.model:
            return 0
        if tier.effort == entry.effort:
            return entry.cached_tokens
        if (tier.provider or entry.provider) in EFFORT_KEYED_CACHE_PROVIDERS:
            return 0
        return min(entry.cached_tokens, entry.stable_prefix_tokens)
