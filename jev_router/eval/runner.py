"""Play a golden dataset through the router and collect one record per turn.

Two backends:

* ``LiveBackend`` sends each turn to a running LiteLLM proxy (``deploy/``) as ``jev-auto`` and
  correlates it with the plugin's ``decision``/``observed`` events in ``JEV_ROUTER_LOG_FILE``.
  Real providers, real replies, real cache accounting, real cost.
* ``SimulateBackend`` drives the router core in-process with the real Jev judge, a simulated
  provider prompt cache, and no model calls. Replies are the turn's pinned ``assistant:`` text or a
  placeholder, so record a live run first (``--record``) when Jev's view of the history matters.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

from ..core.config import RouterConfig, load_config
from ..core.judge import JevJudge, Judge
from ..core.ledger import EFFORT_KEYED_CACHE_PROVIDERS, Ledger, MemoryKV
from ..core.router import Router
from ..core.scorer import cost_of
from ..litellm_plugin import resolve_tiers, usage_of
from .dataset import Conversation, Dataset, Turn

CACHE_MIN_TOKENS = 1024  # OpenAI and Anthropic both cache prefixes of at least this many tokens
OPENAI_CACHE_STEP = 128


@dataclass
class TurnRecord:
    idx: int
    kind: str
    preview: str
    expected_level: tuple[int, int] | None = None
    expected_fastpath: bool | None = None
    expected_complaint: bool | None = None
    tier: str | None = None
    level: int | None = None
    reason: str | None = None
    jev_ms: int | None = None
    jev_level: int | None = None
    jev_confidence: float | None = None
    judgment: dict[str, Any] | None = None
    model: str | None = None
    prompt_tokens: int = 0
    cached_tokens: int = 0
    predicted_cached: int | None = None
    completion_tokens: int = 0
    cost: float | None = None
    latency_ms: int = 0
    assistant_text: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["expected_level"] = list(self.expected_level) if self.expected_level else None
        return d


@dataclass
class ConversationRun:
    name: str
    expect: str
    session: str
    turns: list[TurnRecord] = field(default_factory=list)


@dataclass
class RunResult:
    dataset: str
    mode: str
    run_id: str
    started_at: float
    duration_s: float
    router_config: dict[str, Any]
    router_config_path: str
    target: str
    tier_levels: dict[str, int]
    conversations: list[ConversationRun] = field(default_factory=list)
    cache_source: str = "observed"


def _apply_decision(rec: TurnRecord, decision: dict[str, Any], tier_levels: dict[str, int]) -> None:
    rec.tier = decision.get("tier")
    rec.reason = decision.get("reason")
    rec.level = tier_levels.get(rec.tier or "")
    rec.judgment = decision.get("judgment")
    rt = (rec.judgment or {}).get("required_tier") or {}
    if rt:
        top = max(rt, key=lambda k: rt[k])
        rec.jev_level = int(top)
        rec.jev_confidence = round(float(rt[top]), 3)
    for s in decision.get("scores") or []:
        if s.get("tier") == rec.tier:
            rec.predicted_cached = s.get("predicted_cached")


def scripted_tool_messages(turn: Turn, call_id: str) -> list[dict[str, Any]]:
    assert turn.tool is not None
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": turn.tool.name, "arguments": json.dumps(turn.tool.args)},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": turn.tool.result},
    ]


# --------------------------------------------------------------------------------------------
# Live backend
# --------------------------------------------------------------------------------------------


class LiveBackend:
    def __init__(self, base_url: str, api_key: str, decisions: Path, max_tokens: int = 1200) -> None:
        self.client = httpx.AsyncClient(base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=300)
        self.decisions = decisions
        self.max_tokens = max_tokens
        self.offset = self._size()
        self.target = base_url
        self.cache_source = "observed"

    def _size(self) -> int:
        return self.decisions.stat().st_size if self.decisions.exists() else 0

    def _events_since(self, offset: int) -> tuple[list[dict[str, Any]], int]:
        if not self.decisions.exists():
            return [], offset
        with open(self.decisions) as f:
            f.seek(offset)
            chunk = f.read()
            end = f.tell()
        events = []
        for line in chunk.splitlines():
            with contextlib.suppress(json.JSONDecodeError):
                events.append(json.loads(line))
        return events, end

    async def _wait_events(self, session: str, timeout_s: float = 5.0) -> tuple[dict | None, dict | None]:
        deadline = time.time() + timeout_s
        decision = observed = None
        while time.time() < deadline:
            events, _ = self._events_since(self.offset)
            for ev in events:
                cid = (ev.get("decision") or {}).get("conversation_id") or ev.get("conversation_id") or ""
                if not cid.startswith(session + ":"):
                    continue
                if ev.get("event") == "decision":
                    decision = ev
                elif ev.get("event") == "observed":
                    observed = ev
            if decision and observed:
                break
            await asyncio.sleep(0.2)
        _, self.offset = self._events_since(self.offset)
        return decision, observed

    async def run_turn(
        self,
        session: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        rec: TurnRecord,
        tier_levels: dict[str, int],
    ) -> None:
        body: dict[str, Any] = {"model": "jev-auto", "messages": messages, "max_tokens": self.max_tokens}
        if tools:
            body["tools"] = tools
        t0 = time.time()
        try:
            r = await self.client.post("/v1/chat/completions", json=body, headers={"x-litellm-session-id": session})
            rec.latency_ms = int((time.time() - t0) * 1000)
            if r.status_code != 200:
                rec.error = f"HTTP {r.status_code}: {r.text[:200]}"
            else:
                j = r.json()
                rec.model = j.get("model")
                rec.prompt_tokens, rec.cached_tokens = usage_of(j)
                rec.completion_tokens = (j.get("usage") or {}).get("completion_tokens", 0)
                msg = (j.get("choices") or [{}])[0].get("message") or {}
                rec.assistant_text = msg.get("content") or "(no text output)"
        except httpx.HTTPError as e:
            rec.latency_ms = int((time.time() - t0) * 1000)
            rec.error = f"{type(e).__name__}: {e}"
        decision, observed = await self._wait_events(session)
        if decision:
            _apply_decision(rec, decision["decision"], tier_levels)
            rec.jev_ms = decision.get("jev_ms")
        if observed:
            rec.cost = observed.get("cost")

    async def close(self) -> None:
        await self.client.aclose()


# --------------------------------------------------------------------------------------------
# Simulate backend
# --------------------------------------------------------------------------------------------


class SimulatedCache:
    """Provider prompt cache: per session and model (and effort, where the provider keys on it)."""

    def __init__(self, ttl_s: int) -> None:
        self.ttl = ttl_s
        self.entries: dict[tuple[str, str, str | None], tuple[int, float]] = {}

    def key(self, session: str, model: str, effort: str | None, provider: str | None) -> tuple[str, str, str | None]:
        return session, model, effort if provider in EFFORT_KEYED_CACHE_PROVIDERS else None

    def hit(self, session: str, model: str, effort: str | None, provider: str | None, prompt: int, now: float) -> int:
        k = self.key(session, model, effort, provider)
        prev = self.entries.get(k)
        self.entries[k] = (prompt, now)
        if prev is None or now - prev[1] > self.ttl or prompt < CACHE_MIN_TOKENS:
            return 0
        cached = min(prev[0], prompt)
        if provider == "openai":
            cached -= cached % OPENAI_CACHE_STEP
        return cached


class SimulateBackend:
    def __init__(self, cfg: RouterConfig, judge: Judge, config_path: str) -> None:
        self.cfg = cfg
        self.router = Router(cfg, judge, Ledger(MemoryKV(), cfg.cache_ttl_seconds))
        self.cache = SimulatedCache(cfg.cache_ttl_seconds)
        self.target = f"in-process ({config_path})"
        self.cache_source = "simulated"

    async def run_turn(
        self,
        session: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        rec: TurnRecord,
        tier_levels: dict[str, int],
    ) -> None:
        data: dict[str, Any] = {"model": self.cfg.alias, "messages": messages, "litellm_session_id": session}
        if tools:
            data["tools"] = tools
        start = time.time()
        try:
            decision, turn = await self.router.decide(data, "acompletion", start)
        except Exception as e:
            rec.error = f"{type(e).__name__}: {e}"
            rec.latency_ms = int((time.time() - start) * 1000)
            return
        rec.jev_ms = int((time.time() - start) * 1000)
        _apply_decision(rec, decision.to_dict(), tier_levels)
        tier = self.cfg.tier(decision.tier)
        rec.model = tier.model
        rec.prompt_tokens = turn.prompt_tokens
        rec.cached_tokens = self.cache.hit(session, tier.model, tier.effort, tier.provider, turn.prompt_tokens, start)
        j = decision.judgment
        expected = self.cfg.expected_output_tokens[j.expected_output if j else 1]
        rec.completion_tokens = int(expected * self.cfg.effort_output_multiplier.get(tier.effort or "", 1.0))
        try:
            rec.cost = cost_of(tier, rec.prompt_tokens, rec.cached_tokens, rec.completion_tokens)
        except Exception:
            rec.cost = None
        rec.assistant_text = f"(simulated {tier.name} reply)"
        await self.router.observe(decision, turn, rec.cached_tokens, None, start)
        rec.latency_ms = int((time.time() - start) * 1000)

    async def close(self) -> None:
        return None


def make_simulate_backend(
    router_config: str, litellm_config: str | None, judge: Judge | None = None
) -> SimulateBackend:
    cfg = load_config(router_config)
    model_list: list[dict[str, Any]] = []
    if litellm_config and Path(litellm_config).exists():
        with open(litellm_config) as f:
            model_list = (yaml.safe_load(f) or {}).get("model_list", [])
    resolve_tiers(cfg, model_list)
    return SimulateBackend(cfg, judge or JevJudge(cfg.jev.model, cfg.jev.timeout_ms / 1000), router_config)


# --------------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------------


OnTurn = Callable[[ConversationRun, TurnRecord], None]


async def run_dataset(
    ds: Dataset,
    backend: LiveBackend | SimulateBackend,
    router_config_path: str,
    mode: str,
    on_turn: OnTurn | None = None,
    on_conversation: Callable[[ConversationRun], None] | None = None,
) -> RunResult:
    cfg = load_config(router_config_path)
    with open(router_config_path) as f:
        cfg_raw = yaml.safe_load(f)
    tier_levels = {t.name: t.level for t in cfg.tiers}
    run_id = time.strftime("%Y%m%d-%H%M%S")
    started = time.time()
    result = RunResult(
        dataset=ds.name,
        mode=mode,
        run_id=run_id,
        started_at=started,
        duration_s=0.0,
        router_config=cfg_raw,
        router_config_path=router_config_path,
        target=backend.target,
        tier_levels=tier_levels,
        cache_source=backend.cache_source,
    )
    for conv in ds.conversations:
        run = ConversationRun(conv.name, conv.expect, f"eval-{run_id}-{conv.name}")
        if on_conversation:
            on_conversation(run)
        await _run_conversation(conv, run, backend, tier_levels, on_turn)
        result.conversations.append(run)
    result.duration_s = round(time.time() - started, 1)
    return result


async def _run_conversation(
    conv: Conversation,
    run: ConversationRun,
    backend: LiveBackend | SimulateBackend,
    tier_levels: dict[str, int],
    on_turn: OnTurn | None,
) -> None:
    messages: list[dict[str, Any]] = [{"role": "system", "content": conv.system}]
    for i, turn in enumerate(conv.turns, 1):
        rec = TurnRecord(
            idx=i,
            kind=turn.kind,
            preview=turn.preview(),
            expected_level=turn.expected_range,
            expected_fastpath=turn.expected_fastpath,
            expected_complaint=turn.expected_complaint,
        )
        if turn.tool:
            messages.extend(scripted_tool_messages(turn, f"call_{run.session[-8:]}_{i}"))
        else:
            messages.append({"role": "user", "content": turn.user})
        await backend.run_turn(run.session, messages, conv.tools, rec, tier_levels)
        reply = turn.assistant or rec.assistant_text or "(request failed)"
        messages.append({"role": "assistant", "content": reply})
        run.turns.append(rec)
        if on_turn:
            on_turn(run, rec)
