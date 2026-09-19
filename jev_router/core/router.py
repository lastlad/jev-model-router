import asyncio
import logging
import time
from typing import Any

from . import identity
from .config import RouterConfig
from .fastpath import fast_route
from .judge import Judge
from .ledger import HistoryItem, Ledger, LedgerEntry
from .request import Turn, turn_from_data
from .scorer import Decision, choose
from .state import build_state

log = logging.getLogger("jev_router")


class Router:
    def __init__(self, cfg: RouterConfig, judge: Judge, ledger: Ledger) -> None:
        self.cfg = cfg
        self.judge = judge
        self.ledger = ledger

    def _token_model(self) -> str:
        t = self.cfg.tier(self.cfg.default)
        return t.model_id or t.model

    async def decide(
        self, data: dict[str, Any], call_type: str = "completion", now: float | None = None
    ) -> tuple[Decision, Turn]:
        now = now or time.time()
        turn = turn_from_data(data, call_type, self._token_model())
        conv_id = await identity.resolve_conversation(data, turn, self.ledger)
        entry = await self.ledger.get(conv_id)

        decision = fast_route(turn, entry)
        if decision is None:
            decision = await self._judge_and_choose(turn, entry, now)
        decision.conversation_id = conv_id
        return decision, turn

    async def _judge_and_choose(self, turn: Turn, entry: LedgerEntry | None, now: float) -> Decision:
        state = build_state(turn, entry.history if entry else [], self.cfg.state)
        try:
            judgment = await asyncio.wait_for(self.judge.judge(state), self.cfg.jev.timeout_ms / 1000)
        except Exception as e:  # fail open: never block a request on Jev
            log.warning("jev unavailable (%s); falling back", type(e).__name__)
            return Decision(entry.tier if entry else self.cfg.default, "jev_unavailable")
        return choose(turn, judgment, entry, self.cfg, self.ledger, now)

    async def observe(
        self,
        decision: Decision,
        turn: Turn,
        cached_tokens: int,
        response_fingerprint: str | None,
        start: float,
    ) -> None:
        tier = self.cfg.tier(decision.tier)
        prev = await self.ledger.get(decision.conversation_id)
        j = decision.judgment
        item = HistoryItem(
            tier=tier.name,
            task_type=j.task_type if j else "-",
            required_tier=round(j.expected_tier, 2) if j else -1,
            confidence=round(j.required_confidence, 2) if j else -1,
            cached_tokens=cached_tokens,
            gap_s=round(start - prev.last_start, 1) if prev else 0.0,
        )
        entry = LedgerEntry(
            tier=tier.name,
            model=tier.model,
            effort=tier.effort,
            provider=tier.provider,
            last_start=start,
            cached_tokens=cached_tokens,
            stable_prefix_tokens=turn.stable_prefix_tokens,
            history=(prev.history if prev else []) + [item],
        )
        entry.history = entry.history[-self.cfg.state.routing_history_entries * 2 :]
        await self.ledger.put(decision.conversation_id, entry)
        if response_fingerprint:
            await self.ledger.link_response(response_fingerprint, decision.conversation_id)

    async def forget_cache(self, conversation_id: str) -> None:
        entry = await self.ledger.get(conversation_id)
        if entry:
            entry.cached_tokens = 0
            await self.ledger.put(conversation_id, entry)
