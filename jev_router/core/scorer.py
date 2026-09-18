from dataclasses import dataclass, field
from typing import Any

import litellm

from .config import RouterConfig, Tier
from .judge import Judgment
from .ledger import Ledger, LedgerEntry
from .request import Turn


@dataclass
class Scored:
    tier: str
    cost: float
    quality_risk: float
    switch_cost: float
    utility: float
    predicted_cached: int


@dataclass
class Decision:
    tier: str
    reason: str
    conversation_id: str = ""
    judgment: Judgment | None = None
    scores: list[Scored] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "reason": self.reason,
            "conversation_id": self.conversation_id,
            "judgment": self.judgment.to_dict() if self.judgment else None,
            "scores": [vars(s) for s in self.scores],
        }


def eligible(tier: Tier, turn: Turn) -> bool:
    try:
        info = litellm.get_model_info(tier.model_id or tier.model)
    except Exception:
        return True
    max_in = info.get("max_input_tokens") or 0
    if max_in and turn.prompt_tokens > max_in:
        return False
    if turn.tools and info.get("supports_function_calling") is False:
        return False
    return not (turn.image_count and info.get("supports_vision") is False)


def cost_of(tier: Tier, prompt_tokens: int, cached: int, output_tokens: int) -> float:
    uncached = max(0, prompt_tokens - cached)
    inp, out = litellm.cost_per_token(
        model=tier.model_id or tier.model,
        prompt_tokens=prompt_tokens,
        completion_tokens=output_tokens,
        cache_read_input_tokens=cached,
        cache_creation_input_tokens=uncached,
    )
    return inp + out


def quality_risk(tier: Tier, judgment: Judgment, penalty: list[float]) -> float:
    return sum(
        p * penalty[min(t - tier.level, len(penalty) - 1)] for t, p in judgment.required_tier.items() if t > tier.level
    )


def switch_cost(tier: Tier, entry: LedgerEntry | None, judgment: Judgment, cfg: RouterConfig) -> float:
    if entry is None or tier.name == entry.tier:
        return 0.0
    sc = cfg.switch_cost
    cost = sc.base + sc.continuity * judgment.continues_task * judgment.needs_history
    if tier.model != entry.model:
        cost += sc.thinking_loss
    if tier.provider and entry.provider and tier.provider != entry.provider:
        cost += sc.cross_provider
    return cost


def choose(
    turn: Turn,
    judgment: Judgment,
    entry: LedgerEntry | None,
    cfg: RouterConfig,
    ledger: Ledger,
    now: float,
) -> Decision:
    incumbent = entry.tier if entry else None
    if judgment.required_confidence < cfg.jev.min_confidence:
        return Decision(incumbent or cfg.default, "low_confidence", judgment=judgment)

    complaint = bool(incumbent) and judgment.quality_complaint >= cfg.jev.complaint_threshold
    floor = min(cfg.rank(incumbent) + 1, len(cfg.tiers) - 1) if complaint and incumbent else 0
    output_tokens = cfg.expected_output_tokens[judgment.expected_output]
    scores: list[Scored] = []
    for tier in cfg.tiers:
        if cfg.rank(tier.name) < floor or not eligible(tier, turn):
            continue
        cached = ledger.predict_cached(entry, tier, now)
        out = int(output_tokens * cfg.effort_output_multiplier.get(tier.effort or "", 1.0))
        cost = cost_of(tier, turn.prompt_tokens, cached, out)
        risk = quality_risk(tier, judgment, cfg.tier_penalty)
        switch = switch_cost(tier, entry, judgment, cfg)
        utility = -cfg.objective.lambda_cost * cost - cfg.objective.lambda_quality * risk - switch
        scores.append(Scored(tier.name, cost, risk, switch, utility, cached))
    if not scores:
        return Decision(cfg.default, "no_eligible_tier", judgment=judgment)
    best = max(scores, key=lambda s: s.utility)
    reason = (
        "quality_complaint" if floor else ("fresh" if not incumbent else "stay" if best.tier == incumbent else "switch")
    )
    return Decision(best.tier, reason, judgment=judgment, scores=scores)
