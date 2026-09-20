from conftest import entry, judgment

from jev_router.core.config import Tier
from jev_router.core.request import Turn
from jev_router.core.scorer import choose


def turn(tokens=20000) -> Turn:
    return Turn(messages=[], prompt_tokens=tokens)


def test_cached_incumbent_wins_routine_follow_up(cfg, ledger):
    d = choose(
        turn(),
        judgment(level=1, continues_task=0.9, needs_history=0.8),
        entry(),
        cfg,
        ledger,
        now=1010.0,
    )
    assert d.tier == "sonnet-medium" and d.reason == "stay"


def test_cold_cache_lets_cheaper_tier_win(cfg, ledger):
    d = choose(
        turn(),
        judgment(level=0, continues_task=0.1, needs_history=0.1),
        entry(last_start=0.0),
        cfg,
        ledger,
        now=10000.0,
    )
    assert d.tier in ("haiku", "luna")


def test_quality_complaint_forces_upgrade(cfg, ledger):
    d = choose(turn(), judgment(level=1, quality_complaint=0.95), entry(), cfg, ledger, now=1010.0)
    assert d.reason == "quality_complaint" and cfg.rank(d.tier) > cfg.rank("sonnet-medium")


def test_frontier_task_upgrades_despite_cache(cfg, ledger):
    d = choose(turn(), judgment(level=3), entry(), cfg, ledger, now=1010.0)
    assert d.tier == "opus-medium"


def test_effort_only_downgrade_beats_model_switch_for_adjacent_tiers(cfg, ledger):
    d = choose(
        turn(),
        judgment(level=0, continues_task=0.5, needs_history=0.5),
        entry(),
        cfg,
        ledger,
        now=1010.0,
    )
    scored = {s.tier: s for s in d.scores}
    assert scored["sonnet-low"].predicted_cached == 3000 and scored["haiku"].predicted_cached == 0
    assert scored["sonnet-low"].switch_cost < scored["haiku"].switch_cost < scored["luna"].switch_cost


def test_low_confidence_keeps_incumbent(cfg, ledger):
    d = choose(turn(), judgment(level=3, conf=0.2), entry(), cfg, ledger, now=1010.0)
    assert d.tier == "sonnet-medium" and d.reason == "low_confidence"


def test_context_overflow_excludes_small_context_tier(cfg, ledger):
    d = choose(turn(tokens=300_000), judgment(level=0), None, cfg, ledger, now=0.0)
    assert "haiku" not in {s.tier for s in d.scores}


def test_complaint_at_top_tier_stays_at_top(cfg, ledger):
    top = cfg.tiers[-1].name
    e = entry(top)
    e.model, e.effort = cfg.tiers[-1].model, cfg.tiers[-1].effort
    d = choose(turn(), judgment(level=2, quality_complaint=0.95), e, cfg, ledger, now=1010.0)
    assert d.tier == top and d.reason == "quality_complaint"


def test_effort_switch_on_openai_predicts_cache_miss(cfg, ledger):
    e = entry("sol")
    e.model, e.effort, e.provider = "sol", "high", "openai"
    sol_medium = Tier(name="sol-medium", model="sol", effort="medium", level=2, provider="openai")
    sol_high = cfg.tier("sol")
    assert ledger.predict_cached(e, sol_high, now=1010.0) == 20000
    assert ledger.predict_cached(e, sol_medium, now=1010.0) == 0
