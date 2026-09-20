import textwrap

import pytest
from conftest import TIERS

from jev_router.core.config import RouterConfig
from jev_router.core.judge import Judgment, RecordedJudge
from jev_router.eval.dataset import Dataset, Thresholds, load_dataset
from jev_router.eval.report import build_report, check_thresholds, score_run, to_markdown
from jev_router.eval.runner import ConversationRun, RunResult, SimulateBackend, SimulatedCache, TurnRecord, run_dataset


@pytest.fixture
def dataset_file(tmp_path):
    p = tmp_path / "ds.yaml"
    p.write_text(
        textwrap.dedent(
            """
            name: mini
            thresholds: { level_accuracy: 0.5, fastpath_accuracy: 1.0 }
            contexts:
              code: You are a coding assistant.
            conversations:
              - name: fix
                system: { context: code }
                tools:
                  - { type: function, function: { name: run_tests, parameters: { type: object, properties: {} } } }
                turns:
                  - user: Refactor the parser.
                    expected_level: 1
                  - tool: { name: run_tests, args: {}, result: "1 failed" }
                    expected_fastpath: true
                  - user: That is wrong, think harder.
                    expected_level: [1, 3]
                    expected_complaint: true
                  - user: thanks
                    expected_level: 0
                    assistant: You're welcome.
            """
        )
    )
    return p


def test_dataset_loads_contexts_and_labels(dataset_file):
    ds = load_dataset(dataset_file)
    conv = ds.conversations[0]
    assert conv.system == "You are a coding assistant."
    assert [t.kind for t in conv.turns] == ["user", "tool", "user", "user"]
    assert conv.turns[0].expected_range == (1, 1) and conv.turns[2].expected_range == (1, 3)
    assert conv.turns[1].expected_fastpath and conv.turns[2].expected_complaint
    assert conv.turns[3].assistant == "You're welcome."
    assert ds.thresholds.level_accuracy == 0.5
    assert ds.select(["nope"]).conversations == [] and len(ds.select(["fi"]).conversations) == 1


def test_dataset_rejects_turn_without_user_or_tool():
    with pytest.raises(ValueError):
        Dataset.model_validate({"name": "x", "conversations": [{"name": "c", "system": "s", "turns": [{}]}]})


def test_dataset_rejects_unknown_context(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: x\nconversations:\n  - name: c\n    system: { context: missing }\n    turns: [{ user: hi }]\n")
    with pytest.raises(ValueError, match="unknown context"):
        load_dataset(p)


def _rec(idx, level, expected, reason="stay", fast=None, complaint=None, cached=0, prompt=100, jev=1, cost=0.01):
    return TurnRecord(
        idx=idx,
        kind="tool" if fast else "user",
        preview=f"t{idx}",
        expected_level=expected,
        expected_fastpath=fast,
        expected_complaint=complaint,
        tier=f"t{level}",
        level=level,
        reason=reason,
        jev_ms=None if reason == "tool_result" else 100,
        jev_level=None if reason == "tool_result" else jev,
        jev_confidence=0.9,
        prompt_tokens=prompt,
        cached_tokens=cached,
        cost=cost,
    )


def _result(*runs):
    return RunResult(
        dataset="d",
        mode="simulate",
        run_id="r",
        started_at=0.0,
        duration_s=1.0,
        router_config={"tiers": [{"name": "t0", "level": 0}, {"name": "t1", "level": 1}, {"name": "t2", "level": 2}]},
        router_config_path="router.yaml",
        target="x",
        tier_levels={"t0": 0, "t1": 1, "t2": 2},
        conversations=list(runs),
        cache_source="simulated",
    )


def test_scoring_counts_verdicts_moves_fastpath_and_complaints():
    run = ConversationRun("c", "", "s")
    run.turns = [
        _rec(1, 1, (1, 1), "fresh"),
        _rec(2, 1, (1, 1), "tool_result", fast=True, cached=90, prompt=100),
        _rec(3, 2, (1, 3), "quality_complaint", complaint=True, cached=100, prompt=200),
        _rec(4, 2, (0, 0), "stay", cached=200, prompt=250),
        _rec(5, 0, (1, 2), "switch"),
    ]
    s = score_run(_result(run))
    assert s["labelled_turns"] == 5 and s["level_accuracy"] == 0.6
    assert s["over_provision_rate"] == 0.2 and s["under_provision_rate"] == 0.2
    assert s["mean_level_error"] == 1.5  # over by 2 on turn 4, under by 1 on turn 5
    assert s["moves"] == {"up": 1, "down": 1, "same": 2}
    assert s["fastpath"] == {"expected": 1, "hit": 1} and s["fastpath_accuracy"] == 1.0
    assert s["complaints"] == {"expected": 1, "detected": 1, "true_positive": 1, "false_positive": 0}
    assert s["cache"]["hit_ratio_after_t1"] == round(390 / 650, 3)
    assert s["cost_usd"] == 0.05 and s["tier_switches"] == 2


def test_thresholds_min_and_max():
    summary = {"level_accuracy": 0.7, "under_provision_rate": 0.2, "errors": 0, "jev_ms": {"p95": 900}}
    checks = check_thresholds(
        summary, Thresholds(level_accuracy=0.8, under_provision_rate=0.1, errors=0, jev_p95_ms=1000, cost_usd=1.0)
    )
    assert checks["level_accuracy"] == {"limit": 0.8, "actual": 0.7, "pass": False, "kind": "min"}
    assert checks["under_provision_rate"]["pass"] is False
    assert checks["errors"]["pass"] is True and checks["jev_p95_ms"]["pass"] is True
    assert checks["cost_usd"]["pass"] is None  # metric absent from summary


def test_report_and_markdown_render():
    run = ConversationRun("c", "note", "s")
    run.turns = [_rec(1, 1, (1, 1), "fresh"), _rec(2, 2, (0, 1), "quality_complaint", complaint=False)]
    report = build_report(_result(run), Thresholds(false_complaint_rate=0.0))
    assert report["passed"] is False
    assert report["conversations"][0]["turns"][1]["verdict"] == "over"
    md = to_markdown(report)
    assert "# Router evaluation: d" in md and "| c | 1→2 | 1→0-1 |" in md
    assert "complaint escalation(s) on turns not labelled" in md
    assert "tiers never selected: t0" in md


def test_simulated_cache_is_keyed_per_effort_on_openai_only():
    cache = SimulatedCache(ttl_s=300)
    assert cache.hit("s", "sol", "medium", "openai", 1500, 0.0) == 0  # first sight
    assert cache.hit("s", "sol", "high", "openai", 1600, 1.0) == 0  # different effort: miss
    assert cache.hit("s", "sol", "high", "openai", 1700, 2.0) == 1536  # 1600 rounded down to 128s
    assert cache.hit("s", "sol", "medium", "openai", 1800, 3.0) == 1408  # medium's own entry (1500)
    assert cache.hit("s", "sonnet", "low", "anthropic", 1500, 0.0) == 0
    assert cache.hit("s", "sonnet", "high", "anthropic", 1600, 1.0) == 1500  # anthropic: effort-agnostic
    assert cache.hit("s", "sonnet", "high", "anthropic", 900, 2.0) == 0  # under the caching minimum
    assert cache.hit("s", "sonnet", "high", "anthropic", 1700, 1000.0) == 0  # expired


async def test_simulate_backend_end_to_end(dataset_file):
    ds = load_dataset(dataset_file)
    cfg = RouterConfig(tiers=TIERS, default="sonnet-medium")
    j1 = Judgment(required_tier={1: 0.9, 2: 0.1}, required_confidence=0.9)
    j2 = Judgment(required_tier={1: 0.9, 2: 0.1}, required_confidence=0.9, quality_complaint=0.95, continues_task=0.9)
    j3 = Judgment(required_tier={0: 0.95, 1: 0.05}, required_confidence=0.95)
    backend = SimulateBackend(cfg, RecordedJudge(j1, j2, j3), "router.yaml")
    seen = []
    result = await run_dataset(
        ds, backend, "tests/integration/routers/mixed.yaml", "simulate", on_turn=lambda r, t: seen.append(t)
    )
    [run] = result.conversations
    reasons = [t.reason for t in run.turns]
    assert reasons[0] == "fresh" and reasons[1] == "tool_result" and reasons[2] == "quality_complaint"
    assert run.turns[1].tier == run.turns[0].tier, "tool result stays on the incumbent"
    assert run.turns[1].jev_level is None and run.turns[2].jev_level == 1
    assert cfg.rank(run.turns[2].tier) > cfg.rank(run.turns[1].tier), "complaint escalates at least one rank"
    assert all(t.error is None for t in run.turns) and len(seen) == 4
    assert all(t.cost is not None and t.prompt_tokens > 0 for t in run.turns)
    report = build_report(result, ds.thresholds)
    assert report["summary"]["fastpath"] == {"expected": 1, "hit": 1}
    assert report["summary"]["complaints"]["true_positive"] == 1
