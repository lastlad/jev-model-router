"""Score a run against its labels and render the JSON report and Markdown summary."""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .dataset import MIN_THRESHOLDS, Thresholds
from .runner import ConversationRun, RunResult, TurnRecord


def _verdict(rec: TurnRecord) -> str | None:
    if rec.expected_level is None or rec.level is None:
        return None
    lo, hi = rec.expected_level
    return "match" if lo <= rec.level <= hi else "over" if rec.level > hi else "under"


def _level_error(rec: TurnRecord) -> int:
    """Distance from the actual level to the nearest bound of the expected range (0 when inside)."""
    if rec.expected_level is None or rec.level is None:
        return 0
    lo, hi = rec.expected_level
    return lo - rec.level if rec.level < lo else rec.level - hi if rec.level > hi else 0


def _pct(n: float, d: float) -> float | None:
    return round(n / d, 3) if d else None


def _model(tier: str | None) -> str:
    return (tier or "").split("-")[0]


def score_conversation(run: ConversationRun) -> dict[str, Any]:
    turns = run.turns
    levels = [t.level for t in turns if t.level is not None]
    tiers = [t.tier for t in turns if t.tier]
    verdicts = Counter(v for t in turns if (v := _verdict(t)))
    labelled = sum(verdicts.values())
    later = [t for t in turns[1:] if t.prompt_tokens]
    return {
        "name": run.name,
        "expect": run.expect,
        "session": run.session,
        "turns": len(turns),
        "levels": levels,
        "expected": [list(t.expected_level) if t.expected_level else None for t in turns],
        "tiers": tiers,
        "tier_switches": sum(1 for a, b in zip(tiers, tiers[1:], strict=False) if a != b),
        "model_switches": sum(1 for a, b in zip(tiers, tiers[1:], strict=False) if _model(a) != _model(b)),
        "level_accuracy": _pct(verdicts["match"], labelled),
        "over": verdicts["over"],
        "under": verdicts["under"],
        "cache_ratio_after_t1": _pct(sum(t.cached_tokens for t in later), sum(t.prompt_tokens for t in later)),
        "cost_usd": round(sum(t.cost or 0 for t in turns), 4),
        "errors": sum(1 for t in turns if t.error),
    }


def score_run(result: RunResult) -> dict[str, Any]:
    turns = [t for c in result.conversations for t in c.turns]
    judged = [t for t in turns if t.jev_level is not None]
    verdicts = Counter(v for t in turns if (v := _verdict(t)))
    labelled = sum(verdicts.values())
    errors = [e for t in turns if (e := _level_error(t))]
    up = down = same = 0
    for c in result.conversations:
        prev = None
        for t in c.turns:
            if t.level is None:
                continue
            if prev is not None:
                up += t.level > prev
                down += t.level < prev
                same += t.level == prev
            prev = t.level
    fast_expected = [t for t in turns if t.expected_fastpath]
    fast_hit = sum(1 for t in fast_expected if t.reason == "tool_result")
    detected = [t for t in turns if t.reason == "quality_complaint"]
    expected_c = [t for t in turns if t.expected_complaint]
    tp = sum(1 for t in detected if t.expected_complaint)
    fp = len(detected) - tp
    later = [t for c in result.conversations for t in c.turns[1:] if t.prompt_tokens]
    jev = sorted(t.jev_ms for t in judged if t.jev_ms is not None)
    tiers = [t.tier for t in turns if t.tier]
    return {
        "turns": len(turns),
        "judged_turns": len(judged),
        "labelled_turns": labelled,
        "errors": sum(1 for t in turns if t.error),
        "level_accuracy": _pct(verdicts["match"], labelled),
        "over_provision_rate": _pct(verdicts["over"], labelled),
        "under_provision_rate": _pct(verdicts["under"], labelled),
        "mean_level_error": round(statistics.fmean(errors), 3) if errors else 0.0,
        "moves": {"up": up, "down": down, "same": same},
        "tier_switches": sum(
            sum(1 for a, b in zip(c.turns, c.turns[1:], strict=False) if a.tier and b.tier and a.tier != b.tier)
            for c in result.conversations
        ),
        "model_switches": sum(
            sum(
                1
                for a, b in zip(c.turns, c.turns[1:], strict=False)
                if a.tier and b.tier and _model(a.tier) != _model(b.tier)
            )
            for c in result.conversations
        ),
        "fastpath": {"expected": len(fast_expected), "hit": fast_hit},
        "fastpath_accuracy": _pct(fast_hit, len(fast_expected)),
        "complaints": {
            "expected": len(expected_c),
            "detected": len(detected),
            "true_positive": tp,
            "false_positive": fp,
        },
        "complaint_recall": _pct(tp, len(expected_c)),
        "false_complaint_rate": _pct(fp, len(detected)),
        "low_confidence_rate": _pct(sum(1 for t in judged if t.reason == "low_confidence"), len(judged)),
        "cache": {
            "hit_ratio_after_t1": _pct(sum(t.cached_tokens for t in later), sum(t.prompt_tokens for t in later)),
            "source": result.cache_source,
        },
        "cost_usd": round(sum(t.cost or 0 for t in turns), 4),
        "jev_ms": {
            "avg": round(statistics.fmean(jev)) if jev else None,
            "p50": jev[len(jev) // 2] if jev else None,
            "p95": jev[min(len(jev) - 1, int(len(jev) * 0.95))] if jev else None,
        },
        "reasons": dict(Counter(t.reason for t in turns if t.reason)),
        "tier_usage": dict(Counter(tiers).most_common()),
        "jev_level_distribution": dict(sorted(Counter(t.jev_level for t in judged).items())),  # type: ignore[arg-type]
    }


def check_thresholds(summary: dict[str, Any], thresholds: Thresholds) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, limit in thresholds.model_dump(exclude_none=True).items():
        actual = summary["jev_ms"]["p95"] if name == "jev_p95_ms" else summary.get(name)
        if actual is None:
            ok = None
        elif name in MIN_THRESHOLDS:
            ok = actual >= limit
        else:
            ok = actual <= limit
        out[name] = {"limit": limit, "actual": actual, "pass": ok, "kind": "min" if name in MIN_THRESHOLDS else "max"}
    return out


def build_report(result: RunResult, thresholds: Thresholds) -> dict[str, Any]:
    summary = score_run(result)
    checks = check_thresholds(summary, thresholds)
    return {
        "dataset": result.dataset,
        "mode": result.mode,
        "run_id": result.run_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(result.started_at)),
        "duration_s": result.duration_s,
        "target": result.target,
        "router_config_path": result.router_config_path,
        "router_config": result.router_config,
        "summary": summary,
        "thresholds": checks,
        "passed": all(c["pass"] is not False for c in checks.values()),
        "conversations": [
            {**score_conversation(c), "turns": [{**t.to_dict(), "verdict": _verdict(t)} for t in c.turns]}
            for c in result.conversations
        ],
    }


def write_json(report: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2, default=str))


# --------------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------------


def _f(v: Any, pct: bool = False, digits: int = 0) -> str:
    if v is None:
        return "–"
    if pct:
        return f"{100 * v:.{digits}f}%"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def _traj(levels: list[int]) -> str:
    return "→".join(str(lv) for lv in levels) if levels else "–"


def _expected_traj(expected: list[list[int] | None]) -> str:
    parts = []
    for e in expected:
        if e is None:
            parts.append("·")
        elif e[0] == e[1]:
            parts.append(str(e[0]))
        else:
            parts.append(f"{e[0]}-{e[1]}")
    return "→".join(parts)


def to_markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    cfg = report["router_config"]
    obj = cfg.get("objective", {})
    jev_cfg = cfg.get("jev", {})
    over, under = _f(s["over_provision_rate"], pct=True), _f(s["under_provision_rate"], pct=True)
    c = s["complaints"]
    j = s["jev_ms"]
    lines = [
        f"# Router evaluation: {report['dataset']}",
        "",
        f"- **Mode**: {report['mode']} ({report['target']})",
        f"- **Run**: {report['run_id']} at {report['started_at']}, {report['duration_s']}s",
        f"- **Router config**: `{report['router_config_path']}` — "
        f"λ_cost={obj.get('lambda_cost')}, λ_quality={obj.get('lambda_quality')}, "
        f"tier_penalty={cfg.get('tier_penalty')}, switch_cost={cfg.get('switch_cost')}, "
        f"complaint_threshold={jev_cfg.get('complaint_threshold')}, min_confidence={jev_cfg.get('min_confidence')}",
        "- **Tiers**: "
        + ", ".join(f"{t['name']} (L{t['level']})" for t in cfg.get("tiers", []))
        + f"; default `{cfg.get('default')}`",
        f"- **Result**: {'PASS' if report['passed'] else 'FAIL'}",
        "",
        "## Scorecard",
        "",
        "| metric | value |",
        "|---|---|",
        f"| turns / judged / labelled | {s['turns']} / {s['judged_turns']} / {s['labelled_turns']} |",
        f"| level accuracy (within expected range) | {_f(s['level_accuracy'], pct=True)} |",
        f"| over-provisioned / under-provisioned | {over} / {under} |",
        f"| mean level error (mislabelled turns) | {_f(s['mean_level_error'], digits=2)} |",
        f"| level moves up / down / same | {s['moves']['up']} / {s['moves']['down']} / {s['moves']['same']} |",
        f"| tier switches (model switches) | {s['tier_switches']} ({s['model_switches']}) |",
        f"| tool-result fast path | {s['fastpath']['hit']}/{s['fastpath']['expected']} |",
        f"| complaints: expected / detected / true / false | {c['expected']} / {c['detected']} / "
        f"{c['true_positive']} / {c['false_positive']} |",
        f"| low-confidence rate | {_f(s['low_confidence_rate'], pct=True)} |",
        f"| cache hit ratio after turn 1 ({s['cache']['source']}) | {_f(s['cache']['hit_ratio_after_t1'], pct=True)} |",
        f"| cost | ${s['cost_usd']:.4f} |",
        f"| Jev latency avg / p50 / p95 | {_f(j['avg'])} / {_f(j['p50'])} / {_f(j['p95'])} ms |",
        "| tier usage | " + ", ".join(f"{k} {v}" for k, v in s["tier_usage"].items()) + " |",
        "| Jev level distribution | " + ", ".join(f"L{k} {v}" for k, v in s["jev_level_distribution"].items()) + " |",
        "| reasons | " + ", ".join(f"{k} {v}" for k, v in s["reasons"].items()) + " |",
        "",
    ]
    if report["thresholds"]:
        lines += ["## Thresholds", "", "| metric | kind | limit | actual | result |", "|---|---|---|---|---|"]
        for name, c in report["thresholds"].items():
            res = "pass" if c["pass"] else "FAIL" if c["pass"] is False else "n/a"
            lines.append(f"| {name} | {c['kind']} | {c['limit']} | {_f(c['actual'], digits=3)} | {res} |")
        lines.append("")
    lines += [
        "## Conversations",
        "",
        "| conversation | actual levels | expected | acc | switches | cache | cost | note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cv in report["conversations"]:
        lines.append(
            f"| {cv['name']} | {_traj(cv['levels'])} | {_expected_traj(cv['expected'])} | "
            f"{_f(cv['level_accuracy'], pct=True)} | {cv['tier_switches']} ({cv['model_switches']}) | "
            f"{_f(cv['cache_ratio_after_t1'], pct=True)} | ${cv['cost_usd']:.4f} | {cv['expect']} |"
        )
    lines.append("")
    findings = _findings(report)
    if findings:
        lines += ["## Findings", ""] + [f"- {f}" for f in findings] + [""]
    lines += ["## Turns", ""]
    for cv in report["conversations"]:
        lines += [
            f"### {cv['name']}",
            "",
            "| # | turn | tier | L | exp | verdict | reason | Jev | ms | prompt | cached | cost |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for t in cv["turns"]:
            exp = _expected_traj([t["expected_level"]]) if t["expected_level"] else "·"
            jev = f"L{t['jev_level']}@{t['jev_confidence']:.2f}" if t["jev_level"] is not None else "–"
            cached = f"{t['cached_tokens']}/{t['prompt_tokens']}" if t["prompt_tokens"] else "–"
            cost = f"${t['cost']:.4f}" if t["cost"] is not None else "–"
            err = f" ⚠ {t['error']}" if t["error"] else ""
            lines.append(
                f"| {t['idx']} | {t['kind']} {t['preview']}{err} | {t['tier'] or '–'} | {_f(t['level'])} | {exp} | "
                f"{t['verdict'] or '–'} | {t['reason'] or '–'} | {jev} | {_f(t['jev_ms'])} | "
                f"{t['prompt_tokens']} | {cached} | {cost} |"
            )
        lines.append("")
    return "\n".join(lines)


def _findings(report: dict[str, Any]) -> list[str]:
    s = report["summary"]
    out: list[str] = []
    under = [(c["name"], t) for c in report["conversations"] for t in c["turns"] if t["verdict"] == "under"]
    if under:
        out.append(
            f"{len(under)} turn(s) under-provisioned: "
            + "; ".join(f"{n} #{t['idx']} {t['tier']} for expected {t['expected_level']}" for n, t in under[:6])
        )
    fp = [
        (c["name"], t)
        for c in report["conversations"]
        for t in c["turns"]
        if t["reason"] == "quality_complaint" and not t["expected_complaint"]
    ]
    if fp:
        out.append(
            f"{len(fp)} complaint escalation(s) on turns not labelled as complaints: "
            + "; ".join(
                f"{n} #{t['idx']} “{t['preview']}” (complaint={(t['judgment'] or {}).get('quality_complaint')})"
                for n, t in fp[:6]
            )
        )
    missed = [
        (c["name"], t)
        for c in report["conversations"]
        for t in c["turns"]
        if t["expected_complaint"] and t["reason"] != "quality_complaint"
    ]
    if missed:
        out.append(
            f"{len(missed)} labelled complaint(s) not escalated: " + "; ".join(f"{n} #{t['idx']}" for n, t in missed)
        )
    if s["moves"]["down"] == 0 and s["moves"]["up"] > 0:
        out.append("the router never routed down a level; check switch_cost against per-turn cost.")
    unused = [t["name"] for t in report["router_config"].get("tiers", []) if t["name"] not in s["tier_usage"]]
    if unused:
        out.append("tiers never selected: " + ", ".join(unused))
    fast_missed = [
        (c["name"], t)
        for c in report["conversations"]
        for t in c["turns"]
        if t["expected_fastpath"] and t["reason"] != "tool_result"
    ]
    if fast_missed:
        out.append(
            "tool-result turns that did not take the fast path: "
            + "; ".join(f"{n} #{t['idx']} ({t['reason']})" for n, t in fast_missed)
        )
    if s["errors"]:
        out.append(f"{s['errors']} turn(s) errored.")
    return out
