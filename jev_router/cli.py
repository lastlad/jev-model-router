import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import yaml

from .core.config import load_config
from .core.judge import JevJudge, Judgment, RecordedJudge
from .core.ledger import Ledger, MemoryKV
from .core.router import Router
from .core.state import build_state
from .litellm_plugin import resolve_tiers


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


async def _replay(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    with open(args.litellm_config) as f:
        resolve_tiers(cfg, yaml.safe_load(f).get("model_list", []))
    judge = (
        RecordedJudge(Judgment.from_dict(_load(args.judgment)))
        if args.judgment
        else JevJudge(cfg.jev.model, cfg.jev.timeout_ms / 1000)
    )
    router = Router(cfg, judge, Ledger(MemoryKV(), cfg.cache_ttl_seconds))
    decision, _ = await router.decide(_load(args.request), args.call_type)
    print(f"tier={decision.tier} reason={decision.reason} conversation={decision.conversation_id}")
    if decision.judgment:
        print("judgment:", json.dumps(decision.judgment.to_dict()))
    for s in sorted(decision.scores, key=lambda s: -s.utility):
        print(
            f"  {s.tier:14} utility={s.utility:9.5f} cost={s.cost:8.5f} "
            f"risk={s.quality_risk:6.3f} switch={s.switch_cost:6.3f} cached={s.predicted_cached}"
        )


async def _state(args: argparse.Namespace) -> None:
    from .core.request import turn_from_data

    cfg = load_config(args.config)
    turn = turn_from_data(_load(args.request), args.call_type, cfg.tier(cfg.default).model)
    print(json.dumps(build_state(turn, [], cfg.state), indent=2, ensure_ascii=False))


async def _eval_run(args: argparse.Namespace) -> None:
    from .eval.dataset import dump_dataset, load_dataset
    from .eval.report import build_report, to_markdown, write_json
    from .eval.runner import ClaudeCodeBackend, LiveBackend, make_simulate_backend, run_dataset

    ds = load_dataset(args.dataset).select(args.only)
    if not ds.conversations:
        sys.exit("no conversations selected")
    alias = load_config(args.config).alias
    if args.mode == "live":
        backend = LiveBackend(args.base_url, args.api_key, Path(args.decisions), args.max_tokens, model=alias)
    elif args.mode == "claude-code":
        backend = ClaudeCodeBackend(
            args.base_url, args.api_key, Path(args.decisions), alias, args.claude_bin, args.haiku_model, args.workdir
        )
    else:
        backend = make_simulate_backend(args.config, args.litellm_config)

    def on_conversation(run) -> None:
        print(f"\n== {run.name}  ({run.session})", flush=True)

    def on_turn(run, r) -> None:
        exp = (
            "·"
            if r.expected_level is None
            else (
                str(r.expected_level[0])
                if r.expected_level[0] == r.expected_level[1]
                else f"{r.expected_level[0]}-{r.expected_level[1]}"
            )
        )
        lvl = f"L{r.level}" if r.level is not None else "  "
        jev = f"L{r.jev_level}@{r.jev_confidence:.2f}" if r.jev_level is not None else "-"
        pct = f"{100 * r.cached_tokens / r.prompt_tokens:3.0f}%" if r.prompt_tokens else "  -"
        cost = f"${r.cost:.4f}" if r.cost is not None else "   -   "
        print(
            f"  {r.idx:>2} {r.kind:4} {r.preview:<45} {r.tier or '-':<13} {lvl} exp={exp:<3} {r.reason or '-':<17} "
            f"jev={jev:<8} {r.jev_ms if r.jev_ms is not None else '-':>4}ms "
            f"cached={r.cached_tokens:>5}/{r.prompt_tokens:<5} {pct} {cost}"
            + (f"  ERROR {r.error}" if r.error else ""),
            flush=True,
        )

    try:
        result = await run_dataset(ds, backend, args.config, args.mode, on_turn, on_conversation)
    finally:
        await backend.close()
    report = build_report(result, ds.thresholds)
    md = to_markdown(report)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{ds.name}-{alias}-{args.mode}-{result.run_id}"
    write_json(report, out_dir / f"{stem}.json")
    (out_dir / f"{stem}.md").write_text(md)
    if args.record:
        for conv, run in zip(ds.conversations, result.conversations, strict=True):
            for turn, rec in zip(conv.turns, run.turns, strict=True):
                if rec.assistant_text and not rec.error:
                    turn.assistant = rec.assistant_text
        dump_dataset(ds, args.record)
        print(f"\nrecorded replies into {args.record}")
    print("\n" + md.split("## Turns")[0])
    print(f"report: {out_dir / (stem + '.json')}\nsummary: {out_dir / (stem + '.md')}")
    sys.exit(0 if report["passed"] else 1)


async def _eval_preflight(args: argparse.Namespace) -> None:
    from .eval.preflight import preflight

    checks = preflight(args.config, args.base_url, args.api_key)
    for c in checks:
        print(f"{'ok  ' if c.ok else 'FAIL'} {c.tier:14} {c.model:8} effort={c.effort or '-':7} {c.detail}")
    sys.exit(0 if all(c.ok for c in checks) else 1)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="jev-router")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("replay", _replay), ("state", _state)):
        sp = sub.add_parser(name)
        sp.add_argument("request")
        sp.add_argument("--config", default="deploy/routers/gpt.yaml")
        sp.add_argument("--litellm-config", default="deploy/config.yaml")
        sp.add_argument("--call-type", default="completion")
        if name == "replay":
            sp.add_argument("--judgment", help="JSON file with a recorded Judgment instead of calling Jev")
        sp.set_defaults(fn=fn)

    ev = sub.add_parser("eval", help="evaluate routing quality against a golden dataset").add_subparsers(
        dest="eval_cmd", required=True
    )
    run = ev.add_parser("run", help="play a dataset through the router and write a report")
    run.add_argument("dataset", help="golden dataset YAML")
    run.add_argument(
        "--mode",
        choices=["simulate", "live", "claude-code"],
        default="simulate",
        help="simulate: in-process router, real Jev, no model calls (default); live: the proxy at --base-url; "
        "claude-code: drive `claude -p` against the proxy on your claude.ai login (tool steps skipped)",
    )
    run.add_argument("--config", default="deploy/routers/gpt.yaml", help="the router file to evaluate")
    run.add_argument("--litellm-config", default="deploy/config.yaml", help="model_list for pricing (simulate)")
    run.add_argument("--base-url", default=os.environ.get("LITELLM_BASE", "http://127.0.0.1:4000"))
    run.add_argument("--api-key", default=os.environ.get("LITELLM_MASTER_KEY", "sk-change-me"))
    run.add_argument("--decisions", default="deploy/logs/decisions.jsonl", help="plugin JSONL log (live)")
    run.add_argument("--only", action="append", default=[], help="run conversations whose name contains this")
    run.add_argument("--max-tokens", type=int, default=1200)
    run.add_argument("--out-dir", default="evals/reports")
    run.add_argument("--record", help="write a copy of the dataset with live replies pinned as `assistant:`")
    run.add_argument("--claude-bin", default="claude", help="claude-code mode: the Claude Code executable")
    run.add_argument("--haiku-model", default="cc-haiku", help="claude-code mode: deployment for background calls")
    run.add_argument("--workdir", help="claude-code mode: directory Claude Code runs in (default: a temp dir)")
    run.set_defaults(fn=_eval_run)
    pre = ev.add_parser("preflight", help="check every tier accepts its effort and caches, against a live proxy")
    pre.add_argument("--config", default="deploy/routers/gpt.yaml")
    pre.add_argument("--base-url", default=os.environ.get("LITELLM_BASE", "http://127.0.0.1:4000"))
    pre.add_argument("--api-key", default=os.environ.get("LITELLM_MASTER_KEY", "sk-change-me"))
    pre.set_defaults(fn=_eval_preflight)

    args = p.parse_args(argv or sys.argv[1:])
    asyncio.run(args.fn(args))


if __name__ == "__main__":
    main()
