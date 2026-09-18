import argparse
import asyncio
import json
import sys

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


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="jev-router")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("replay", _replay), ("state", _state)):
        sp = sub.add_parser(name)
        sp.add_argument("request")
        sp.add_argument("--config", default="deploy/router.yaml")
        sp.add_argument("--litellm-config", default="deploy/config.yaml")
        sp.add_argument("--call-type", default="completion")
        if name == "replay":
            sp.add_argument("--judgment", help="JSON file with a recorded Judgment instead of calling Jev")
        sp.set_defaults(fn=fn)
    args = p.parse_args(argv or sys.argv[1:])
    asyncio.run(args.fn(args))


if __name__ == "__main__":
    main()
