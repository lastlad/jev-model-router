"""Build evals/datasets/coding-swebench.yaml from SWE-bench Verified.

The conversations are real issues from the benchmark, so the difficulty distribution is the
benchmark's rather than an invention, and the level labels come from SWE-bench Verified's human
difficulty annotations. Instance ids are pinned, so the dataset is reproducible.

    python evals/build_coding_swebench.py [--out evals/datasets/coding-swebench.yaml]

This produces a dataset for `jev-router eval`, which scores routing decisions. It is not a
SWE-bench run: resolution rate needs the benchmark's own harness, which applies a patch and runs
the repository's tests.
"""

from __future__ import annotations

import argparse
import json
import textwrap
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml

ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=princeton-nlp%2FSWE-bench_Verified&config=default&split=test&offset={off}&length=100"
)

# SWE-bench Verified's human difficulty annotation -> the tier levels that should serve the issue,
# on Jev's 0-3 scale (0 trivial, 1 routine, 2 demanding, 3 frontier). The ranges do not overlap, so
# that level accuracy measures whether the router separates an easy issue from a hard one. Ranges
# that share a level (such as 1-2 and 2-3) cannot: one tier satisfies both.
LEVEL_BY_DIFFICULTY = {
    "<15 min fix": [1, 1],
    "15 min - 1 hour": [1, 2],
    "1-4 hours": [2, 3],
    ">4 hours": [3, 3],
}

# (instance_id, shape). Chosen for a spread of difficulty and repository, and problem statements
# that stand alone without the repository checked out.
INSTANCES: list[tuple[str, str]] = [
    ("django__django-10999", "plain"),
    ("django__django-11099", "trivial-tail"),
    ("astropy__astropy-7336", "plain"),
    ("astropy__astropy-12907", "complaint"),
    ("astropy__astropy-14365", "tools"),
    ("scikit-learn__scikit-learn-10844", "plain"),
    ("django__django-11885", "complaint"),
    ("django__django-12708", "plain"),
    ("sympy__sympy-12489", "trivial-tail"),
    ("pydata__xarray-6992", "deep"),
]

SYSTEM = """You are helping maintain {repo} (version {version}). You do not have the repository \
checked out in this conversation, so reason from the issue text and your knowledge of the project, \
and say when you would need to read a file to be sure. Prefer concrete code over description.

The issue under discussion:

{problem}"""


def _turns(base: list[int], shape: str) -> list[dict[str, Any]]:
    """The turns of one conversation. `base` is the level range the issue itself deserves."""
    analysis = {
        "user": "What is the likely root cause, and which modules would you read first to confirm it?",
        "expected_level": base,
    }
    patch = {"user": "Write the fix as a unified diff.", "expected_level": base}
    tests = {
        "user": "Write the regression test that would have caught this, and say where it belongs.",
        "expected_level": [1, 2],
    }
    commit = {"user": "Commit message for that change, conventional-commits style.", "expected_level": [0, 1]}
    thanks = {"user": "Thanks, that is what I needed.", "expected_level": [0, 1]}

    if shape == "complaint":
        return [
            analysis,
            patch,
            {
                "user": "That patch is wrong. It does not handle the case the reporter describes, and it would "
                "break callers that rely on the current behaviour. Work through it properly this time.",
                "expected_level": base,
                "expected_complaint": True,
            },
            tests,
            commit,
        ]
    if shape == "deep":
        return [
            analysis,
            {
                "user": "Be rigorous: enumerate every invariant this code is supposed to hold, show which one the "
                "issue violates, and prove that your proposed change restores it without weakening the others.",
                "expected_level": [2, 3],
            },
            patch,
            tests,
            commit,
        ]
    if shape == "trivial-tail":
        return [analysis, patch, tests, commit, thanks]
    if shape == "tools":
        return [
            analysis,
            {
                "tool": {
                    "name": "run_tests",
                    "args": {"path": "tests/"},
                    "result": "1 failed, 412 passed\nFAILED tests/test_format.py::test_roundtrip - AssertionError",
                },
                "expected_fastpath": True,
            },
            patch,
            {
                "tool": {"name": "run_tests", "args": {"path": "tests/"}, "result": "413 passed in 4.1s"},
                "expected_fastpath": True,
            },
            tests,
            commit,
        ]
    return [analysis, patch, tests, commit]


def fetch_rows(cache: Path) -> list[dict[str, Any]]:
    if cache.exists():
        return json.loads(cache.read_text())
    rows: list[dict[str, Any]] = []
    for off in range(0, 500, 100):
        with urllib.request.urlopen(ROWS_URL.format(off=off), timeout=60) as r:
            rows += [x["row"] for x in json.load(r)["rows"]]
        time.sleep(0.3)
    cache.write_text(json.dumps(rows))
    return rows


def build(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {r["instance_id"]: r for r in rows}
    contexts: dict[str, str] = {}
    conversations: list[dict[str, Any]] = []
    for instance_id, shape in INSTANCES:
        row = by_id.get(instance_id)
        if row is None:
            raise SystemExit(f"{instance_id} is not in SWE-bench Verified")
        problem = textwrap.shorten(" ".join(row["problem_statement"].split()), 2200, placeholder=" […]")
        key = instance_id.replace("__", "-")
        contexts[key] = SYSTEM.format(repo=row["repo"], version=row["version"], problem=problem)
        base = LEVEL_BY_DIFFICULTY[row["difficulty"]]
        conversations.append(
            {
                "name": key,
                "system": {"context": key},
                "expect": f"SWE-bench Verified, {row['difficulty']}: issue turns at level {base[0]}-{base[1]}, "
                f"tests at 1-2, commit message at 0-1",
                "turns": _turns(base, shape),
            }
        )
    return {
        "name": "coding-swebench",
        "description": (
            "Ten conversations seeded by real SWE-bench Verified issues, across all four of its human difficulty "
            "annotations and five repositories. Each conversation works one issue: root cause, the patch, a "
            "regression test, then a trivial turn. Two conversations carry a quality complaint and one carries "
            "scripted tool results. Levels follow the four-point Jev scale; the mapping from SWE-bench difficulty "
            "is in evals/build_coding_swebench.py. Scores routing decisions only, not issue resolution."
        ),
        "thresholds": {
            "level_accuracy": 0.5,
            "under_provision_rate": 0.15,
            "false_complaint_rate": 0.34,
            "fastpath_accuracy": 1.0,
            "errors": 0,
        },
        "contexts": contexts,
        "conversations": conversations,
    }


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(
    str, lambda d, v: d.represent_scalar("tag:yaml.org,2002:str", v, style="|" if "\n" in v else None)
)
_Dumper.add_representer(
    list,
    lambda d, v: d.represent_sequence(
        "tag:yaml.org,2002:seq", v, flow_style=bool(v) and all(isinstance(x, int | float) for x in v)
    ),
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="evals/datasets/coding-swebench.yaml")
    ap.add_argument("--cache", default=".swebench_verified.json", help="local copy of the benchmark rows")
    args = ap.parse_args()
    ds = build(fetch_rows(Path(args.cache)))
    with open(args.out, "w") as f:
        yaml.dump(ds, f, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=110)
    turns = sum(len(c["turns"]) for c in ds["conversations"])
    print(f"wrote {args.out}: {len(ds['conversations'])} conversations, {turns} turns")


if __name__ == "__main__":
    main()
