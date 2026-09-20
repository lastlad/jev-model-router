"""Golden dataset schema for router evaluation.

A dataset is a YAML file of scripted conversations. Each turn is a user message or a tool result,
optionally labelled with the routing behaviour it should produce. Assistant replies come from the
model at run time unless a turn pins one with `assistant:`.

    name: routing-golden
    thresholds: { level_accuracy: 0.6, under_provision_rate: 0.05 }
    contexts:
      faq: |
        You are the support assistant for ...
    conversations:
      - name: support-easy
        system: { context: faq }          # or a literal string
        tools: [ ... ]                     # optional OpenAI tool definitions
        expect: level 0 throughout        # free-text note shown in the report
        turns:
          - user: Does it run on a Pi?
            expected_level: 0             # int, or [lo, hi]
          - tool: { name: run_tests, args: { path: tests/ }, result: "1 failed" }
            expected_fastpath: true
          - user: That is wrong, think harder.
            expected_level: [1, 2]
            expected_complaint: true
            assistant: Pinned reply used instead of the live one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class ToolStep(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: str


class Turn(BaseModel):
    user: str | None = None
    tool: ToolStep | None = None
    assistant: str | None = None  # pinned reply; overrides the live one
    expected_level: int | list[int] | None = None
    expected_fastpath: bool | None = None
    expected_complaint: bool | None = None

    @model_validator(mode="after")
    def _one_of(self) -> Turn:
        if (self.user is None) == (self.tool is None):
            raise ValueError("a turn needs exactly one of `user` or `tool`")
        if isinstance(self.expected_level, list) and len(self.expected_level) != 2:
            raise ValueError("expected_level range must be [lo, hi]")
        return self

    @property
    def kind(self) -> str:
        return "tool" if self.tool else "user"

    @property
    def expected_range(self) -> tuple[int, int] | None:
        if self.expected_level is None:
            return None
        if isinstance(self.expected_level, int):
            return self.expected_level, self.expected_level
        return self.expected_level[0], self.expected_level[1]

    def preview(self, n: int = 44) -> str:
        text = self.user if self.user else f"{self.tool.name} → {self.tool.result}"  # type: ignore[union-attr]
        text = " ".join(text.split())
        return text if len(text) <= n else text[: n - 1] + "…"


class Conversation(BaseModel):
    name: str
    system: str
    turns: list[Turn]
    expect: str = ""
    tools: list[dict[str, Any]] = Field(default_factory=list)


class Thresholds(BaseModel):
    """Minimum/maximum values for summary metrics; a run fails if any is violated."""

    level_accuracy: float | None = None  # min
    under_provision_rate: float | None = None  # max
    over_provision_rate: float | None = None  # max
    false_complaint_rate: float | None = None  # max
    fastpath_accuracy: float | None = None  # min
    cost_usd: float | None = None  # max
    jev_p95_ms: float | None = None  # max
    errors: int | None = None  # max


# Threshold metrics that are minimums; every other threshold is a maximum.
MIN_THRESHOLDS = frozenset({"level_accuracy", "fastpath_accuracy"})


class Dataset(BaseModel):
    name: str
    description: str = ""
    thresholds: Thresholds = Thresholds()
    conversations: list[Conversation]

    def select(self, only: list[str]) -> Dataset:
        if not only:
            return self
        convs = [c for c in self.conversations if any(o in c.name for o in only)]
        return self.model_copy(update={"conversations": convs})


def load_dataset(path: str | Path) -> Dataset:
    with open(path) as f:
        raw = yaml.safe_load(f)
    contexts: dict[str, str] = raw.pop("contexts", {}) or {}
    for conv in raw.get("conversations", []):
        system = conv.get("system", "")
        if isinstance(system, dict):
            key = system.get("context")
            if key not in contexts:
                raise ValueError(f"conversation {conv.get('name')!r}: unknown context {key!r}")
            conv["system"] = contexts[key]
    return Dataset.model_validate(raw)


def dump_dataset(ds: Dataset, path: str | Path) -> None:
    """Write a dataset back to YAML (used by --record to pin live replies)."""
    data = ds.model_dump(exclude_none=True, exclude_defaults=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, width=100)
