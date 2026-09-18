from dataclasses import dataclass, field
from typing import Any, Protocol

from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, Score

TASK_TYPES = [
    "chit_chat",
    "factual_qa",
    "writing",
    "code",
    "math_or_logic",
    "data_analysis",
    "agentic_multistep",
    "other",
]

QUESTIONS: dict[str, Any] = {
    "required_tier": Score(
        instructions="What capability does answering the current_message well require, given the conversation?",
        criteria=[
            "trivial: greeting, acknowledgement, or restating something already in the conversation",
            "routine: a clear single-step task a competent assistant handles reliably",
            "demanding: multi-step reasoning, non-trivial code, or ambiguity that must be resolved",
            "frontier: hard reasoning, long agentic work, or a high cost of error",
        ],
    ),
    "task_type": Choice(
        instructions="What kind of task is the current_message?",
        criteria={t: None for t in TASK_TYPES},
    ),
    "continues_task": Noul(
        instructions=(
            "The current_message continues the task worked on in the recent turns rather than starting a new one."
        )
    ),
    "needs_history": Noul(
        instructions="Answering the current_message well requires details from earlier in the conversation."
    ),
    "quality_complaint": Noul(
        instructions="The user is saying the previous answer was wrong, incomplete, or of poor quality."
    ),
    "expected_output": Score(
        instructions="How long will a good answer to the current_message be?",
        criteria=[
            "short: a sentence or a few lines",
            "medium: a few paragraphs or a small code change",
            "long: a document, a large code change, or many steps",
        ],
    ),
}


@dataclass
class Judgment:
    required_tier: dict[int, float] = field(default_factory=lambda: {1: 1.0})
    required_confidence: float = 1.0
    task_type: str = "other"
    continues_task: float = 0.0
    needs_history: float = 0.0
    quality_complaint: float = 0.0
    expected_output: int = 1

    @property
    def expected_tier(self) -> float:
        return sum(k * v for k, v in self.required_tier.items())

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_tier": {str(k): round(v, 3) for k, v in self.required_tier.items()},
            "required_confidence": round(self.required_confidence, 3),
            "task_type": self.task_type,
            "continues_task": round(self.continues_task, 3),
            "needs_history": round(self.needs_history, 3),
            "quality_complaint": round(self.quality_complaint, 3),
            "expected_output": self.expected_output,
        }


class Judge(Protocol):
    async def judge(self, state: dict[str, Any]) -> Judgment: ...


class JevJudge:
    def __init__(self, model: str = "jev-latest", timeout_s: float = 1.5, api_key: str | None = None) -> None:
        self.client = AsyncTypeSafeClient(api_key=api_key, model=model, timeout=timeout_s)

    async def judge(self, state: dict[str, Any]) -> Judgment:
        r = await self.client.system_one(state=state, questions=QUESTIONS)
        scores, choices, nouls = r.scores, r.choices, r.nouls
        return Judgment(
            required_tier={int(k): v for k, v in scores["required_tier"].probabilities.items()},
            required_confidence=scores["required_tier"].confidence,
            task_type=choices["task_type"].choice,
            continues_task=nouls["continues_task"].noul,
            needs_history=nouls["needs_history"].noul,
            quality_complaint=nouls["quality_complaint"].noul,
            expected_output=max(0, min(2, round(scores["expected_output"].score))),
        )


class RecordedJudge:
    def __init__(self, *judgments: Judgment) -> None:
        self.judgments = list(judgments)
        self.calls: list[dict[str, Any]] = []

    async def judge(self, state: dict[str, Any]) -> Judgment:
        self.calls.append(state)
        return self.judgments.pop(0) if len(self.judgments) > 1 else self.judgments[0]
