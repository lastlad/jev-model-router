from conftest import chat

from jev_router.core.config import StateConfig
from jev_router.core.request import turn_from_data
from jev_router.core.state import _estimate_tokens, build_state


def test_zoom_by_recency():
    data = chat(n_turns=10, current="fix the failing test")
    data["messages"].insert(
        3,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "run_tests", "arguments": '{"path": "tests/"}'},
                }
            ],
        },
    )
    data["messages"].insert(4, {"role": "tool", "content": "FAILED tests/test_x.py " + "z" * 5000})
    turn = turn_from_data(data, "completion", "anthropic/claude-sonnet-5")
    state = build_state(turn, [], StateConfig())
    assert state["current_message"] == "fix the failing test"
    assert len(state["history"]["recent"]) == 6
    assert state["history"]["older"][0]["text"].startswith("user message 0")
    tool = next(i for i in state["history"]["older"] if i["role"] == "tool")
    assert tool["chars"] > 5000 and len(tool["text"]) <= 121
    assert state["tools"]["count"] == 0 and state["system_prompt"]["chars"] > 0


def test_budget_enforcement_drops_oldest_first():
    data = chat(n_turns=60, current="thanks")
    cfg = StateConfig(budget_tokens=1500)
    turn = turn_from_data(data, "completion", "anthropic/claude-sonnet-5")
    state = build_state(turn, [], cfg)
    assert _estimate_tokens(state) <= cfg.budget_tokens
    assert state["current_message"] == "thanks"


def test_current_message_head_tail():
    data = chat(current="A" * 5000 + "\nPlease refactor this" + "B" * 5000)
    turn = turn_from_data(data, "completion", "anthropic/claude-sonnet-5")
    state = build_state(turn, [], StateConfig())
    assert "elided" in state["current_message"] and state["current_message"].endswith("B" * 100)
