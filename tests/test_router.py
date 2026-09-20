from conftest import chat, judgment

from jev_router.core.judge import RecordedJudge
from jev_router.core.request import fingerprint_assistant
from jev_router.core.router import Router

TOOLS = [
    {
        "type": "function",
        "function": {"name": "run_tests", "parameters": {"type": "object", "properties": {}}},
    }
]


async def test_two_turn_stickiness_and_observe(cfg, ledger):
    router = Router(cfg, RecordedJudge(judgment(level=1, continues_task=0.9, needs_history=0.9)), ledger)
    data = chat(n_turns=2, current="write the function")
    data["litellm_session_id"] = "sess-1"
    d1, t1 = await router.decide(data, "completion", now=100.0)
    assert d1.tier == "sonnet-low" and d1.reason == "fresh" and d1.conversation_id.startswith("sess-1:")
    await router.observe(d1, t1, cached_tokens=15000, response_fingerprint="tx:abc", start=100.0)

    entry = await ledger.get(d1.conversation_id)
    assert entry and entry.cached_tokens == 15000 and entry.history[-1].tier == "sonnet-low"

    d2, _ = await router.decide(data, "completion", now=130.0)
    assert d2.tier == "sonnet-low" and d2.reason == "stay"
    assert {s.tier: s.predicted_cached for s in d2.scores}["sonnet-low"] == 15000


async def test_tool_result_turn_skips_jev(cfg, ledger):
    judge = RecordedJudge(judgment(level=1))
    router = Router(cfg, judge, ledger)
    data = chat(n_turns=1, current="run the tests", tools=TOOLS)
    data["litellm_session_id"] = "sess-2"
    d1, t1 = await router.decide(data, "completion", now=100.0)
    await router.observe(d1, t1, cached_tokens=0, response_fingerprint=None, start=100.0)
    data["messages"] += [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_9", "function": {"name": "run_tests", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_9", "content": "ok"},
    ]
    d2, _ = await router.decide(data, "completion", now=101.0)
    assert d2.reason == "tool_result" and d2.tier == d1.tier and len(judge.calls) == 1


async def test_response_chaining_without_session_id(cfg, ledger):
    router = Router(cfg, RecordedJudge(judgment(level=1)), ledger)
    data = chat(n_turns=0, current="hello")
    data["metadata"] = {"litellm_session_id_omitted": True}
    data["litellm_session_id"] = "random-uuid"
    d1, t1 = await router.decide(data, "completion", now=1.0)
    assert d1.conversation_id.startswith("h:")
    reply = {
        "content": "",
        "tool_calls": [{"id": "call_42", "function": {"name": "f", "arguments": "{}"}}],
    }
    await router.observe(d1, t1, cached_tokens=500, response_fingerprint=fingerprint_assistant(reply), start=1.0)

    data["messages"] += [
        {"role": "assistant", **reply},
        {"role": "tool", "tool_call_id": "call_42", "content": "done"},
        {"role": "user", "content": "next"},
    ]
    d2, _ = await router.decide(data, "completion", now=2.0)
    assert d2.conversation_id == d1.conversation_id


async def test_jev_failure_falls_back(cfg, ledger):
    class Broken:
        async def judge(self, state):
            raise RuntimeError("down")

    router = Router(cfg, Broken(), ledger)
    d, _ = await router.decide(chat(), "completion", now=1.0)
    assert d.tier == cfg.default and d.reason == "jev_unavailable"


async def test_anthropic_messages_call_type(cfg, ledger):
    router = Router(cfg, RecordedJudge(judgment(level=0)), ledger)
    data = {
        "model": "jev-auto",
        "system": "You help.",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }
    d, turn = await router.decide(data, "aanthropic_messages", now=1.0)
    assert turn.current_text == "hi" and turn.system_text == "You help." and d.scores


async def test_mid_conversation_system_entries_do_not_break_identity_or_fast_path(cfg, ledger):
    """Claude Code (mid-conversation-system beta) inserts per-turn `role: system` entries into `messages`,
    and the last message of a tool-result turn is one of them."""
    router = Router(cfg, RecordedJudge(judgment(level=1)), ledger)
    tool_use = {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "calc.py"}}
    tool_result = {"type": "tool_result", "tool_use_id": "toolu_1", "content": "def add(a, b): ..."}
    turn1 = {
        "model": "jev-auto",
        "system": [{"type": "text", "text": "You are Claude Code."}],
        "max_tokens": 100,
        "litellm_session_id": "cc-session",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Read calc.py"}]},
            {"role": "system", "content": "# Environment\nturn 1 counter"},
        ],
    }
    turn2 = {
        **turn1,
        "messages": [
            turn1["messages"][0],
            {"role": "system", "content": "# Environment\nturn 2 counter"},
            {"role": "assistant", "content": [tool_use]},
            {"role": "user", "content": [tool_result]},
            {"role": "system", "content": "<total_tokens>99</total_tokens>"},
        ],
    }
    d1, t1 = await router.decide(turn1, "aanthropic_messages", now=1.0)
    await router.observe(d1, t1, cached_tokens=0, response_fingerprint=None, start=1.0)
    d2, t2 = await router.decide(turn2, "aanthropic_messages", now=2.0)
    assert t1.system_text == t2.system_text == "You are Claude Code."
    assert t1.stable_prefix_tokens == t2.stable_prefix_tokens
    assert d2.conversation_id == d1.conversation_id
    assert t2.is_tool_result and d2.reason == "tool_result" and d2.tier == d1.tier
