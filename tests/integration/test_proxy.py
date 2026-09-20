"""End-to-end plugin wiring through a real LiteLLM proxy with mock deployments and a fake Jev."""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.integration

SYSTEM = {"role": "system", "content": "You are a coding assistant."}
TOOLS = [{"type": "function", "function": {"name": "run_tests", "parameters": {"type": "object", "properties": {}}}}]


def chat(client, messages, session=None, **extra):
    headers = {"x-litellm-session-id": session} if session else {}
    r = client.post("/v1/chat/completions", json={"model": "jev-auto", "messages": messages, **extra}, headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    return body["choices"][0]["message"]["content"], body


def test_fresh_turn_is_rewritten_to_a_tier(client, decisions):
    text, _ = chat(
        client, [SYSTEM, {"role": "user", "content": "Refactor the parser to support nested lists."}], "s-fresh"
    )
    assert text != "from jev-auto default", "request reached the jev-auto deployment instead of a routed tier"
    (d,) = decisions()
    assert d["decision"]["reason"] == "fresh"
    assert d["decision"]["conversation_id"].startswith("s-fresh:")
    assert d["jev_ms"] >= 0 and d["decision"]["judgment"]["required_tier"]["1"] == 0.85


def test_follow_up_stays_and_complaint_escalates(client, decisions):
    m = [SYSTEM, {"role": "user", "content": "Refactor the parser to support nested lists."}]
    chat(client, m, "s-conv")
    m += [{"role": "assistant", "content": "Done."}, {"role": "user", "content": "now add a unit test for it"}]
    chat(client, m, "s-conv")
    m += [{"role": "assistant", "content": "Added test_parser.py"}, {"role": "user", "content": "that test is wrong"}]
    chat(client, m, "s-conv")
    fresh, stay, complaint = [d["decision"] for d in decisions()]
    assert fresh["reason"] == "fresh"
    assert stay["reason"] in ("stay", "switch")
    assert complaint["reason"] == "quality_complaint"
    assert complaint["judgment"]["quality_complaint"] == 0.95
    assert len({d["conversation_id"] for d in (fresh, stay, complaint)}) == 1


def test_tool_result_takes_fast_path_on_incumbent(client, decisions):
    m = [SYSTEM, {"role": "user", "content": "Run the tests and fix failures."}]
    chat(client, m, "s-tool", tools=TOOLS)
    m += [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "run_tests", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "1 failed"},
    ]
    chat(client, m, "s-tool", tools=TOOLS)
    first, tool = [d for d in decisions()]
    assert tool["decision"]["reason"] == "tool_result"
    assert tool["decision"]["tier"] == first["decision"]["tier"]
    assert tool["decision"]["judgment"] is None, "fast path must not call Jev"


def test_frontier_request_routes_to_top_level(client, decisions):
    chat(client, [SYSTEM, {"role": "user", "content": "Prove the grammar is unambiguous."}], "s-prove")
    (d,) = decisions()
    assert d["decision"]["judgment"]["required_tier"]["3"] == 0.85
    assert d["decision"]["tier"] in ("opus-medium", "opus-xhigh")


def test_streaming_is_routed(client, decisions):
    body = {"model": "jev-auto", "messages": [SYSTEM, {"role": "user", "content": "hello there"}], "stream": True}
    text = ""
    with client.stream("POST", "/v1/chat/completions", json=body, headers={"x-litellm-session-id": "s-stream"}) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                if chunk.get("choices"):
                    text += (chunk["choices"][0].get("delta") or {}).get("content") or ""
    assert text.startswith("from ") and text != "from jev-auto default"
    assert decisions()[0]["decision"]["reason"] == "fresh"


def test_anthropic_messages_route(client, decisions):
    r = client.post(
        "/v1/messages",
        headers={"x-litellm-session-id": "s-anthropic"},
        json={
            "model": "jev-auto",
            "max_tokens": 50,
            "system": "You are a coding assistant.",
            "messages": [{"role": "user", "content": "prove the grammar is unambiguous"}],
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["content"][0]["text"] != "from jev-auto default"
    (d,) = decisions()
    assert d["call_type"].endswith("anthropic_messages") and d["decision"]["reason"] == "fresh"


def test_responses_api_route(client, decisions):
    r = client.post(
        "/v1/responses",
        headers={"x-litellm-session-id": "s-responses"},
        json={"model": "jev-auto", "input": "Refactor the parser to support nested lists."},
    )
    assert r.status_code == 200, r.text
    texts = [o["content"][0]["text"] for o in r.json()["output"] if o.get("type") == "message"]
    assert texts and texts[0] != "from jev-auto default"
    (d,) = decisions()
    assert d["call_type"].endswith("responses") and d["decision"]["reason"] == "fresh"


def test_header_less_follow_up_chains_by_response_fingerprint(client, decisions):
    m = [SYSTEM, {"role": "user", "content": "hello, no session header here"}]
    text, _ = chat(client, m)
    m += [{"role": "assistant", "content": text}, {"role": "user", "content": "and a follow-up"}]
    chat(client, m)
    first, second = [d["decision"] for d in decisions()]
    assert first["conversation_id"].startswith("h:")
    assert second["conversation_id"] == first["conversation_id"]
    assert second["reason"] in ("stay", "switch")


def test_observed_event_records_usage_and_cost(client, decisions):
    chat(client, [SYSTEM, {"role": "user", "content": "hello"}], "s-observe")
    (o,) = decisions("observed")
    assert o["conversation_id"].startswith("s-observe:")
    assert o["tier"] and "prompt_tokens" in o and "cached_tokens" in o


def test_routers_dispatch_by_alias_with_separate_ledgers(client, decisions):
    """Two router files are served at once; the same session on each alias gets its own incumbent."""
    m = [SYSTEM, {"role": "user", "content": "Refactor the parser to support nested lists."}]
    text_a, _ = chat(client, m, "s-multi")
    r = client.post(
        "/v1/chat/completions",
        json={"model": "jev-auto-gpt", "messages": m},
        headers={"x-litellm-session-id": "s-multi"},
    )
    assert r.status_code == 200, r.text
    text_b = r.json()["choices"][0]["message"]["content"]
    assert text_b in ("from luna", "from terra", "from sol") and text_b != "from jev-auto-gpt default"
    a, b = [d for d in decisions(at_least=2)]
    assert a["alias"] == "jev-auto" and b["alias"] == "jev-auto-gpt"
    assert a["decision"]["conversation_id"] == b["decision"]["conversation_id"]
    assert b["decision"]["reason"] == "fresh", "the second router must not see the first router's incumbent"
    assert b["decision"]["tier"] in ("luna", "terra", "sol")
    m += [{"role": "assistant", "content": text_a}, {"role": "user", "content": "now add a unit test for it"}]
    chat(client, m, "s-multi")
    assert decisions(at_least=3)[2]["decision"]["reason"] in ("stay", "switch")
