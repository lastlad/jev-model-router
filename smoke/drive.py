import json

import httpx

B = "http://127.0.0.1:4001"
H = {"Authorization": "Bearer sk-smoke"}
c = httpx.Client(base_url=B, headers=H, timeout=60)


def chat(msgs, session=None, stream=False, **extra):
    h = {"x-litellm-session-id": session} if session else {}
    body = {"model": "jev-auto", "messages": msgs, "stream": stream, **extra}
    if stream:
        text = ""
        model = None
        with c.stream("POST", "/v1/chat/completions", json=body, headers=h) as r:
            for line in r.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    ch = json.loads(line[6:])
                    model = ch.get("model")
                    text += (ch["choices"][0].get("delta") or {}).get("content") or "" if ch.get("choices") else ""
        return model, text
    r = c.post("/v1/chat/completions", json=body, headers=h).json()
    return r.get("model"), r["choices"][0]["message"]["content"]


sysm = {"role": "system", "content": "You are a coding assistant."}
m = [sysm, {"role": "user", "content": "Refactor the parser to support nested lists."}]
print("T1 chat       ", chat(m, "s1"))
m += [
    {"role": "assistant", "content": "Done, here is the refactor."},
    {"role": "user", "content": "now add a unit test for it"},
]
print("T2 follow-up  ", chat(m, "s1"))
m += [
    {"role": "assistant", "content": "Added test_parser.py"},
    {"role": "user", "content": "that test is wrong, it fails"},
]
print("T3 complaint  ", chat(m, "s1"))
tools = [{"type": "function", "function": {"name": "run_tests", "parameters": {"type": "object", "properties": {}}}}]
m += [
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "run_tests", "arguments": "{}"}}],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "1 failed"},
]
print("T4 tool-result", chat(m, "s1", tools=tools))
m += [{"role": "assistant", "content": "Fixed."}, {"role": "user", "content": "thanks"}]
print("T5 streaming  ", chat(m, "s1", stream=True))
a = c.post(
    "/v1/messages",
    headers={"x-litellm-session-id": "s1"},
    json={
        "model": "jev-auto",
        "max_tokens": 50,
        "system": "You are a coding assistant.",
        "messages": [
            {"role": "user", "content": "Refactor the parser to support nested lists."},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "prove the grammar is unambiguous"},
        ],
    },
).json()
print("T6 /v1/messages", a.get("model"), (a.get("content") or [{}])[0].get("text"), a.get("error"))
r = c.post(
    "/v1/responses",
    headers={"x-litellm-session-id": "s2"},
    json={"model": "jev-auto", "input": "Refactor the parser to support nested lists."},
).json()
print(
    "T7 /v1/responses",
    r.get("model"),
    r.get("error")
    or [o.get("content", [{}])[0].get("text") for o in r.get("output", []) if o.get("type") == "message"],
)
n = [sysm, {"role": "user", "content": "hello, no session header here"}]
model, text = chat(n)
n += [{"role": "assistant", "content": text}, {"role": "user", "content": "and a follow-up"}]
print("T8 chaining   ", chat(n))
