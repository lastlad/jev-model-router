import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import litellm


@dataclass
class Turn:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    system_text: str = ""
    current_text: str = ""
    first_user_text: str = ""
    user: str = ""
    is_tool_result: bool = False
    image_count: int = 0
    prompt_tokens: int = 0
    stable_prefix_tokens: int = 0
    last_assistant_fingerprint: str | None = None


def text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def count_images(messages: list[dict[str, Any]]) -> int:
    n = 0
    for m in messages:
        if isinstance(m.get("content"), list):
            n += sum(1 for p in m["content"] if isinstance(p, dict) and p.get("type") == "image_url")
    return n


def fingerprint_assistant(message: dict[str, Any] | None) -> str | None:
    if not message:
        return None
    ids = [c.get("id") for c in message.get("tool_calls") or [] if c.get("id")]
    if ids:
        return "tc:" + ",".join(ids)
    text = text_of(message.get("content"))
    return "tx:" + hashlib.sha256(text[:2000].encode()).hexdigest()[:24] if text else None


def _openai_messages(data: dict[str, Any], call_type: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if call_type.endswith("anthropic_messages"):
        from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter,
        )

        req, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
            anthropic_message_request=data  # type: ignore[arg-type]
        )
        return [dict(m) for m in req.get("messages") or []], [dict(t) for t in req.get("tools") or []]
    if call_type.endswith("responses"):
        return _responses_messages(data), list(data.get("tools") or [])
    return list(data.get("messages") or []), list(data.get("tools") or [])


def _responses_messages(data: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if data.get("instructions"):
        out.append({"role": "system", "content": data["instructions"]})
    items = data.get("input") or []
    if isinstance(items, str):
        return out + [{"role": "user", "content": items}]
    for it in items:
        t = it.get("type")
        if t == "function_call_output":
            out.append({"role": "tool", "content": str(it.get("output", ""))})
        elif t == "function_call":
            out.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": it.get("call_id"),
                            "function": {
                                "name": it.get("name"),
                                "arguments": it.get("arguments", ""),
                            },
                        }
                    ],
                }
            )
        elif "role" in it:
            content = it.get("content")
            if isinstance(content, list):
                content = [
                    {"type": "text", "text": p.get("text", "")} if p.get("type") in ("input_text", "output_text") else p
                    for p in content
                ]
            out.append({"role": it["role"], "content": content})
    return out


def turn_from_data(data: dict[str, Any], call_type: str, token_model: str) -> Turn:
    messages, tools = _openai_messages(data, call_type)
    system = "\n".join(text_of(m.get("content")) for m in messages if m.get("role") == "system")
    users = [m for m in messages if m.get("role") == "user"]
    assistants = [m for m in messages if m.get("role") == "assistant"]
    last = messages[-1] if messages else {}
    turn = Turn(
        messages=messages,
        tools=tools,
        system_text=system,
        current_text=text_of(last.get("content")) if last.get("role") == "user" else "",
        first_user_text=text_of(users[0].get("content")) if users else "",
        user=str(data.get("user") or ""),
        is_tool_result=last.get("role") == "tool",
        image_count=count_images(messages),
        last_assistant_fingerprint=fingerprint_assistant(assistants[-1]) if assistants else None,
    )
    turn.prompt_tokens = _count(token_model, messages, tools)
    turn.stable_prefix_tokens = _count(token_model, [m for m in messages if m.get("role") == "system"], tools)
    return turn


def _count(model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> int:
    try:
        return litellm.token_counter(model=model, messages=messages, tools=tools or None)  # type: ignore[arg-type]
    except Exception:
        return len(json.dumps(messages) + json.dumps(tools)) // 4
