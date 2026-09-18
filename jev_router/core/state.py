import hashlib
import importlib
import json
from dataclasses import asdict
from typing import Any

from .config import StateConfig
from .ledger import HistoryItem
from .request import Turn, text_of


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "…"


def _head_tail(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    h = n * 2 // 3
    return s[:h] + "\n[…elided…]\n" + s[-(n - h) :]


def _recent_item(m: dict[str, Any], cfg: StateConfig) -> dict[str, Any]:
    role = m.get("role", "user")
    text = text_of(m.get("content"))
    item: dict[str, Any] = {"role": role, "chars": len(text)}
    if role == "tool":
        item["text"] = _clip(text, cfg.tool_result_chars)
        item["error"] = "error" in text[:200].lower()
    elif role == "assistant":
        item["text"] = _clip(text, cfg.recent_assistant_chars)
        calls = m.get("tool_calls") or []
        if calls:
            item["tool_calls"] = [{"name": c.get("function", {}).get("name"), "args": _arg_keys(c)} for c in calls]
    else:
        item["text"] = _clip(text, cfg.recent_user_chars)
    return item


def _arg_keys(call: dict[str, Any]) -> list[str]:
    try:
        return sorted(json.loads(call.get("function", {}).get("arguments") or "{}").keys())
    except Exception:
        return []


def _stub(m: dict[str, Any], chars: int) -> dict[str, Any]:
    text = text_of(m.get("content"))
    stub: dict[str, Any] = {"role": m.get("role"), "text": _clip(text, chars), "chars": len(text)}
    names = [c.get("function", {}).get("name") for c in m.get("tool_calls") or []]
    if names:
        stub["tools"] = names
    return stub


def _estimate_tokens(state: dict[str, Any]) -> int:
    return len(json.dumps(state, ensure_ascii=False)) // 4


def build_state(turn: Turn, history: list[HistoryItem], cfg: StateConfig) -> dict[str, Any]:
    body = [m for m in turn.messages if m.get("role") != "system"]
    current = body[-1] if body and body[-1].get("role") == "user" else None
    prior = body[:-1] if current else body
    recent, older = prior[-cfg.recent_messages :], prior[: -cfg.recent_messages or None]
    if not older and len(prior) <= cfg.recent_messages:
        older = []

    stubs = [_stub(m, cfg.first_message_stub_chars if i == 0 else cfg.older_stub_chars) for i, m in enumerate(older)]
    stubs = stubs[-cfg.older_stubs :] if len(stubs) > cfg.older_stubs else stubs

    state: dict[str, Any] = {
        "system_prompt": {
            "head": _clip(turn.system_text, cfg.system_prompt_chars),
            "chars": len(turn.system_text),
            "hash": hashlib.sha256(turn.system_text.encode()).hexdigest()[:12],
        },
        "tools": {
            "names": [t.get("function", {}).get("name", t.get("name")) for t in turn.tools],
            "count": len(turn.tools),
        },
        "history": {
            "turn_count": len(body),
            "older": stubs,
            "recent": [_recent_item(m, cfg) for m in recent],
        },
        "current_message": _head_tail(turn.current_text, cfg.current_message_chars),
        "signals": {
            "image_count": turn.image_count,
            "code_blocks_in_message": turn.current_text.count("```") // 2,
            "pending_tool_result": turn.is_tool_result,
        },
        "routing_history": [asdict(h) for h in history[-cfg.routing_history_entries :]],
    }
    _enforce_budget(state, cfg)
    if cfg.filter:
        mod, _, fn = cfg.filter.rpartition(".")
        state = getattr(importlib.import_module(mod), fn)(state)
    return state


def _enforce_budget(state: dict[str, Any], cfg: StateConfig) -> None:
    hist = state["history"]
    while hist["older"] and _estimate_tokens(state) > cfg.budget_tokens:
        hist["older"].pop(0)
    if _estimate_tokens(state) > cfg.budget_tokens:
        for item in hist["recent"]:
            if "text" in item:
                item["text"] = _clip(item["text"], 200)
    if _estimate_tokens(state) > cfg.budget_tokens:
        state["current_message"] = _head_tail(state["current_message"], cfg.current_message_chars // 2)
