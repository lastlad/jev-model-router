import hashlib
from typing import Any

from .ledger import Ledger
from .request import Turn

SESSION_OMITTED_KEY = "litellm_session_id_omitted"


def client_session_id(data: dict[str, Any]) -> str | None:
    sid = data.get("litellm_session_id")
    md = data.get("metadata") or {}
    if not sid or md.get(SESSION_OMITTED_KEY):
        return None
    return str(sid)


def thread_hash(turn: Turn) -> str:
    raw = "\x1f".join([turn.system_text, turn.first_user_text, turn.user])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


async def resolve_conversation(data: dict[str, Any], turn: Turn, ledger: Ledger) -> str:
    sid = client_session_id(data)
    if sid:
        return f"{sid}:{thread_hash(turn)}"
    if turn.last_assistant_fingerprint:
        found = await ledger.lookup_response(turn.last_assistant_fingerprint)
        if found:
            return found
    return f"h:{thread_hash(turn)}"
