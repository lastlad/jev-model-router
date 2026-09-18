from .ledger import LedgerEntry
from .request import Turn
from .scorer import Decision


def fast_route(turn: Turn, entry: LedgerEntry | None) -> Decision | None:
    if turn.is_tool_result and entry is not None:
        return Decision(entry.tier, "tool_result")
    return None
