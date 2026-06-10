"""In-process short-term conversation history, keyed by session id.

Long-term memory lives in Qdrant (M3); this is just the live chat window.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

MAX_TURNS = 40


class SessionStore:
    def __init__(self) -> None:
        self._history: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def history(self, session_id: str) -> list[dict[str, Any]]:
        return self._history[session_id]

    def append(self, session_id: str, message: dict[str, Any]) -> None:
        hist = self._history[session_id]
        hist.append(message)
        if len(hist) > MAX_TURNS:
            del hist[: len(hist) - MAX_TURNS]
