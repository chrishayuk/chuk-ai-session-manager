# chuk_ai_session_manager/models/event_source.py
from __future__ import annotations

from enum import Enum


class EventSource(str, Enum):
    """Source of the session event."""

    USER = "user"
    LLM = "llm"
    SYSTEM = "system"
