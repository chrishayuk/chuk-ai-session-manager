# chuk_ai_session_manager/models/event_source.py
from enum import Enum

class EventSource(str, Enum):
    """Source of the session event."""
    USER = "user"
    LLM = "llm"
    SYSTEM = "system"