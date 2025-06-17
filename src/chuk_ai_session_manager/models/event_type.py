# chuk_ai_session_manager/models/event_type.py
from enum import Enum
class EventType(str, Enum):
    """Type of the session event."""
    MESSAGE = "message"
    SUMMARY = "summary"
    TOOL_CALL = "tool_call"
    REFERENCE = "reference"
    CONTEXT_BRIDGE = "context_bridge"