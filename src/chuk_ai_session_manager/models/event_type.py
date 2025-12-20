# chuk_ai_session_manager/models/event_type.py
from enum import Enum


class EventType(str, Enum):
    """Type of the session event."""

    # Episodic memory
    MESSAGE = "message"
    SUMMARY = "summary"
    TOOL_CALL = "tool_call"
    REFERENCE = "reference"
    CONTEXT_BRIDGE = "context_bridge"

    # Procedural memory
    TOOL_TRACE = "tool_trace"  # Rich tool invocation record
    TOOL_PATTERN = "tool_pattern"  # Aggregated tool patterns
    TOOL_FIX = "tool_fix"  # Fix relationship (failure -> success)
