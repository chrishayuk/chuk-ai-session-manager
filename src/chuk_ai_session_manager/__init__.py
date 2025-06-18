# chuk_ai_session_manager/__init__.py
"""
CHUK AI Session Manager - Simple Developer API

Quick Start:
    from chuk_ai_session_manager import track_conversation

    # Track any conversation
    await track_conversation("Hello!", "Hi there! How can I help?")

    # Or use the session manager directly
    from chuk_ai_session_manager import SessionManager
    sm = SessionManager()
    await sm.user_says("What's the weather?")
    await sm.ai_responds("It's sunny and 72°F", model="gpt-4")
"""

# Simple API exports - the main things developers need
from chuk_ai_session_manager.api.simple_api import (
    SessionManager,
    track_conversation,
    track_llm_call,
    quick_conversation,
    track_infinite_conversation
)

# Core models for advanced users
from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType

# Storage backend setup
from chuk_ai_session_manager.session_storage import setup_chuk_sessions_storage

__version__ = "0.1.0"

# Main exports - keep it simple
__all__ = [
    # Simple API - most developers start here
    "SessionManager",
    "track_conversation", 
    "track_llm_call",
    "quick_conversation",
    "track_infinite_conversation",
    
    # Core models for advanced usage
    "Session",
    "SessionEvent", 
    "EventSource",
    "EventType",
    
    # Setup
    "setup_chuk_sessions_storage",
]

# Auto-setup with sensible defaults
def _auto_setup():
    """Auto-setup with good defaults if not already configured."""
    try:
        from chuk_ai_session_manager.session_storage import get_backend
        get_backend()  # This will trigger setup if needed
    except Exception:
        # Silently setup with defaults
        setup_chuk_sessions_storage(
            sandbox_id="chuk-ai-session-manager",
            default_ttl_hours=24
        )

# Run auto-setup on import
_auto_setup()