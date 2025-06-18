# src/chuk_ai_session_manager/__init__.py
"""
CHUK AI Session Manager - Simple Developer API

A powerful session management system for AI applications that provides:
- Automatic conversation tracking
- Token usage monitoring  
- Tool call logging
- Infinite context support with automatic summarization
- Hierarchical session relationships
- CHUK Sessions backend integration

Quick Start:
    from chuk_ai_session_manager import track_conversation

    # Track any conversation
    await track_conversation("Hello!", "Hi there! How can I help?")

    # Or use the session manager directly
    from chuk_ai_session_manager import SessionManager
    sm = SessionManager()
    await sm.user_says("What's the weather?")
    await sm.ai_responds("It's sunny and 72°F", model="gpt-4")

Infinite Context Example:
    # Automatically handles long conversations with summarization
    sm = SessionManager(infinite_context=True, token_threshold=4000)
    await sm.user_says("Tell me about machine learning")
    await sm.ai_responds("Machine learning is...", model="gpt-4")
    # Session will auto-segment when limits are reached

Advanced Usage:
    # Access the full Session model for detailed control
    from chuk_ai_session_manager import Session, SessionEvent
    
    session = await Session.create()
    event = await SessionEvent.create_with_tokens(
        message="Hello", prompt="Hello", model="gpt-4"
    )
    await session.add_event_and_save(event)
"""

import logging
from typing import Optional

# Package version
__version__ = "0.4"

# Set up package-level logger
logger = logging.getLogger(__name__)

# Core enums and constants (no dependencies)
try:
    from chuk_ai_session_manager.models.event_source import EventSource
    from chuk_ai_session_manager.models.event_type import EventType
except ImportError as e:
    logger.warning(f"Could not import core enums: {e}")
    EventSource = None
    EventType = None

# Storage setup function (no circular dependencies)
try:
    from chuk_ai_session_manager.session_storage import setup_chuk_sessions_storage
except ImportError as e:
    logger.warning(f"Could not import storage setup: {e}")
    setup_chuk_sessions_storage = None

# Exception classes
try:
    from chuk_ai_session_manager.exceptions import (
        SessionManagerError,
        SessionNotFound,
        SessionAlreadyExists,
        InvalidSessionOperation,
        TokenLimitExceeded,
        StorageError,
        ToolProcessingError
    )
except ImportError as e:
    logger.warning(f"Could not import exceptions: {e}")
    # Define minimal fallback exceptions
    class SessionManagerError(Exception):
        """Base exception for session manager errors."""
        pass
    
    SessionNotFound = SessionManagerError
    SessionAlreadyExists = SessionManagerError
    InvalidSessionOperation = SessionManagerError
    TokenLimitExceeded = SessionManagerError
    StorageError = SessionManagerError
    ToolProcessingError = SessionManagerError

# Core models (may have some dependencies)
Session = None
SessionEvent = None
SessionMetadata = None
SessionRun = None
RunStatus = None
TokenUsage = None

try:
    from chuk_ai_session_manager.models.session_metadata import SessionMetadata
except ImportError as e:
    logger.debug(f"Could not import SessionMetadata: {e}")

try:
    from chuk_ai_session_manager.models.session_run import SessionRun, RunStatus
except ImportError as e:
    logger.debug(f"Could not import SessionRun: {e}")

try:
    from chuk_ai_session_manager.models.token_usage import TokenUsage, TokenSummary
except ImportError as e:
    logger.debug(f"Could not import TokenUsage: {e}")
    TokenSummary = None

try:
    from chuk_ai_session_manager.models.session_event import SessionEvent
except ImportError as e:
    logger.debug(f"Could not import SessionEvent: {e}")

try:
    from chuk_ai_session_manager.models.session import Session
except ImportError as e:
    logger.debug(f"Could not import Session: {e}")

# Simple API (highest level, most dependencies)
SessionManager = None
track_conversation = None
track_llm_call = None
quick_conversation = None
track_infinite_conversation = None

try:
    from chuk_ai_session_manager.api.simple_api import (
        SessionManager,
        track_conversation,
        track_llm_call,
        quick_conversation,
        track_infinite_conversation
    )
except ImportError as e:
    logger.warning(f"Could not import simple API: {e}")

# Advanced components (optional)
InfiniteConversationManager = None
SessionAwareToolProcessor = None
build_prompt_from_session = None
PromptStrategy = None

try:
    from chuk_ai_session_manager.infinite_conversation import InfiniteConversationManager
except ImportError as e:
    logger.debug(f"Could not import InfiniteConversationManager: {e}")

try:
    from chuk_ai_session_manager.session_aware_tool_processor import SessionAwareToolProcessor
except ImportError as e:
    logger.debug(f"Could not import SessionAwareToolProcessor: {e}")

try:
    from chuk_ai_session_manager.session_prompt_builder import (
        build_prompt_from_session,
        PromptStrategy
    )
except ImportError as e:
    logger.debug(f"Could not import prompt builder: {e}")


def _auto_setup_storage(sandbox_id: str = "chuk-ai-session-manager", 
                       default_ttl_hours: int = 24) -> bool:
    """
    Auto-setup storage with sensible defaults if not already configured.
    
    Args:
        sandbox_id: CHUK Sessions sandbox ID to use
        default_ttl_hours: Default TTL for sessions
        
    Returns:
        True if setup was successful, False otherwise
    """
    if setup_chuk_sessions_storage is None:
        logger.warning("Storage setup not available - imports failed")
        return False
        
    try:
        # Try to get existing backend first
        from chuk_ai_session_manager.session_storage import get_backend
        get_backend()  # This will trigger setup if needed
        logger.debug("Storage backend already configured")
        return True
    except Exception:
        # Setup with defaults if not configured
        try:
            setup_chuk_sessions_storage(
                sandbox_id=sandbox_id,
                default_ttl_hours=default_ttl_hours
            )
            logger.info(f"Auto-configured storage with sandbox_id='{sandbox_id}'")
            return True
        except Exception as e:
            logger.error(f"Failed to auto-setup storage: {e}")
            return False


def configure_storage(sandbox_id: str = "chuk-ai-session-manager", 
                     default_ttl_hours: int = 24) -> bool:
    """
    Explicitly configure the storage backend.
    
    Args:
        sandbox_id: CHUK Sessions sandbox ID to use
        default_ttl_hours: Default TTL for sessions
        
    Returns:
        True if configuration was successful, False otherwise
    """
    return _auto_setup_storage(sandbox_id, default_ttl_hours)


def get_version() -> str:
    """Get the package version."""
    return __version__


def is_available() -> dict:
    """
    Check which components are available.
    
    Returns:
        Dictionary showing availability of each component
    """
    return {
        "core_enums": EventSource is not None and EventType is not None,
        "core_models": Session is not None and SessionEvent is not None,
        "simple_api": SessionManager is not None,
        "storage": setup_chuk_sessions_storage is not None,
        "infinite_context": InfiniteConversationManager is not None,
        "tool_processor": SessionAwareToolProcessor is not None,
        "prompt_builder": build_prompt_from_session is not None,
        "token_tracking": TokenUsage is not None,
        "exceptions": True,  # Always available (fallbacks defined)
    }


# Main exports - prioritize what works
__all__ = []

# Always available
__all__.extend([
    "__version__",
    "get_version", 
    "is_available",
    "configure_storage",
])

# Add exception classes (always available)
__all__.extend([
    "SessionManagerError",
    "SessionNotFound", 
    "SessionAlreadyExists",
    "InvalidSessionOperation",
    "TokenLimitExceeded",
    "StorageError",
    "ToolProcessingError",
])

# Add available components conditionally
if EventSource is not None:
    __all__.append("EventSource")
if EventType is not None:
    __all__.append("EventType")

if setup_chuk_sessions_storage is not None:
    __all__.append("setup_chuk_sessions_storage")

# Simple API (most important for users)
if SessionManager is not None:
    __all__.extend([
        "SessionManager",
        "track_conversation", 
        "track_llm_call",
        "quick_conversation",
        "track_infinite_conversation",
    ])

# Core models
if Session is not None:
    __all__.append("Session")
if SessionEvent is not None:
    __all__.append("SessionEvent")
if SessionMetadata is not None:
    __all__.append("SessionMetadata")
if SessionRun is not None:
    __all__.extend(["SessionRun", "RunStatus"])
if TokenUsage is not None:
    __all__.append("TokenUsage")
if TokenSummary is not None:
    __all__.append("TokenSummary")

# Advanced components
if InfiniteConversationManager is not None:
    __all__.append("InfiniteConversationManager")
if SessionAwareToolProcessor is not None:
    __all__.append("SessionAwareToolProcessor")
if build_prompt_from_session is not None:
    __all__.extend(["build_prompt_from_session", "PromptStrategy"])

# Auto-setup on import (with error handling)
try:
    _auto_setup_storage()
except Exception as e:
    logger.debug(f"Auto-setup failed (this is normal on first import): {e}")

# Log successful import
available = is_available()
available_count = sum(available.values())
total_count = len(available)
logger.debug(f"CHUK AI Session Manager imported successfully "
            f"({available_count}/{total_count} components available)")

# Show warning if core components are missing
if not available.get("simple_api", False):
    logger.warning(
        "Simple API not available - you may need to install missing dependencies. "
        "Check the logs above for specific import errors."
    )
elif available_count == total_count:
    logger.debug("All components loaded successfully")