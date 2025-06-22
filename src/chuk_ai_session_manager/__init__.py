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

Storage Configuration:
    # Default: Memory storage (no Redis required)
    pip install chuk-ai-session-manager
    
    # Redis: For production persistence
    pip install chuk-ai-session-manager[redis]
    export SESSION_PROVIDER=redis
    export SESSION_REDIS_URL=redis://localhost:6379/0
    
    # Environment variables:
    SESSION_PROVIDER=memory (default - fast, no persistence)
    SESSION_PROVIDER=redis  (persistent - requires [redis] extra)
"""

import logging

# Package version
__version__ = "0.5"

# Set up package-level logger
logger = logging.getLogger(__name__)

# Core enums and constants
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType

# Exception classes
from chuk_ai_session_manager.exceptions import (
    SessionManagerError,
    SessionNotFound,
    SessionAlreadyExists,
    InvalidSessionOperation,
    TokenLimitExceeded,
    StorageError,
    ToolProcessingError
)

# Core models
from chuk_ai_session_manager.models.session_metadata import SessionMetadata
from chuk_ai_session_manager.models.session_run import SessionRun, RunStatus
from chuk_ai_session_manager.models.token_usage import TokenUsage, TokenSummary
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.session import Session

# Storage backend
from chuk_ai_session_manager.session_storage import setup_chuk_sessions_storage

# High-level session manager
from chuk_ai_session_manager.session_manager import SessionManager

# Simple API - The main interface most users will use
from chuk_ai_session_manager.api.simple_api import (
    track_conversation,
    track_llm_call,
    quick_conversation,
    track_infinite_conversation,
    track_tool_use,
    get_session_stats,
    get_conversation_history
)

# Advanced components
from chuk_ai_session_manager.infinite_conversation import (
    InfiniteConversationManager,
    SummarizationStrategy
)

from chuk_ai_session_manager.session_aware_tool_processor import SessionAwareToolProcessor

from chuk_ai_session_manager.session_prompt_builder import (
    build_prompt_from_session,
    PromptStrategy,
    truncate_prompt_to_token_limit
)

# Configuration functions
def configure_storage(sandbox_id: str = "chuk-ai-session-manager", 
                     default_ttl_hours: int = 24) -> bool:
    """
    Configure the storage backend.
    
    Args:
        sandbox_id: CHUK Sessions sandbox ID to use
        default_ttl_hours: Default TTL for sessions
        
    Returns:
        True if configuration was successful, False otherwise
        
    Note:
        Storage provider is controlled by SESSION_PROVIDER environment variable:
        - memory (default): Fast, no persistence, no extra dependencies
        - redis: Persistent, requires pip install chuk-ai-session-manager[redis]
    """
    try:
        setup_chuk_sessions_storage(
            sandbox_id=sandbox_id,
            default_ttl_hours=default_ttl_hours
        )
        logger.info(f"Storage configured with sandbox_id='{sandbox_id}'")
        return True
    except Exception as e:
        logger.error(f"Failed to configure storage: {e}")
        return False


def get_version() -> str:
    """Get the package version."""
    return __version__


def is_available() -> dict:
    """
    Check which components are available.
    
    Returns:
        Dictionary showing availability of each component
    """
    # Check if Redis is available
    redis_available = False
    try:
        import redis
        redis_available = True
    except ImportError:
        pass
    
    # Check if tiktoken is available for enhanced token counting
    tiktoken_available = False
    try:
        import tiktoken
        tiktoken_available = True
    except ImportError:
        pass
    
    return {
        "core_enums": True,
        "core_models": True,
        "simple_api": True,
        "storage": True,
        "infinite_context": True,
        "tool_processor": True,
        "prompt_builder": True,
        "token_tracking": True,
        "exceptions": True,
        "session_manager": True,
        "redis_support": redis_available,
        "enhanced_token_counting": tiktoken_available,
    }


def get_storage_info() -> dict:
    """
    Get information about the current storage configuration.
    
    Returns:
        Dictionary with storage configuration details
    """
    import os
    from chuk_ai_session_manager.session_storage import get_backend
    
    try:
        backend = get_backend()
        stats = backend.get_stats()
        
        return {
            "provider": os.getenv("SESSION_PROVIDER", "memory"),
            "backend": stats.get("backend", "unknown"),
            "sandbox_id": stats.get("sandbox_id", "unknown"),
            "redis_url": os.getenv("SESSION_REDIS_URL", "not_set"),
            "stats": stats
        }
    except Exception as e:
        return {
            "provider": os.getenv("SESSION_PROVIDER", "memory"),
            "error": str(e)
        }


# Main exports - everything should be available
__all__ = [
    # Version and utilities
    "__version__",
    "get_version", 
    "is_available",
    "configure_storage",
    "get_storage_info",
    
    # Core enums
    "EventSource",
    "EventType",
    
    # Exception classes
    "SessionManagerError",
    "SessionNotFound", 
    "SessionAlreadyExists",
    "InvalidSessionOperation",
    "TokenLimitExceeded",
    "StorageError",
    "ToolProcessingError",
    
    # Core models
    "Session",
    "SessionEvent",
    "SessionMetadata",
    "SessionRun",
    "RunStatus",
    "TokenUsage",
    "TokenSummary",
    
    # Storage
    "setup_chuk_sessions_storage",
    
    # Primary interfaces - what most users will use
    "SessionManager",
    "track_conversation", 
    "track_llm_call",
    "quick_conversation",
    "track_infinite_conversation",
    "track_tool_use",
    "get_session_stats",
    "get_conversation_history",
    
    # Advanced components
    "InfiniteConversationManager",
    "SummarizationStrategy",
    "SessionAwareToolProcessor",
    "build_prompt_from_session",
    "PromptStrategy",
    "truncate_prompt_to_token_limit",
]

# Auto-setup storage on import
try:
    configure_storage()
    logger.debug("Auto-configured storage backend")
except Exception as e:
    logger.debug(f"Auto-setup skipped: {e}")

# Log successful import with storage info
try:
    storage_info = get_storage_info()
    provider = storage_info.get("provider", "unknown")
    logger.debug(f"CHUK AI Session Manager v{__version__} imported successfully (storage: {provider})")
except Exception:
    logger.debug(f"CHUK AI Session Manager v{__version__} imported successfully")