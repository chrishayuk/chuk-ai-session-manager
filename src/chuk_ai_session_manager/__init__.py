# chuk_ai_session_manager/__init__.py
"""
chuk session manager package.

This package provides session management capabilities.
"""
# Import core components for easier access
try:
    from chuk_ai_session_manager.models.event_source import EventSource
    from chuk_ai_session_manager.models.event_type import EventType
    from chuk_ai_session_manager.models.session import Session
    from chuk_ai_session_manager.models.session_event import SessionEvent
    from chuk_ai_session_manager.models.session_metadata import SessionMetadata
    from chuk_ai_session_manager.models.session_run import SessionRun, RunStatus
except ImportError:
    # During package setup or circular imports, these might not be available
    pass

# Import storage components
try:
    from chuk_ai_session_manager.storage.base import SessionStoreInterface, SessionStoreProvider
except ImportError:
    # During package setup or circular imports, these might not be available
    pass

# Import exceptions
try:
    from chuk_ai_session_manager.exceptions import (
        SessionManagerError,
        SessionNotFound,
        SessionAlreadyExists,
        InvalidSessionOperation,
    )
except ImportError:
    # During package setup or circular imports, these might not be available
    pass

__version__ = "0.1.0"

# Define __all__ only if imports succeeded
__all__ = []

# Check which imports succeeded and add them to __all__
for name in [
    # Models
    'EventSource', 'EventType', 'Session', 'SessionEvent', 
    'SessionMetadata', 'SessionRun', 'RunStatus',
    
    # Storage
    'SessionStoreInterface', 'SessionStoreProvider',
    
    # Exceptions
    'SessionManagerError', 'SessionNotFound', 
    'SessionAlreadyExists', 'InvalidSessionOperation',
]:
    if name in globals():
        __all__.append(name)