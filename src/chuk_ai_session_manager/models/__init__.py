# chuk_ai_session_manager/models/__init__.py
"""
Core models for the chuk session manager.
"""
# Import each model separately to avoid circular imports
try:
    from chuk_ai_session_manager.models.event_source import EventSource
except ImportError:
    pass

try:
    from chuk_ai_session_manager.models.event_type import EventType
except ImportError:
    pass

try:
    from chuk_ai_session_manager.models.session_event import SessionEvent
except ImportError:
    pass

try:
    from chuk_ai_session_manager.models.session_metadata import SessionMetadata
except ImportError:
    pass

try:
    from chuk_ai_session_manager.models.session_run import SessionRun, RunStatus
except ImportError:
    pass

# Import Session last since it might depend on the above
try:
    from chuk_ai_session_manager.models.session import Session
except ImportError:
    pass

# Define __all__ based on what was successfully imported
__all__ = []

# Check which imports succeeded and add them to __all__
for name in ['EventSource', 'EventType', 'SessionEvent', 'SessionMetadata', 
             'SessionRun', 'RunStatus', 'Session']:
    if name in globals():
        __all__.append(name)