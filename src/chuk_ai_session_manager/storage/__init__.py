# chuk_ai_session_manager/storage/__init__.py
"""
Storage module for the chuk session manager.
"""
# Import base components first to avoid circular imports
try:
    from chuk_ai_session_manager.storage.base import SessionStoreInterface, SessionStoreProvider
except ImportError:
    pass

# Try to import providers if available
try:
    from chuk_ai_session_manager.storage.providers.memory import InMemorySessionStore
except ImportError:
    pass

try:
    from chuk_ai_session_manager.storage.providers.file import FileSessionStore, create_file_session_store
except ImportError:
    pass

# Try to import Redis - this is optional
try:
    from chuk_ai_session_manager.storage.providers.redis import RedisSessionStore, create_redis_session_store
    _has_redis = True
except ImportError:
    _has_redis = False

# Define __all__ based on what was successfully imported
__all__ = []

# Basic components
for name in ['SessionStoreInterface', 'SessionStoreProvider', 'InMemorySessionStore']:
    if name in globals():
        __all__.append(name)

# File store
for name in ['FileSessionStore', 'create_file_session_store']:
    if name in globals():
        __all__.append(name)

# Redis store (optional)
if _has_redis:
    __all__.extend(['RedisSessionStore', 'create_redis_session_store'])