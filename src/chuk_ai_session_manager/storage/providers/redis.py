# chuk_ai_session_manager/storage/providers/redis.py
"""
Async Redis-based session storage implementation.
"""
import json
import logging
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Union, cast

# Note: redis is an optional dependency, so we import it conditionally
try:
    import redis.asyncio as aioredis
    from redis.asyncio import Redis
    from redis.exceptions import RedisError
    AIOREDIS_AVAILABLE = True
except ImportError:
    AIOREDIS_AVAILABLE = False
    # Define a dummy class for type checking
    class Redis:  # type: ignore
        pass
    # Standard redis for sync fallback if needed
    try:
        import redis
        REDIS_AVAILABLE = True
    except ImportError:
        REDIS_AVAILABLE = False

from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.storage.base import SessionStoreInterface
from chuk_ai_session_manager.exceptions import SessionManagerError

# Type variable for serializable models
T = TypeVar('T', bound='Session')

# Setup logging
logger = logging.getLogger(__name__)


class RedisStorageError(SessionManagerError):
    """Raised when Redis storage operations fail."""
    pass


class RedisSessionStore(SessionStoreInterface, Generic[T]):
    """
    An async session store that persists sessions to Redis.
    
    This implementation stores sessions as JSON documents in Redis,
    with configurable key prefixes and expiration.
    """
    
    def __init__(self, 
                redis_client: Any,  # Can be async or sync Redis client
                key_prefix: str = "session:",
                expiration_seconds: Optional[int] = None,
                session_class: Type[T] = Session,
                auto_save: bool = True):
        """
        Initialize the async Redis session store.
        
        Args:
            redis_client: Pre-configured Redis client
            key_prefix: Prefix for Redis keys
            expiration_seconds: Optional TTL for sessions
            session_class: The Session class to use for deserialization
            auto_save: Whether to automatically save on each update
        """
        if not (AIOREDIS_AVAILABLE or REDIS_AVAILABLE):
            raise ImportError(
                "Redis package is not installed. "
                "Install it with 'pip install redis[asyncio]'."
            )
        
        self.redis = redis_client
        self.is_client = AIOREDIS_AVAILABLE and isinstance(redis_client, aioredis.Redis)
        self.key_prefix = key_prefix
        self.expiration_seconds = expiration_seconds
        self.session_class = session_class
        self.auto_save = auto_save
        # In-memory cache for better performance
        self._cache: Dict[str, T] = {}
    
    def _get_key(self, session_id: str) -> str:
        """Get the Redis key for a session ID."""
        return f"{self.key_prefix}{session_id}"
    
    def _json_default(self, obj: Any) -> Any:
        """Handle non-serializable objects in JSON serialization."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    async def get(self, session_id: str) -> Optional[T]:
        """Async: Retrieve a session by its ID."""
        # Check cache first
        if session_id in self._cache:
            return self._cache[session_id]
        
        # If not in cache, try to load from Redis
        key = self._get_key(session_id)
        try:
            if self.is_client:
                data = await self.redis.get(key)
            else:
                # Fall back to sync client in executor if needed
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(None, lambda: self.redis.get(key))
                
            if not data:
                return None
            
            # Convert bytes to str if needed
            if isinstance(data, bytes):
                data = data.decode('utf-8')
                
            session_dict = json.loads(data)
            session = cast(T, self.session_class.model_validate(session_dict))
            
            # Update cache
            self._cache[session_id] = session
            return session
        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load session {session_id} from Redis: {e}")
            return None

    async def save(self, session: T) -> None:
        """Async: Save a session to the store."""
        session_id = session.id
        # Update cache
        self._cache[session_id] = session
        
        if self.auto_save:
            await self._save_to_redis(session)
    
    async def _save_to_redis(self, session: T) -> None:
        """Async: Save a session to Redis."""
        session_id = session.id
        key = self._get_key(session_id)
        
        try:
            # Convert session to JSON
            session_dict = session.model_dump()
            data = json.dumps(session_dict, default=self._json_default)
            
            # Save to Redis with optional expiration
            if self.is_client:
                if self.expiration_seconds:
                    await self.redis.setex(key, self.expiration_seconds, data)
                else:
                    await self.redis.set(key, data)
            else:
                # Fall back to sync client in executor if needed
                loop = asyncio.get_event_loop()
                if self.expiration_seconds:
                    await loop.run_in_executor(
                        None, 
                        lambda: self.redis.setex(key, self.expiration_seconds, data)
                    )
                else:
                    await loop.run_in_executor(
                        None,
                        lambda: self.redis.set(key, data)
                    )
        except (RedisError, TypeError) as e:
            logger.error(f"Failed to save session {session_id} to Redis: {e}")
            raise RedisStorageError(f"Failed to save session {session_id}: {str(e)}")

    async def delete(self, session_id: str) -> None:
        """Async: Delete a session by its ID."""
        # Remove from cache
        if session_id in self._cache:
            del self._cache[session_id]
        
        # Remove from Redis
        key = self._get_key(session_id)
        try:
            if self.is_client:
                await self.redis.delete(key)
            else:
                # Fall back to sync client in executor if needed
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: self.redis.delete(key))
        except RedisError as e:
            logger.error(f"Failed to delete session {session_id} from Redis: {e}")
            raise RedisStorageError(f"Failed to delete session {session_id}: {str(e)}")
    
    async def list_sessions(self, prefix: str = "") -> List[str]:
        """Async: List all session IDs, optionally filtered by prefix."""
        search_pattern = f"{self.key_prefix}{prefix}*"
        try:
            # Get all keys matching the pattern
            if self.is_client:
                keys = await self.redis.keys(search_pattern)
            else:
                # Fall back to sync client in executor if needed
                loop = asyncio.get_event_loop()
                keys = await loop.run_in_executor(None, lambda: self.redis.keys(search_pattern))
                
            # Extract session IDs by removing the prefix
            session_ids = [
                key.decode('utf-8').replace(self.key_prefix, '') if isinstance(key, bytes)
                else key.replace(self.key_prefix, '')
                for key in keys
            ]
            return session_ids
        except RedisError as e:
            logger.error(f"Failed to list sessions from Redis: {e}")
            raise RedisStorageError(f"Failed to list sessions: {str(e)}")
    
    async def flush(self) -> None:
        """Async: Force save all cached sessions to Redis."""
        for session in self._cache.values():
            try:
                await self._save_to_redis(session)
            except RedisStorageError:
                # Already logged in _save_to_redis
                pass
    
    async def clear_cache(self) -> None:
        """Async: Clear the in-memory cache."""
        self._cache.clear()
    
    async def set_expiration(self, session_id: str, seconds: int) -> None:
        """Async: Set or update expiration for a session."""
        key = self._get_key(session_id)
        try:
            if self.is_client:
                await self.redis.expire(key, seconds)
            else:
                # Fall back to sync client in executor if needed
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: self.redis.expire(key, seconds))
        except RedisError as e:
            logger.error(f"Failed to set expiration for session {session_id}: {e}")
            raise RedisStorageError(f"Failed to set expiration for session {session_id}: {str(e)}")


async def create_redis_session_store(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    password: Optional[str] = None,
    key_prefix: str = "session:",
    expiration_seconds: Optional[int] = None,
    session_class: Type[T] = Session,
    auto_save: bool = True,
    **redis_kwargs: Any
) -> RedisSessionStore[T]:
    """
    Create an async Redis-based session store.
    
    Args:
        host: Redis host
        port: Redis port
        db: Redis database number
        password: Optional Redis password
        key_prefix: Prefix for Redis keys
        expiration_seconds: Optional TTL for sessions
        session_class: The Session class to use
        auto_save: Whether to automatically save on each update
        **redis_kwargs: Additional arguments for Redis client
        
    Returns:
        A configured RedisSessionStore
    """
    if AIOREDIS_AVAILABLE:
        # Use async Redis client
        redis_client = aioredis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            **redis_kwargs
        )
    elif REDIS_AVAILABLE:
        # Fall back to sync Redis client
        redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            **redis_kwargs
        )
        logger.warning("Using synchronous Redis client. Install 'redis[asyncio]' for better performance.")
    else:
        raise ImportError("Redis package is not installed. Install with 'pip install redis[asyncio]'")
    
    return RedisSessionStore(
        redis_client=redis_client,
        key_prefix=key_prefix,
        expiration_seconds=expiration_seconds,
        session_class=session_class,
        auto_save=auto_save
    )