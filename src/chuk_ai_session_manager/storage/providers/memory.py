# chuk_ai_session_manager/storage/providers/memory.py
"""
Async in-memory session storage implementation with improved async semantics.
"""
from typing import Any, Dict, List, Optional
import asyncio
from datetime import datetime

from chuk_ai_session_manager.storage.base import SessionStoreInterface


class InMemorySessionStore(SessionStoreInterface):
    """A simple in-memory store for Session objects with proper async interface.
    
    This implementation stores sessions in a dictionary and is not
    persistent across application restarts. It uses asyncio locks to
    ensure thread safety when multiple coroutines access the store.
    """
    
    def __init__(self) -> None:
        """Initialize an empty in-memory store."""
        self._data: Dict[str, Any] = {}
        self._lock = asyncio.Lock()  # For thread safety in async operations

    async def get(self, session_id: str) -> Optional[Any]:
        """Async: Retrieve a session by its ID, or None if not found."""
        # Read operations don't need locking
        return self._data.get(session_id)

    async def save(self, session: Any) -> None:
        """Async: Save or update a session object in the store."""
        async with self._lock:
            self._data[session.id] = session
            
            # Update metadata timestamp if available
            if hasattr(session, 'metadata') and hasattr(session.metadata, 'update_timestamp'):
                await session.metadata.update_timestamp()
    
    async def delete(self, session_id: str) -> None:
        """Async: Delete a session by its ID."""
        async with self._lock:
            if session_id in self._data:
                del self._data[session_id]
    
    async def list_sessions(self, prefix: str = "") -> List[str]:
        """Async: List all session IDs, optionally filtered by prefix."""
        # Read operations don't need locking
        if not prefix:
            return list(self._data.keys())
        return [sid for sid in self._data.keys() if sid.startswith(prefix)]
    
    async def clear(self) -> None:
        """Async: Clear all sessions from the store."""
        async with self._lock:
            self._data.clear()
            
    async def get_by_property(self, key: str, value: Any) -> List[Any]:
        """
        Async: Find sessions by a specific metadata property value.
        
        Args:
            key: The metadata property key to search for
            value: The value to match
            
        Returns:
            A list of matching sessions
        """
        results = []
        for session in self._data.values():
            if (hasattr(session, 'metadata') and 
                hasattr(session.metadata, 'properties') and
                session.metadata.properties.get(key) == value):
                results.append(session)
        return results
    
    async def get_by_state(self, key: str, value: Any) -> List[Any]:
        """
        Async: Find sessions by a specific state value.
        
        Args:
            key: The state key to search for
            value: The value to match
            
        Returns:
            A list of matching sessions
        """
        results = []
        for session in self._data.values():
            if (hasattr(session, 'state') and 
                session.state.get(key) == value):
                results.append(session)
        return results
    
    async def count(self) -> int:
        """Async: Count the number of sessions in the store."""
        return len(self._data)