# src/chuk_ai_session_manager/session_storage.py
"""
CHUK Sessions storage backend for AI Session Manager.

Simple integration that stores AI sessions as JSON blobs in CHUK Sessions.
CHUK Sessions handles all storage concerns (memory, Redis, TTL, multi-tenancy).
"""

from __future__ import annotations
import json
import logging
from typing import Any, Dict, List, Optional

# Import the correct class from chuk_sessions
# We need to determine the correct import based on what's actually available
try:
    from chuk_sessions import SessionManager as ChukSessionManager
except ImportError as e:
    raise ImportError(
        f"Cannot import SessionManager from chuk_sessions: {e}\n"
        "Please ensure chuk_sessions is properly installed with: uv add chuk-sessions\n"
        "Or check the available classes in chuk_sessions by running:\n"
        "python -c \"import chuk_sessions; print(dir(chuk_sessions))\""
    ) from e

from chuk_ai_session_manager.models.session import Session

logger = logging.getLogger(__name__)


class SessionStorage:
    """
    CHUK Sessions backend for AI Session Manager.
    
    Stores AI sessions as JSON in CHUK Sessions custom metadata.
    All provider logic is handled by CHUK Sessions.
    """
    
    def __init__(
        self, 
        sandbox_id: str = "ai-session-manager",
        default_ttl_hours: int = 24
    ):
        self.chuk = ChukSessionManager(
            sandbox_id=sandbox_id,
            default_ttl_hours=default_ttl_hours
        )
        self.sandbox_id = sandbox_id
        self._cache: Dict[str, Session] = {}
        
        logger.info(f"AI Session Manager using CHUK Sessions (sandbox: {sandbox_id})")
    
    async def get(self, session_id: str) -> Optional[Session]:
        """Get AI session by ID."""
        if session_id in self._cache:
            return self._cache[session_id]
        
        try:
            # Get session info from CHUK Sessions
            info = await self.chuk.get_session_info(session_id)
            if not info:
                return None
            
            # Check if it's an AI session manager session
            custom_metadata = info.get('custom_metadata', {})
            if custom_metadata.get('session_type') != 'ai_session_manager':
                return None
            
            ai_session_json = custom_metadata.get('ai_session_data')
            if not ai_session_json:
                return None
            
            # Parse the JSON data
            session_data = json.loads(ai_session_json)
            ai_session = Session.model_validate(session_data)
            
            self._cache[session_id] = ai_session
            return ai_session
            
        except Exception as e:
            logger.error(f"Failed to get AI session {session_id}: {e}")
            return None
    
    async def save(self, session: Session) -> None:
        """Save AI session to CHUK Sessions."""
        try:
            # Use Pydantic's model_dump_json which handles datetime serialization properly
            session_json = session.model_dump_json()
            user_id = self._extract_user_id(session)
            
            custom_metadata = {
                'ai_session_data': session_json,
                'event_count': len(session.events),
                'session_type': 'ai_session_manager'
            }
            
            await self.chuk.allocate_session(
                session_id=session.id,
                user_id=user_id,
                custom_metadata=custom_metadata
            )
            
            self._cache[session.id] = session
            
        except Exception as e:
            logger.error(f"Failed to save AI session {session.id}: {e}")
            raise
    
    async def delete(self, session_id: str) -> None:
        """Delete AI session."""
        try:
            await self.chuk.delete_session(session_id)
            self._cache.pop(session_id, None)
        except Exception as e:
            logger.error(f"Failed to delete AI session {session_id}: {e}")
            raise
    
    async def list_sessions(self, prefix: str = "") -> List[str]:
        """List AI session IDs."""
        session_ids = list(self._cache.keys())
        if prefix:
            session_ids = [sid for sid in session_ids if sid.startswith(prefix)]
        return session_ids
    
    def _extract_user_id(self, session: Session) -> Optional[str]:
        """Extract user ID from AI session metadata."""
        try:
            if hasattr(session.metadata, 'properties'):
                return session.metadata.properties.get('user_id')
        except:
            pass
        return None
    
    async def extend_session_ttl(self, session_id: str, additional_hours: int) -> bool:
        """Extend session TTL."""
        return await self.chuk.extend_session_ttl(session_id, additional_hours)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            'backend': 'chuk_sessions',
            'sandbox_id': self.sandbox_id,
            'cached_ai_sessions': len(self._cache),
            'chuk_sessions_stats': self.chuk.get_cache_stats()
        }


# Global backend
_backend: Optional[SessionStorage] = None


def get_backend() -> SessionStorage:
    """Get the global CHUK Sessions backend."""
    global _backend
    if _backend is None:
        _backend = SessionStorage()
    return _backend


def setup_chuk_sessions_storage(
    sandbox_id: str = "ai-session-manager",
    default_ttl_hours: int = 24
) -> SessionStorage:
    """Set up CHUK Sessions as the storage backend."""
    global _backend
    _backend = SessionStorage(
        sandbox_id=sandbox_id,
        default_ttl_hours=default_ttl_hours
    )
    return _backend


class ChukSessionsStore:
    """Storage interface adapter for CHUK Sessions."""
    
    def __init__(self, backend: Optional[SessionStorage] = None):
        self.backend = backend or get_backend()
    
    async def get(self, session_id: str) -> Optional[Session]:
        return await self.backend.get(session_id)
    
    async def save(self, session: Session) -> None:
        await self.backend.save(session)
    
    async def delete(self, session_id: str) -> None:
        await self.backend.delete(session_id)
    
    async def list_sessions(self, prefix: str = "") -> List[str]:
        return await self.backend.list_sessions(prefix)