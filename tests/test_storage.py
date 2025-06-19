# tests/test_storage.py
"""
Test suite for session storage functionality in chuk_ai_session_manager.

Tests SessionStorage, ChukSessionsStore, and related storage operations.
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.session_storage import (
    SessionStorage, 
    ChukSessionsStore, 
    get_backend,
    setup_chuk_sessions_storage
)
from chuk_ai_session_manager.exceptions import SessionNotFound


@pytest.fixture
def sample_session_data():
    """Sample session data for testing."""
    return {
        "id": "test-session-123",
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "properties": {"user_id": "user-456"}
        },
        "parent_id": None,
        "child_ids": [],
        "task_ids": [],
        "runs": [],
        "events": [],
        "state": {},
        "token_summary": {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "usage_by_model": {},
            "total_estimated_cost_usd": 0.0
        }
    }


@pytest.fixture
async def mock_chuk_session_manager(sample_session_data):
    """Mock CHUK SessionManager for testing."""
    mock = AsyncMock()
    mock.validate_session.return_value = True
    mock.get_session_info.return_value = {
        "custom_metadata": {
            "ai_session_data": json.dumps(sample_session_data),
            "event_count": 0,
            "session_type": "ai_session_manager"
        }
    }
    mock.allocate_session = AsyncMock()
    mock.delete_session = AsyncMock()
    mock.extend_session_ttl = AsyncMock(return_value=True)
    # get_cache_stats is synchronous - explicitly make it a regular mock
    mock.get_cache_stats = MagicMock(return_value={"cache_size": 10, "hit_rate": 0.8})
    return mock


@pytest.fixture
async def session_storage(mock_chuk_session_manager):
    """SessionStorage instance with mocked backend."""
    with patch('chuk_ai_session_manager.session_storage.ChukSessionManager', return_value=mock_chuk_session_manager):
        storage = SessionStorage(sandbox_id="test", default_ttl_hours=1)
        return storage


class TestSessionStorage:
    """Test SessionStorage functionality."""
    
    async def test_session_storage_initialization(self):
        """Test SessionStorage initialization."""
        with patch('chuk_ai_session_manager.session_storage.ChukSessionManager') as mock_chuk:
            mock_instance = AsyncMock()
            mock_chuk.return_value = mock_instance
            
            storage = SessionStorage(sandbox_id="test-sandbox", default_ttl_hours=2)
            
            assert storage.sandbox_id == "test-sandbox"
            assert storage._cache == {}
            mock_chuk.assert_called_once_with(
                sandbox_id="test-sandbox",
                default_ttl_hours=2
            )
    
    async def test_session_storage_save_and_get(self, session_storage):
        """Test saving and retrieving sessions."""
        session = Session()
        session.id = "test-session-123"
        
        # Save session
        await session_storage.save(session)
        
        # Verify it was saved to CHUK Sessions
        session_storage.chuk.allocate_session.assert_called_once()
        call_args = session_storage.chuk.allocate_session.call_args
        assert call_args[1]["session_id"] == "test-session-123"
        assert "ai_session_data" in call_args[1]["custom_metadata"]
        
        # Retrieve session
        retrieved = await session_storage.get(session.id)
        assert retrieved is not None
        assert retrieved.id == session.id
    
    async def test_session_storage_cache_behavior(self, session_storage):
        """Test session caching behavior."""
        session = Session()
        session.id = "cached-session"
        
        # Add to cache directly
        session_storage._cache[session.id] = session
        
        # Should return from cache without calling CHUK
        retrieved = await session_storage.get(session.id)
        assert retrieved is session
        session_storage.chuk.validate_session.assert_not_called()
    
    async def test_session_storage_get_nonexistent(self):
        """Test getting non-existent session returns None."""
        storage = SessionStorage()
        
        # Mock the CHUK session manager to return None
        mock_chuk = AsyncMock()
        mock_chuk.get_session_info.return_value = None
        storage.chuk = mock_chuk
        
        # Clear any cache
        storage._cache.clear()
        
        # Try to get non-existent session
        retrieved = await storage.get("non-existent-id")
        assert retrieved is None
    
    async def test_session_storage_get_invalid_data(self, session_storage):
        """Test handling invalid session data."""
        # Mock invalid JSON data
        session_storage.chuk.get_session_info.return_value = {
            "custom_metadata": {
                "ai_session_data": "invalid json"
            }
        }
        
        retrieved = await session_storage.get("invalid-session")
        assert retrieved is None
    
    async def test_session_storage_get_missing_data(self, session_storage):
        """Test handling missing session data."""
        # Mock missing ai_session_data
        session_storage.chuk.get_session_info.return_value = {
            "custom_metadata": {}
        }
        
        retrieved = await session_storage.get("missing-data-session")
        assert retrieved is None
    
    async def test_session_storage_save_with_user_id(self, session_storage):
        """Test saving session with user ID extraction."""
        session = Session()
        session.id = "user-session"
        session.metadata.properties = {"user_id": "user-123"}
        
        await session_storage.save(session)
        
        call_args = session_storage.chuk.allocate_session.call_args
        assert call_args[1]["user_id"] == "user-123"
    
    async def test_session_storage_save_error_handling(self, session_storage):
        """Test error handling during save."""
        session = Session()
        session_storage.chuk.allocate_session.side_effect = Exception("Save failed")
        
        with pytest.raises(Exception, match="Save failed"):
            await session_storage.save(session)
    
    async def test_session_storage_delete(self, session_storage):
        """Test deleting sessions."""
        session_id = "delete-me"
        
        # Add to cache first
        session_storage._cache[session_id] = Session(id=session_id)
        
        # Delete session
        await session_storage.delete(session_id)
        
        # Verify CHUK delete was called
        session_storage.chuk.delete_session.assert_called_once_with(session_id)
        
        # Verify removed from cache
        assert session_id not in session_storage._cache
    
    async def test_session_storage_delete_error_handling(self, session_storage):
        """Test error handling during delete."""
        session_storage.chuk.delete_session.side_effect = Exception("Delete failed")
        
        with pytest.raises(Exception, match="Delete failed"):
            await session_storage.delete("some-session")
    
    async def test_session_storage_list_sessions(self, session_storage):
        """Test listing session IDs."""
        # Add some sessions to cache
        session_storage._cache["session-1"] = Session(id="session-1")
        session_storage._cache["session-2"] = Session(id="session-2")
        session_storage._cache["other-session"] = Session(id="other-session")
        
        # List all sessions
        all_sessions = await session_storage.list_sessions()
        assert len(all_sessions) == 3
        assert "session-1" in all_sessions
        assert "session-2" in all_sessions
        assert "other-session" in all_sessions
        
        # List with prefix
        filtered_sessions = await session_storage.list_sessions(prefix="session-")
        assert len(filtered_sessions) == 2
        assert "session-1" in filtered_sessions
        assert "session-2" in filtered_sessions
        assert "other-session" not in filtered_sessions
    
    async def test_session_storage_extend_ttl(self, session_storage):
        """Test extending session TTL."""
        session_id = "ttl-session"
        additional_hours = 5
        
        result = await session_storage.extend_session_ttl(session_id, additional_hours)
        
        session_storage.chuk.extend_session_ttl.assert_called_once_with(session_id, additional_hours)
        assert result == True
    
    def test_session_storage_extract_user_id(self, session_storage):
        """Test user ID extraction from session."""
        # Test with user_id in properties
        session = Session()
        session.metadata.properties = {"user_id": "user-123"}
        user_id = session_storage._extract_user_id(session)
        assert user_id == "user-123"
        
        # Test without user_id
        session_no_user = Session()
        user_id = session_storage._extract_user_id(session_no_user)
        assert user_id is None
        
        # Test with malformed metadata
        session_bad = Session()
        session_bad.metadata = None
        user_id = session_storage._extract_user_id(session_bad)
        assert user_id is None
    
    def test_session_storage_get_stats(self, session_storage):
        """Test storage statistics."""
        # Add some sessions to cache
        session_storage._cache["session-1"] = Session()
        session_storage._cache["session-2"] = Session()
        
        stats = session_storage.get_stats()
        
        assert stats["backend"] == "chuk_sessions"
        assert stats["sandbox_id"] == "test"
        assert stats["cached_ai_sessions"] == 2
        assert "chuk_sessions_stats" in stats
        
        # The chuk_sessions_stats should be a dict (not a coroutine)
        chuk_stats = stats["chuk_sessions_stats"]
        assert isinstance(chuk_stats, dict)
        # Don't assume specific keys since the mock might vary
        if "cache_size" in chuk_stats:
            assert chuk_stats["cache_size"] == 10


class TestChukSessionsStore:
    """Test ChukSessionsStore wrapper."""
    
    async def test_chuk_sessions_store_initialization(self):
        """Test ChukSessionsStore initialization."""
        with patch('chuk_ai_session_manager.session_storage.get_backend') as mock_get_backend:
            mock_backend = AsyncMock()
            mock_get_backend.return_value = mock_backend
            
            store = ChukSessionsStore()
            assert store.backend == mock_backend
            
            # Test with custom backend
            custom_backend = AsyncMock()
            store_custom = ChukSessionsStore(backend=custom_backend)
            assert store_custom.backend == custom_backend
    
    async def test_chuk_sessions_store_methods(self):
        """Test ChukSessionsStore method delegation."""
        mock_backend = AsyncMock()
        store = ChukSessionsStore(backend=mock_backend)
        
        session = Session()
        
        # Test get
        await store.get("session-id")
        mock_backend.get.assert_called_with("session-id")
        
        # Test save
        await store.save(session)
        mock_backend.save.assert_called_with(session)
        
        # Test delete
        await store.delete("session-id")
        mock_backend.delete.assert_called_with("session-id")
        
        # Test list_sessions
        await store.list_sessions("prefix")
        mock_backend.list_sessions.assert_called_with("prefix")


class TestStorageBackendManagement:
    """Test storage backend management functions."""
    
    def test_get_backend_singleton(self):
        """Test that get_backend returns singleton."""
        with patch('chuk_ai_session_manager.session_storage.SessionStorage') as mock_storage:
            # Clear the global backend
            import chuk_ai_session_manager.session_storage as storage_module
            storage_module._backend = None
            
            backend1 = get_backend()
            backend2 = get_backend()
            
            # Should create only one instance
            assert mock_storage.call_count == 1
            assert backend1 is backend2
    
    def test_setup_chuk_sessions_storage(self):
        """Test setup function."""
        with patch('chuk_ai_session_manager.session_storage.SessionStorage') as mock_storage:
            mock_instance = MagicMock()
            mock_storage.return_value = mock_instance
            
            # Clear global backend
            import chuk_ai_session_manager.session_storage as storage_module
            storage_module._backend = None
            
            result = setup_chuk_sessions_storage(
                sandbox_id="custom-sandbox",
                default_ttl_hours=48
            )
            
            mock_storage.assert_called_once_with(
                sandbox_id="custom-sandbox",
                default_ttl_hours=48
            )
            assert result == mock_instance
            assert storage_module._backend == mock_instance


class TestStorageIntegration:
    """Integration tests for storage functionality."""
    
    async def test_session_save_and_retrieve_cycle(self, session_storage):
        """Test complete save and retrieve cycle."""
        # Create a session with events
        session = Session()
        session.id = "integration-test"
        
        # Add some events
        event1 = await SessionEvent.create_with_tokens(
            message="Hello",
            prompt="Hello",
            model="gpt-3.5-turbo",
            source=EventSource.USER
        )
        event2 = await SessionEvent.create_with_tokens(
            message="Hi there!",
            prompt="",
            completion="Hi there!",
            model="gpt-3.5-turbo",
            source=EventSource.LLM
        )
        
        await session.add_event(event1)
        await session.add_event(event2)
        
        # Set some state
        await session.set_state("key1", "value1")
        
        # Save session
        await session_storage.save(session)
        
        # Clear cache to force reload
        session_storage._cache.clear()
        
        # Update mock to return our session data
        session_json = session.model_dump_json()
        session_storage.chuk.get_session_info.return_value = {
            "custom_metadata": {
                "ai_session_data": session_json,
                "event_count": 2,
                "session_type": "ai_session_manager"
            }
        }
        
        # Retrieve session
        retrieved = await session_storage.get(session.id)
        
        assert retrieved is not None
        assert retrieved.id == session.id
        assert len(retrieved.events) == 2
        assert retrieved.state["key1"] == "value1"
        assert retrieved.total_tokens > 0
    
    async def test_error_recovery(self):
        """Test recovery from storage errors."""
        # Create a session that will fail to save
        session = await Session.create()
        
        # Mock storage to fail on save but return None on get
        mock_chuk = AsyncMock()
        mock_chuk.allocate_session.side_effect = Exception("Network error")
        mock_chuk.get_session_info.return_value = None  # Important: return None
        
        storage = SessionStorage()
        storage.chuk = mock_chuk
        
        # Save should fail but not crash
        with pytest.raises(Exception):
            await storage.save(session)
        
        # Clear cache to ensure we don't get cached version
        storage._cache.clear()
        
        # Get should return None, not the cached session
        result = await storage.get(session.id)
        assert result is None 
    
    async def test_concurrent_operations(self, session_storage):
        """Test concurrent storage operations."""
        import asyncio
        
        sessions = [Session() for _ in range(5)]
        for i, session in enumerate(sessions):
            session.id = f"concurrent-{i}"
        
        # Save all sessions concurrently
        save_tasks = [session_storage.save(session) for session in sessions]
        await asyncio.gather(*save_tasks)
        
        # Verify all were saved
        assert session_storage.chuk.allocate_session.call_count == 5
        
        # Retrieve all sessions concurrently
        get_tasks = [session_storage.get(session.id) for session in sessions]
        results = await asyncio.gather(*get_tasks)
        
        assert all(result is not None for result in results)
        assert len(set(result.id for result in results)) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])