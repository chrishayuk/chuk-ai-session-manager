# tests/test_chuk_session_storage.py
"""
Tests for CHUK Sessions storage backend.
"""

import pytest
import pytest_asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from chuk_ai_session_manager.session_storage import (
    SessionStorage,
    ChukSessionsStore,
    setup_chuk_sessions_storage,
    get_backend
)
from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.session_metadata import SessionMetadata


@pytest_asyncio.fixture  # ✅ FIXED: Use pytest_asyncio.fixture for async fixtures
async def sample_session():
    """Create a sample AI session for testing."""
    metadata = await SessionMetadata.create(properties={'user_id': 'test_user'})
    
    session = Session(
        id="test_session_123",
        metadata=metadata,
        events=[
            SessionEvent(
                message="Hello AI",
                source=EventSource.USER,
                type=EventType.MESSAGE
            ),
            SessionEvent(
                message="Hello human!",
                source=EventSource.LLM,
                type=EventType.MESSAGE
            )
        ],
        state={'conversation_started': True}
    )
    return session


@pytest.fixture
def mock_chuk_manager():
    """Mock CHUK SessionManager."""
    mock = AsyncMock()
    mock.validate_session = AsyncMock()
    mock.get_session_info = AsyncMock()
    mock.allocate_session = AsyncMock()
    mock.delete_session = AsyncMock()
    mock.extend_session_ttl = AsyncMock()
    mock.get_cache_stats = MagicMock(return_value={'cached_sessions': 0})
    return mock


@pytest.fixture
def backend(mock_chuk_manager):
    """Create SessionStorage with mocked CHUK SessionManager."""
    backend = SessionStorage(sandbox_id="test", default_ttl_hours=24)
    backend.chuk = mock_chuk_manager  # ✅ FIXED: Directly assign the mock
    return backend


class TestChukSessionsBackend:
    """Test SessionStorage functionality."""
    
    @pytest.mark.asyncio
    async def test_get_from_cache(self, backend, sample_session):
        """Test getting session from cache."""
        backend._cache[sample_session.id] = sample_session
        
        result = await backend.get(sample_session.id)
        
        assert result == sample_session
        backend.chuk.validate_session.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_from_chuk_sessions(self, backend, sample_session):
        """Test getting session from CHUK Sessions."""
        # ✅ FIXED: Use model_dump_json for proper datetime serialization
        session_json = sample_session.model_dump_json()
        
        backend.chuk.validate_session.return_value = True
        backend.chuk.get_session_info.return_value = {
            'custom_metadata': {
                'ai_session_data': session_json
            }
        }
        
        result = await backend.get(sample_session.id)
        
        assert result is not None
        assert result.id == sample_session.id
        assert len(result.events) == 2
        assert sample_session.id in backend._cache
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, backend):
        """Test getting non-existent session."""
        backend.chuk.validate_session.return_value = False
        
        result = await backend.get("nonexistent")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_save_session(self, backend, sample_session):
        """Test saving session."""
        backend.chuk.allocate_session.return_value = None  # ✅ FIXED: Return None for success
        
        await backend.save(sample_session)
        
        backend.chuk.allocate_session.assert_called_once()
        call_args = backend.chuk.allocate_session.call_args
        assert call_args.kwargs['session_id'] == sample_session.id
        assert call_args.kwargs['user_id'] == 'test_user'
        assert sample_session.id in backend._cache
    
    @pytest.mark.asyncio
    async def test_delete_session(self, backend, sample_session):
        """Test deleting session."""
        backend._cache[sample_session.id] = sample_session
        
        await backend.delete(sample_session.id)
        
        backend.chuk.delete_session.assert_called_once_with(sample_session.id)
        assert sample_session.id not in backend._cache
    
    @pytest.mark.asyncio
    async def test_list_sessions(self, backend):
        """Test listing sessions."""
        backend._cache = {
            'session_1': MagicMock(),
            'session_2': MagicMock(),
            'other_123': MagicMock()
        }
        
        # Test without prefix
        result = await backend.list_sessions()
        assert set(result) == {'session_1', 'session_2', 'other_123'}
        
        # Test with prefix
        result = await backend.list_sessions(prefix="session_")
        assert set(result) == {'session_1', 'session_2'}
    
    @pytest.mark.asyncio
    async def test_extend_ttl(self, backend):
        """Test extending session TTL."""
        backend.chuk.extend_session_ttl.return_value = True
        
        result = await backend.extend_session_ttl("test_session", 12)
        
        assert result is True
        backend.chuk.extend_session_ttl.assert_called_once_with("test_session", 12)
    
    def test_get_stats(self, backend):
        """Test getting statistics."""
        backend._cache = {'session_1': MagicMock()}
        
        stats = backend.get_stats()
        
        assert stats['backend'] == 'chuk_sessions'
        assert stats['sandbox_id'] == 'test'
        assert stats['cached_ai_sessions'] == 1
        assert 'chuk_sessions_stats' in stats


class TestChukSessionsStore:
    """Test ChukSessionsStore wrapper."""
    
    @pytest.mark.asyncio
    async def test_store_delegates_to_backend(self, sample_session):
        """Test store methods delegate to backend."""
        mock_backend = AsyncMock()
        store = ChukSessionsStore(backend=mock_backend)
        
        # Test all methods delegate properly
        await store.get("test_id")
        mock_backend.get.assert_called_once_with("test_id")
        
        await store.save(sample_session)
        mock_backend.save.assert_called_once_with(sample_session)
        
        await store.delete("test_id")
        mock_backend.delete.assert_called_once_with("test_id")
        
        await store.list_sessions("prefix")
        mock_backend.list_sessions.assert_called_once_with("prefix")


class TestGlobalBackend:
    """Test global backend management."""
    
    def test_setup_creates_backend(self):
        """Test setup function creates backend."""
        with patch('chuk_sessions.SessionManager'):
            backend = setup_chuk_sessions_storage(
                sandbox_id="test_sandbox",
                default_ttl_hours=48
            )
            
            assert isinstance(backend, SessionStorage)
            assert backend.sandbox_id == "test_sandbox"
    
    def test_get_backend_creates_default(self):
        """Test get_backend creates default if none exists."""
        # Reset global backend
        import chuk_ai_session_manager.session_storage as storage_module
        original_backend = storage_module._backend
        storage_module._backend = None
        
        try:
            with patch('chuk_sessions.SessionManager'):
                backend = get_backend()
                
                assert isinstance(backend, SessionStorage)
                assert backend.sandbox_id == "ai-session-manager"
        finally:
            # Restore original backend
            storage_module._backend = original_backend


@pytest.mark.asyncio
async def test_full_lifecycle(backend, sample_session):
    """Test complete session lifecycle."""
    session_id = sample_session.id
    
    # Mock CHUK Sessions responses
    backend.chuk.allocate_session.return_value = None  # Success
    backend.chuk.validate_session.return_value = True
    
    # Save session
    await backend.save(sample_session)
    assert session_id in backend._cache
    
    # Get session (from cache)
    retrieved = await backend.get(session_id)
    assert retrieved == sample_session
    
    # Delete session
    await backend.delete(session_id)
    assert session_id not in backend._cache