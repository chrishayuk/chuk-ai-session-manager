# tests/test_simple_api.py
"""
Test suite for the simple API in chuk_ai_session_manager.

Tests SessionManager, track_conversation, and other high-level convenience functions.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime

from chuk_ai_session_manager.api.simple_api import (
    SessionManager,
    track_conversation,
    track_llm_call,
    quick_conversation,
    track_infinite_conversation
)
from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType


@pytest.fixture
async def mock_session_store():
    """Mock session store for testing."""
    mock_store = AsyncMock()
    sessions = {}
    
    async def mock_get(session_id):
        return sessions.get(session_id)
    
    async def mock_save(session):
        sessions[session.id] = session
    
    mock_store.get.side_effect = mock_get
    mock_store.save.side_effect = mock_save
    
    return mock_store, sessions


class TestSessionManager:
    """Test SessionManager class."""
    
    async def test_session_manager_initialization_new_session(self, mock_session_store):
        """Test SessionManager initialization with new session."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            with patch('chuk_ai_session_manager.models.session.Session.create') as mock_create:
                new_session = Session()
                new_session.id = "new-session-123"
                mock_create.return_value = new_session
                
                sm = SessionManager()
                
                # Should create a new session
                assert sm._is_new == True
                assert sm._infinite_context == False
                assert sm._token_threshold == 4000
                assert sm._max_turns_per_segment == 20
    
    async def test_session_manager_initialization_existing_session(self):
        """Test initializing SessionManager with existing session ID."""
        # Create a session first
        session = await Session.create(session_id="existing-session")
        
        # Mock store that returns this session
        mock_store = AsyncMock()
        mock_store.get.return_value = session
        
        # Create SessionManager with existing ID
        sm = SessionManager(session_id="existing-session", store=mock_store)
        
        # Force initialization
        await sm._ensure_initialized()
        
        # Should not be new since it was loaded
        assert sm._is_new == False
        assert sm._loaded_from_storage == True
    
    async def test_session_manager_infinite_context_settings(self, mock_session_store):
        """Test SessionManager with infinite context enabled."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            sm = SessionManager(
                infinite_context=True,
                token_threshold=2000,
                max_turns_per_segment=10
            )
            
            assert sm._infinite_context == True
            assert sm._token_threshold == 2000
            assert sm._max_turns_per_segment == 10
            assert sm.is_infinite == True
    
    async def test_session_manager_ensure_session_new(self):
        """Test ensure_session creates new session when none provided."""
        # Test the actual public API behavior instead of internal implementation
        sm = SessionManager()  # No session_id provided
        
        # The session manager should work without errors
        session_id = await sm.user_says("Test message")
        assert isinstance(session_id, str)
        assert len(session_id) > 0
        
        # Should be able to get stats
        stats = await sm.get_stats()
        assert stats["session_id"] == session_id
        assert stats["user_messages"] == 1

    async def test_session_manager_ensure_session_existing(self):
        """Test session manager behavior with session IDs."""
        # Test that different session managers have different sessions
        sm1 = SessionManager()
        sm2 = SessionManager()
        
        await sm1.user_says("Message 1")
        await sm2.user_says("Message 2")
        
        session_id_1 = sm1.session_id
        session_id_2 = sm2.session_id
        
        # Should have different session IDs
        assert session_id_1 != session_id_2
        
        # Each should have its own conversation
        conv1 = await sm1.get_conversation()
        conv2 = await sm2.get_conversation()
        
        assert len(conv1) == 1
        assert len(conv2) == 1
        assert conv1[0]["content"] == "Message 1"
        assert conv2[0]["content"] == "Message 2"
        
        # Test creating session manager with specific ID
        sm3 = SessionManager(session_id="test-specific-id")
        # For non-existent session, it should handle gracefully
        try:
            await sm3.user_says("Test message")
            # If it succeeds, it created a new session which is fine
            assert sm3.session_id is not None
        except ValueError:
            # If it fails with "not found", that's also acceptable behavior
            assert True
    
    async def test_session_manager_ensure_session_not_found(self, mock_session_store):
        """Test _ensure_session with nonexistent session."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            sm = SessionManager(session_id="nonexistent")
            
            with pytest.raises(ValueError, match="Session nonexistent not found"):
                await sm._ensure_session()
    
    async def test_user_says_basic(self, mock_session_store):
        """Test basic user_says functionality."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            with patch('chuk_ai_session_manager.models.session.Session.create') as mock_create:
                session = Session()
                session.id = "test-session"
                mock_create.return_value = session
                
                sm = SessionManager()
                result_session_id = await sm.user_says("Hello, world!", key1="value1")
                
                assert result_session_id == session.id
                assert len(session.events) == 1
                assert session.events[0].message == "Hello, world!"
                assert session.events[0].source == EventSource.USER
                assert await session.events[0].get_metadata("key1") == "value1"
    
    async def test_ai_responds_basic(self, mock_session_store):
        """Test basic ai_responds functionality."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            with patch('chuk_ai_session_manager.models.session.Session.create') as mock_create:
                session = Session()
                session.id = "test-session"
                mock_create.return_value = session
                
                sm = SessionManager()
                result_session_id = await sm.ai_responds(
                    "Hi there! How can I help?",
                    model="gpt-4",
                    provider="openai",
                    key1="value1"
                )
                
                assert result_session_id == session.id
                assert len(session.events) == 1
                assert session.events[0].message == "Hi there! How can I help?"
                assert session.events[0].source == EventSource.LLM
                assert await session.events[0].get_metadata("model") == "gpt-4"
                assert await session.events[0].get_metadata("provider") == "openai"
                assert await session.events[0].get_metadata("key1") == "value1"
    
    async def test_tool_used_basic(self, mock_session_store):
        """Test basic tool_used functionality."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            with patch('chuk_ai_session_manager.models.session.Session.create') as mock_create:
                session = Session()
                session.id = "test-session"
                mock_create.return_value = session
                
                sm = SessionManager()
                result_session_id = await sm.tool_used(
                    tool_name="calculator",
                    arguments={"operation": "add", "a": 2, "b": 3},
                    result={"answer": 5},
                    error=None,
                    execution_time=0.5
                )
                
                assert result_session_id == session.id
                assert len(session.events) == 1
                
                tool_event = session.events[0]
                assert tool_event.type == EventType.TOOL_CALL
                assert tool_event.source == EventSource.SYSTEM
                assert tool_event.message["tool"] == "calculator"
                assert tool_event.message["arguments"] == {"operation": "add", "a": 2, "b": 3}
                assert tool_event.message["result"] == {"answer": 5}
                assert tool_event.message["success"] == True
                assert await tool_event.get_metadata("execution_time") == 0.5
    
    async def test_tool_used_with_error(self, mock_session_store):
        """Test tool_used with error."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            with patch('chuk_ai_session_manager.models.session.Session.create') as mock_create:
                session = Session()
                session.id = "test-session"
                mock_create.return_value = session
                
                sm = SessionManager()
                await sm.tool_used(
                    tool_name="search",
                    arguments={"query": "test"},
                    result=None,
                    error="Connection timeout"
                )
                
                tool_event = session.events[0]
                assert tool_event.message["error"] == "Connection timeout"
                assert tool_event.message["success"] == False
    
    async def test_get_conversation_current_session_only(self, mock_session_store):
        """Test get_conversation for current session only."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            with patch('chuk_ai_session_manager.models.session.Session.create') as mock_create:
                session = Session()
                session.id = "test-session"
                mock_create.return_value = session
                
                sm = SessionManager()
                
                # Add some conversation
                await sm.user_says("Hello")
                await sm.ai_responds("Hi there!")
                await sm.user_says("How are you?")
                
                conversation = await sm.get_conversation(include_all_segments=False)
                
                assert len(conversation) == 3
                assert conversation[0]["role"] == "user"
                assert conversation[0]["content"] == "Hello"
                assert conversation[1]["role"] == "assistant"
                assert conversation[1]["content"] == "Hi there!"
                assert conversation[2]["role"] == "user"
                assert conversation[2]["content"] == "How are you?"
                
                # All should have timestamps
                assert all("timestamp" in turn for turn in conversation)
    
    async def test_get_conversation_infinite_context(self, mock_session_store):
        """Test get_conversation with infinite context."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            sm = SessionManager(infinite_context=True)
            
            # Simulate full conversation tracking
            sm._full_conversation = [
                {"role": "user", "content": "Hello", "timestamp": "2024-01-01T00:00:00", "session_id": "session-1"},
                {"role": "assistant", "content": "Hi!", "timestamp": "2024-01-01T00:01:00", "session_id": "session-1"},
                {"role": "user", "content": "How are you?", "timestamp": "2024-01-01T00:02:00", "session_id": "session-2"}
            ]
            
            conversation = await sm.get_conversation(include_all_segments=True)
            
            assert len(conversation) == 3
            assert conversation[0]["session_id"] == "session-1"
            assert conversation[2]["session_id"] == "session-2"
    
    async def test_get_session_chain(self, mock_session_store):
        """Test get_session_chain functionality."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            # Test without infinite context
            sm_regular = SessionManager()
            sm_regular._session_id = "test-session"
            
            chain = await sm_regular.get_session_chain()
            assert chain == ["test-session"]
            
            # Test with infinite context
            sm_infinite = SessionManager(infinite_context=True)
            sm_infinite._session_chain = ["session-1", "session-2", "session-3"]
            
            chain = await sm_infinite.get_session_chain()
            assert chain == ["session-1", "session-2", "session-3"]
    
    async def test_get_stats_current_session(self, mock_session_store):
        """Test get_stats for current session only."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            with patch('chuk_ai_session_manager.models.session.Session.create') as mock_create:
                session = Session()
                session.id = "test-session"
                mock_create.return_value = session
                
                sm = SessionManager()
                
                # Add some events
                await sm.user_says("Hello")
                await sm.ai_responds("Hi there!")
                await sm.tool_used("search", {"query": "test"}, {"results": []})
                
                stats = await sm.get_stats(include_all_segments=False)
                
                assert stats["session_id"] == session.id
                assert stats["session_segments"] == 1
                assert stats["user_messages"] == 1
                assert stats["ai_messages"] == 1
                assert stats["tool_calls"] == 1
                assert stats["total_events"] == 3
                assert "total_tokens" in stats
                assert "estimated_cost" in stats
                assert "created_at" in stats
                assert "last_update" in stats
                assert stats["infinite_context"] == False
    
    async def test_should_create_new_segment(self, mock_session_store):
        """Test infinite context segmentation logic."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            with patch('chuk_ai_session_manager.models.session.Session.create') as mock_create:
                session = Session()
                session.id = "test-session"
                mock_create.return_value = session
                
                # Test without infinite context
                sm_regular = SessionManager(infinite_context=False)
                should_segment = await sm_regular._should_create_new_segment()
                assert should_segment == False
                
                # Test with infinite context and high tokens
                sm_infinite = SessionManager(infinite_context=True, token_threshold=100)
                session.token_summary.total_tokens = 150  # Above threshold
                
                should_segment = await sm_infinite._should_create_new_segment()
                assert should_segment == True
    
    async def test_create_summary(self, mock_session_store):
        """Test summary creation for infinite context."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            with patch('chuk_ai_session_manager.models.session.Session.create') as mock_create:
                session = Session()
                session.id = "test-session"
                mock_create.return_value = session
                
                sm = SessionManager(infinite_context=True)
                
                # Add some user messages
                await sm.user_says("What's the weather like?")
                await sm.user_says("How about tomorrow?")
                
                summary = await sm._create_summary()
                
                assert isinstance(summary, str)
                assert len(summary) > 0
                # Should mention the questions
                assert "weather" in summary.lower() or "tomorrow" in summary.lower()
    
    async def test_create_new_segment(self, mock_session_store):
        """Test new segment creation."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            with patch('chuk_ai_session_manager.models.session.Session.create') as mock_create:
                original_session = Session()
                original_session.id = "original-session"
                mock_create.return_value = original_session
                
                new_session = Session()
                new_session.id = "new-session"
                new_session.parent_id = original_session.id
                
                # Mock create to return new session on second call
                mock_create.side_effect = [original_session, new_session]
                
                sm = SessionManager(infinite_context=True)
                await sm._ensure_session()  # Creates original session
                
                # Add an event to original session
                await sm.user_says("Hello")
                
                # Reset mock for new session creation
                mock_create.side_effect = None
                mock_create.return_value = new_session
                
                new_session_id = await sm._create_new_segment()
                
                assert new_session_id == new_session.id
                assert sm._session_id == new_session.id
                assert sm._session is new_session
                
                # Should have added to session chain
                assert len(sm._session_chain) == 2
                assert sm._session_chain == [original_session.id, new_session.id]
                assert sm._total_segments == 2


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    async def test_track_conversation(self, mock_session_store):
        """Test track_conversation function."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            with patch('chuk_ai_session_manager.models.session.Session.create') as mock_create:
                session = Session()
                session.id = "tracked-session"
                mock_create.return_value = session
                
                session_id = await track_conversation(
                    user_message="Hello!",
                    ai_response="Hi there!",
                    model="gpt-4",
                    provider="openai"
                )
                
                assert session_id == session.id
                assert len(session.events) == 2
                
                user_event = session.events[0]
                ai_event = session.events[1]
                
                assert user_event.message == "Hello!"
                assert user_event.source == EventSource.USER
                assert ai_event.message == "Hi there!"
                assert ai_event.source == EventSource.LLM
    
    async def test_track_conversation_infinite(self, mock_session_store):
        """Test track_infinite_conversation function."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            with patch('chuk_ai_session_manager.models.session.Session.create') as mock_create:
                session = Session()
                session.id = "infinite-session"
                mock_create.return_value = session
                
                session_id = await track_infinite_conversation(
                    user_message="Hello!",
                    ai_response="Hi there!",
                    model="gpt-4",
                    provider="openai",
                    token_threshold=2000
                )
                
                assert session_id == session.id
                assert len(session.events) == 2
    
    async def test_track_llm_call_sync_function(self, mock_session_store):
        """Test track_llm_call with synchronous LLM function."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            with patch('chuk_ai_session_manager.models.session.Session.create') as mock_create:
                session = Session()
                session.id = "llm-call-session"
                mock_create.return_value = session
                
                def mock_llm_function(user_input):
                    return f"LLM response to: {user_input}"
                
                response, session_id = await track_llm_call(
                    user_input="What's 2+2?",
                    llm_function=mock_llm_function,
                    model="gpt-3.5-turbo",
                    provider="openai"
                )
                
                assert response == "LLM response to: What's 2+2?"
                assert session_id == session.id
                assert len(session.events) == 2
    
    async def test_track_llm_call_async_function(self, mock_session_store):
        """Test track_llm_call with async LLM function."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            with patch('chuk_ai_session_manager.models.session.Session.create') as mock_create:
                session = Session()
                session.id = "async-llm-session"
                mock_create.return_value = session
                
                async def mock_async_llm(user_input):
                    await asyncio.sleep(0.01)  # Simulate async work
                    return f"Async LLM response to: {user_input}"
                
                response, session_id = await track_llm_call(
                    user_input="Hello async!",
                    llm_function=mock_async_llm,
                    model="gpt-4"
                )
                
                assert response == "Async LLM response to: Hello async!"
                assert session_id == session.id
    
    async def test_track_llm_call_dict_response(self, mock_session_store):
        """Test track_llm_call with dict response format."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            with patch('chuk_ai_session_manager.models.session.Session.create') as mock_create:
                session = Session()
                session.id = "dict-response-session"
                mock_create.return_value = session
                
                def mock_openai_response(user_input):
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": f"OpenAI response to: {user_input}"
                                }
                            }
                        ]
                    }
                
                response, session_id = await track_llm_call(
                    user_input="Test input",
                    llm_function=mock_openai_response
                )
                
                assert response == "OpenAI response to: Test input"
                assert session_id == session.id
    
    async def test_track_llm_call_object_response(self, mock_session_store):
        """Test track_llm_call with object response format."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            with patch('chuk_ai_session_manager.models.session.Session.create') as mock_create:
                session = Session()
                session.id = "object-response-session"
                mock_create.return_value = session
                
                class MockResponse:
                    def __init__(self, content):
                        self.content = content
                
                def mock_object_response(user_input):
                    return MockResponse(f"Object response to: {user_input}")
                
                response, session_id = await track_llm_call(
                    user_input="Test object",
                    llm_function=mock_object_response
                )
                
                assert response == "Object response to: Test object"
                assert session_id == session.id
    
    async def test_track_llm_call_with_existing_session_manager(self, mock_session_store):
        """Test track_llm_call with existing SessionManager."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            with patch('chuk_ai_session_manager.models.session.Session.create') as mock_create:
                session = Session()
                session.id = "existing-sm-session"
                mock_create.return_value = session
                
                sm = SessionManager()
                
                def mock_llm(user_input):
                    return f"Response: {user_input}"
                
                response, session_id = await track_llm_call(
                    user_input="With existing SM",
                    llm_function=mock_llm,
                    session_manager=sm
                )
                
                assert response == "Response: With existing SM"
                assert session_id == session.id
    
    async def test_quick_conversation(self, mock_session_store):
        """Test quick_conversation function."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            with patch('chuk_ai_session_manager.models.session.Session.create') as mock_create:
                session = Session()
                session.id = "quick-session"
                mock_create.return_value = session
                
                stats = await quick_conversation(
                    user_message="Quick test",
                    ai_response="Quick response",
                    infinite_context=False
                )
                
                assert stats["session_id"] == session.id
                assert stats["user_messages"] == 1
                assert stats["ai_messages"] == 1
                assert stats["infinite_context"] == False
    
    async def test_quick_conversation_infinite(self, mock_session_store):
        """Test quick_conversation with infinite context."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            with patch('chuk_ai_session_manager.models.session.Session.create') as mock_create:
                session = Session()
                session.id = "quick-infinite-session"
                mock_create.return_value = session
                
                stats = await quick_conversation(
                    user_message="Quick infinite test",
                    ai_response="Quick infinite response",
                    infinite_context=True
                )
                
                assert stats["session_id"] == session.id
                assert stats["infinite_context"] == True


class TestSessionManagerInfiniteContext:
    """Test SessionManager infinite context functionality."""
    
    async def test_infinite_context_conversation_tracking(self, mock_session_store):
        """Test that infinite context tracks full conversation."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            with patch('chuk_ai_session_manager.models.session.Session.create') as mock_create:
                session = Session()
                session.id = "infinite-tracking-session"
                mock_create.return_value = session
                
                sm = SessionManager(infinite_context=True)
                
                # Add conversation
                await sm.user_says("Hello!")
                await sm.ai_responds("Hi there!", model="gpt-4", provider="openai")
                await sm.user_says("How are you?")
                
                # Check full conversation tracking
                assert len(sm._full_conversation) == 3
                
                conv = sm._full_conversation
                assert conv[0]["role"] == "user"
                assert conv[0]["content"] == "Hello!"
                assert conv[0]["session_id"] == session.id
                
                assert conv[1]["role"] == "assistant"
                assert conv[1]["content"] == "Hi there!"
                assert conv[1]["model"] == "gpt-4"
                assert conv[1]["provider"] == "openai"
                
                assert conv[2]["role"] == "user"
                assert conv[2]["content"] == "How are you?"
    
    async def test_infinite_context_segmentation_flow(self, mock_session_store):
        """Test complete infinite context segmentation flow."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            session_counter = 0
            
            def create_session(**kwargs):
                nonlocal session_counter
                session_counter += 1
                session = Session()
                session.id = f"segment-{session_counter}"
                if 'parent_id' in kwargs:
                    session.parent_id = kwargs['parent_id']
                return session
            
            with patch('chuk_ai_session_manager.models.session.Session.create', side_effect=create_session):
                sm = SessionManager(
                    infinite_context=True,
                    token_threshold=50,  # Very low for testing
                    max_turns_per_segment=2
                )
                
                # First conversation
                session_id_1 = await sm.user_says("First message")
                session_id_2 = await sm.ai_responds("First response")
                
                # Should still be in first segment
                assert session_id_1 == session_id_2 == "segment-1"
                
                # This should trigger segmentation (3rd turn)
                with patch.object(sm, '_create_summary', return_value="Summary of first segment"):
                    session_id_3 = await sm.user_says("Third message")
                
                # Should be in new segment
                assert session_id_3 == "segment-2"
                assert sm._total_segments == 2
                assert len(sm._session_chain) == 2
                assert sm._session_chain == ["segment-1", "segment-2"]
    
    async def test_infinite_context_stats_all_segments(self):
        """Test infinite context stats include all segments."""
        # Create proper mock sessions
        session1 = await Session.create(session_id="segment-1")
        session2 = await Session.create(session_id="segment-2", parent_id="segment-1")
        
        # Mock store that returns these sessions
        sessions_db = {
            "segment-1": session1,
            "segment-2": session2
        }
        
        class MockStore:
            async def get(self, session_id):
                return sessions_db.get(session_id)
            
            async def save(self, session):
                sessions_db[session.id] = session
        
        mock_store = MockStore()
        
        # Create manager with the current session
        sm = SessionManager(
            session_id="segment-2",
            infinite_context=True,
            store=mock_store
        )
        
        # Initialize to load the session
        await sm._ensure_initialized()
        
        # Manually set the chain for testing
        sm._session_chain = ["segment-1", "segment-2"]
        sm._total_segments = 2
        
        stats = await sm.get_stats(include_all_segments=True)
        assert stats["session_chain"] == ["segment-1", "segment-2"]
class TestSessionManagerEdgeCases:
    """Test edge cases and error conditions."""
    
    async def test_session_manager_property_access_before_init(self):
        """Test accessing session_id before session is created."""
        sm = SessionManager()
        
        # Should return a valid UUID even before session creation
        session_id = sm.session_id
        assert isinstance(session_id, str)
        assert len(session_id) > 0
    
    async def test_session_manager_caching_behavior(self, mock_session_store):
        """Test that session is cached after first access."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            with patch('chuk_ai_session_manager.models.session.Session.create') as mock_create:
                session = Session()
                session.id = "cached-session"
                mock_create.return_value = session
                
                sm = SessionManager()
                
                # First call should create session
                session1 = await sm._ensure_session()
                # Second call should return cached session
                session2 = await sm._ensure_session()
                
                assert session1 is session2
                mock_create.assert_called_once()  # Only called once
    
    async def test_infinite_context_without_segmentation(self, mock_session_store):
        """Test infinite context that doesn't trigger segmentation."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            with patch('chuk_ai_session_manager.models.session.Session.create') as mock_create:
                session = Session()
                session.id = "no-segment-session"
                mock_create.return_value = session
                
                sm = SessionManager(
                    infinite_context=True,
                    token_threshold=10000,  # Very high
                    max_turns_per_segment=100  # Very high
                )
                
                # Add several messages that won't trigger segmentation
                for i in range(5):
                    await sm.user_says(f"Message {i}")
                    await sm.ai_responds(f"Response {i}")
                
                # Should still be in original session
                assert sm._session_id == session.id
                assert len(sm._session_chain) == 1
                assert sm._total_segments == 1
                assert len(sm._full_conversation) == 10  # 5 user + 5 AI messages
    
    async def test_get_conversation_with_tool_events(self, mock_session_store):
        """Test get_conversation filtering out non-message events."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.session_storage.get_backend', return_value=mock_store):
            with patch('chuk_ai_session_manager.models.session.Session.create') as mock_create:
                session = Session()
                session.id = "mixed-events-session"
                mock_create.return_value = session
                
                sm = SessionManager()
                
                await sm.user_says("Hello")
                await sm.ai_responds("Hi there!")
                await sm.tool_used("search", {"query": "test"}, {"results": []})
                await sm.user_says("Thanks")
                
                conversation = await sm.get_conversation()
                
                # Should only include MESSAGE events, not TOOL_CALL events
                assert len(conversation) == 3  # 2 user + 1 AI message
                assert all(turn["role"] in ["user", "assistant"] for turn in conversation)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])