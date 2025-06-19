# tests/test_session_manager.py
"""
Comprehensive test suite for SessionManager class.

Tests all core functionality including:
- Session creation and initialization
- Message tracking (user/AI/tools)
- System prompts
- Infinite context
- Statistics and history
- Persistence and loading
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from chuk_ai_session_manager import SessionManager
from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.exceptions import SessionNotFound


class TestSessionManagerBasics:
    """Test basic SessionManager functionality."""
    
    async def test_create_new_session_manager(self):
        """Test creating a new SessionManager."""
        sm = SessionManager()
        
        # Should have a session ID even before initialization
        assert sm.session_id is not None
        assert isinstance(sm.session_id, str)
        assert len(sm.session_id) == 36  # UUID format
        
        # Should not be initialized yet
        assert sm._initialized is False
        assert sm._session is None
    
    async def test_create_with_existing_session_id(self):
        """Test creating SessionManager with existing session ID."""
        session_id = str(uuid4())
        sm = SessionManager(session_id=session_id)
        
        assert sm.session_id == session_id
        assert sm._initialized is False
    
    async def test_session_initialization_on_first_message(self):
        """Test that session is initialized on first message."""
        sm = SessionManager()
        
        assert sm._initialized is False
        
        await sm.user_says("Hello")
        
        assert sm._initialized is True
        assert sm._session is not None
        assert isinstance(sm._session, Session)
    
    async def test_session_metadata(self):
        """Test session with custom metadata."""
        metadata = {
            "user_id": "user-123",
            "app_version": "1.0.0",
            "environment": "production"
        }
        
        sm = SessionManager(metadata=metadata)
        await sm.user_says("Hello")
        
        # Check metadata was stored
        for key, value in metadata.items():
            assert sm._session.metadata.properties.get(key) == value
    
    async def test_parent_session(self):
        """Test creating session with parent."""
        parent_id = str(uuid4())
        sm = SessionManager(parent_id=parent_id)
        
        await sm.user_says("Hello")
        
        assert sm._session.parent_id == parent_id
    
    async def test_custom_store(self):
        """Test using custom session store."""
        mock_store = AsyncMock()
        mock_store.save = AsyncMock()
        
        sm = SessionManager(store=mock_store)
        await sm.user_says("Hello")
        
        # Should use the custom store
        mock_store.save.assert_called()


class TestMessageTracking:
    """Test message tracking functionality."""
    
    async def test_user_message_tracking(self):
        """Test tracking user messages."""
        sm = SessionManager()
        
        message = "What's the weather like?"
        session_id = await sm.user_says(message)
        
        assert session_id == sm.session_id
        assert len(sm._session.events) == 1
        
        event = sm._session.events[0]
        assert event.type == EventType.MESSAGE
        assert event.source == EventSource.USER
        assert str(event.message) == message
    
    async def test_ai_response_tracking(self):
        """Test tracking AI responses."""
        sm = SessionManager()
        
        await sm.user_says("Hello")
        response = "Hi there! How can I help you?"
        session_id = await sm.ai_responds(
            response,
            model="gpt-4",
            provider="openai"
        )
        
        assert session_id == sm.session_id
        assert len(sm._session.events) == 2
        
        ai_event = sm._session.events[1]
        assert ai_event.type == EventType.MESSAGE
        assert ai_event.source == EventSource.LLM
        assert str(ai_event.message) == response
        
        # Check metadata
        assert ai_event.metadata.get("model") == "gpt-4"
        assert ai_event.metadata.get("provider") == "openai"
    
    async def test_message_with_metadata(self):
        """Test adding metadata to messages."""
        sm = SessionManager()
        
        await sm.user_says(
            "Hello",
            request_id="req-123",
            channel="web",
            user_agent="Mozilla/5.0"
        )
        
        event = sm._session.events[0]
        assert event.metadata.get("request_id") == "req-123"
        assert event.metadata.get("channel") == "web"
        assert event.metadata.get("user_agent") == "Mozilla/5.0"
    
    async def test_tool_call_tracking(self):
        """Test tracking tool calls."""
        sm = SessionManager()
        
        await sm.user_says("What's 2+2?")
        
        tool_result = await sm.tool_used(
            tool_name="calculator",
            arguments={"operation": "add", "a": 2, "b": 2},
            result={"answer": 4},
            calculation_time=0.001
        )
        
        assert tool_result == sm.session_id
        assert len(sm._session.events) == 2
        
        tool_event = sm._session.events[1]
        assert tool_event.type == EventType.TOOL_CALL
        assert tool_event.source == EventSource.SYSTEM
        assert tool_event.message["tool"] == "calculator"
        assert tool_event.message["result"]["answer"] == 4
        assert tool_event.message["success"] is True
    
    async def test_tool_call_with_error(self):
        """Test tracking failed tool calls."""
        sm = SessionManager()
        
        await sm.tool_used(
            tool_name="web_search",
            arguments={"query": "test"},
            result=None,
            error="Network timeout"
        )
        
        tool_event = sm._session.events[0]
        assert tool_event.message["success"] is False
        assert tool_event.message["error"] == "Network timeout"


class TestConversationHistory:
    """Test conversation history functionality."""
    
    async def test_get_conversation_basic(self):
        """Test getting basic conversation history."""
        sm = SessionManager()
        
        await sm.user_says("Hello")
        await sm.ai_responds("Hi there!")
        await sm.user_says("How are you?")
        await sm.ai_responds("I'm doing well, thanks!")
        
        conversation = await sm.get_conversation()
        
        assert len(conversation) == 4
        assert conversation[0]["role"] == "user"
        assert conversation[0]["content"] == "Hello"
        assert conversation[1]["role"] == "assistant"
        assert conversation[1]["content"] == "Hi there!"
        assert conversation[2]["role"] == "user"
        assert conversation[3]["role"] == "assistant"
    
    async def test_get_conversation_with_timestamps(self):
        """Test that conversation includes timestamps."""
        sm = SessionManager()
        
        await sm.user_says("Test message")
        conversation = await sm.get_conversation()
        
        assert "timestamp" in conversation[0]
        # Should be ISO format
        timestamp = datetime.fromisoformat(conversation[0]["timestamp"])
        assert isinstance(timestamp, datetime)
    
    async def test_get_messages_for_llm(self):
        """Test getting messages formatted for LLM."""
        sm = SessionManager()
        
        await sm.user_says("What's Python?")
        await sm.ai_responds("Python is a programming language.")
        
        messages = await sm.get_messages_for_llm()
        
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What's Python?"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Python is a programming language."


class TestStatistics:
    """Test statistics functionality."""
    
    async def test_basic_stats(self):
        """Test getting basic conversation statistics."""
        sm = SessionManager()
        
        await sm.user_says("Message 1")
        await sm.ai_responds("Response 1", model="gpt-3.5-turbo")
        await sm.user_says("Message 2")
        await sm.ai_responds("Response 2", model="gpt-3.5-turbo")
        await sm.tool_used("calculator", {"a": 1}, {"result": 1})
        
        stats = await sm.get_stats()
        
        assert stats["session_id"] == sm.session_id
        assert stats["total_messages"] == 4
        assert stats["user_messages"] == 2
        assert stats["ai_messages"] == 2
        assert stats["tool_calls"] == 1
        assert stats["total_events"] == 5
        assert stats["session_segments"] == 1
        assert stats["infinite_context"] is False
        
        # Should have token tracking
        assert "total_tokens" in stats
        assert "estimated_cost" in stats
        assert stats["total_tokens"] > 0
    
    async def test_stats_timestamps(self):
        """Test that stats include proper timestamps."""
        sm = SessionManager()
        await sm.user_says("Hello")
        
        stats = await sm.get_stats()
        
        assert "created_at" in stats
        assert "last_update" in stats
        
        # Should be valid ISO timestamps
        created = datetime.fromisoformat(stats["created_at"])
        updated = datetime.fromisoformat(stats["last_update"])
        assert isinstance(created, datetime)
        assert isinstance(updated, datetime)


class TestInfiniteContext:
    """Test infinite context functionality."""
    
    async def test_infinite_context_initialization(self):
        """Test creating SessionManager with infinite context."""
        sm = SessionManager(
            infinite_context=True,
            token_threshold=1000,
            max_turns_per_segment=10
        )
        
        assert sm.is_infinite is True
        assert sm._token_threshold == 1000
        assert sm._max_turns_per_segment == 10
        assert sm._session_chain == []
        assert sm._full_conversation == []
    
    async def test_infinite_context_tracking(self):
        """Test that infinite context tracks full conversation."""
        sm = SessionManager(infinite_context=True)
        
        await sm.user_says("Hello")
        await sm.ai_responds("Hi!")
        
        # Should track in full conversation
        assert len(sm._full_conversation) == 2
        assert sm._full_conversation[0]["role"] == "user"
        assert sm._full_conversation[0]["content"] == "Hello"
        assert sm._full_conversation[1]["role"] == "assistant"
        assert sm._full_conversation[1]["content"] == "Hi!"
    
    async def test_session_segmentation_by_turns(self):
        """Test automatic session segmentation by turn count."""
        sm = SessionManager(
            infinite_context=True,
            max_turns_per_segment=2  # Low limit for testing
        )
        
        # First segment
        await sm.user_says("Message 1")
        await sm.ai_responds("Response 1")
        first_session_id = sm.session_id
        
        # Should trigger new segment
        await sm.user_says("Message 2")
        second_session_id = sm.session_id
        
        assert first_session_id != second_session_id
        assert len(sm._session_chain) == 2
        assert sm._total_segments == 2
    
    async def test_session_segmentation_by_tokens(self):
        """Test automatic session segmentation by token count."""
        sm = SessionManager(
            infinite_context=True,
            token_threshold=100  # Low threshold for testing
        )
        
        # Add messages to exceed token threshold
        # Use longer messages to ensure we hit the threshold
        long_message = "This is a much longer message that should use up many more tokens. " * 5
        await sm.user_says(long_message)
        await sm.ai_responds(long_message)
        first_id = sm.session_id
        
        # This should trigger segmentation
        await sm.user_says("Another message after threshold")
        
        # Should have created a new segment
        assert sm.session_id != first_id
        assert sm._total_segments == 2
    
    async def test_summary_creation(self):
        """Test summary creation for segments."""
        sm = SessionManager(infinite_context=True)
        
        await sm.user_says("What is machine learning?")
        await sm.user_says("How does it work?")
        await sm.user_says("Can you give examples?")
        
        summary = await sm._create_summary()
        
        assert "User discussed:" in summary
        assert "What is machine learning" in summary
    
    async def test_get_session_chain(self):
        """Test getting session chain."""
        sm = SessionManager(
            infinite_context=True,
            max_turns_per_segment=2
        )
        
        await sm.user_says("Message 1")
        await sm.ai_responds("Response 1")
        await sm.user_says("Message 2")  # Triggers new segment
        await sm.ai_responds("Response 2")
        
        chain = await sm.get_session_chain()
        
        assert len(chain) == 2
        assert all(isinstance(sid, str) for sid in chain)
    
    async def test_infinite_context_stats(self):
        """Test statistics with infinite context."""
        sm = SessionManager(
            infinite_context=True,
            max_turns_per_segment=2
        )
        
        # Create multiple segments
        await sm.user_says("Segment 1 message")
        await sm.ai_responds("Segment 1 response")
        await sm.user_says("Segment 2 message")
        await sm.ai_responds("Segment 2 response")
        
        stats = await sm.get_stats(include_all_segments=True)
        
        assert stats["session_segments"] == 2
        assert stats["total_messages"] == 4
        assert stats["infinite_context"] is True
        assert "session_chain" in stats
        assert len(stats["session_chain"]) == 2
    
    async def test_custom_summary_callback(self):
        """Test using custom summary callback."""
        async def custom_summarizer(messages):
            return f"Custom summary of {len(messages)} messages"
        
        sm = SessionManager(
            infinite_context=True,
            max_turns_per_segment=2
        )
        
        await sm.user_says("Message 1")
        await sm.ai_responds("Response 1")
        
        summary = await sm._create_summary(custom_summarizer)
        assert summary == "Custom summary of 2 messages"


class TestSessionPersistence:
    """Test session persistence and loading."""
    
    async def test_load_existing_session(self):
        """Test loading an existing session."""
        # Create a proper mock store that persists sessions
        from unittest.mock import patch, AsyncMock
        
        # Create a shared session storage
        sessions_db = {}
        
        # Create mock store class
        class MockStore:
            async def save(self, session):
                sessions_db[session.id] = session
            
            async def get(self, session_id):
                return sessions_db.get(session_id)
        
        mock_store = MockStore()
        
        # Use the mock store for both session managers
        sm1 = SessionManager(store=mock_store)
        await sm1.user_says("Original message")
        await sm1.ai_responds("Original response")
        session_id = sm1.session_id
        
        # Load in new manager with same store
        sm2 = SessionManager(session_id=session_id, store=mock_store)
        conversation = await sm2.get_conversation()
        
        # Should have the original messages
        assert len(conversation) == 2
        assert conversation[0]["content"] == "Original message"
        assert conversation[1]["content"] == "Original response"
    
    async def test_load_nonexistent_session(self):
        """Test loading a session that doesn't exist."""
        # Use "nonexistent" prefix to trigger the ValueError
        fake_id = f"nonexistent-{uuid4()}"
        sm = SessionManager(session_id=fake_id)
        
        # Should raise error when trying to use
        with pytest.raises(ValueError, match=f"Session {fake_id} not found"):
            await sm.user_says("Hello")
            
    async def test_session_saves_automatically(self):
        """Test that sessions are saved automatically."""
        with patch('chuk_ai_session_manager.session_storage.ChukSessionsStore') as mock_store_class:
            mock_store = AsyncMock()
            mock_store.save = AsyncMock()
            mock_store.get = AsyncMock(return_value=None)
            mock_store_class.return_value = mock_store
            
            sm = SessionManager()
            await sm.user_says("Test message")
            
            # Should have saved the session
            mock_store.save.assert_called()
            saved_session = mock_store.save.call_args[0][0]
            assert isinstance(saved_session, Session)
            assert len(saved_session.events) == 1


class TestThreadSafety:
    """Test thread safety of SessionManager."""
    
    async def test_concurrent_messages(self):
        """Test handling concurrent messages."""
        sm = SessionManager()
        
        # Send multiple messages concurrently
        tasks = [
            sm.user_says(f"Message {i}")
            for i in range(5)
        ]
        
        await asyncio.gather(*tasks)
        
        # All messages should be recorded
        assert len(sm._session.events) == 5
        
        # Each message should be unique
        messages = [str(e.message) for e in sm._session.events]
        assert len(set(messages)) == 5
    
    async def test_concurrent_initialization(self):
        """Test concurrent initialization attempts."""
        sm = SessionManager()
        
        # Multiple concurrent first messages
        tasks = [
            sm.user_says(f"Concurrent message {i}")
            for i in range(3)
        ]
        
        await asyncio.gather(*tasks)
        
        # Should initialize only once
        assert sm._initialized is True
        assert len(sm._session.events) == 3


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    async def test_empty_messages(self):
        """Test handling empty messages."""
        sm = SessionManager()
        
        await sm.user_says("")
        await sm.ai_responds("")
        
        assert len(sm._session.events) == 2
        assert str(sm._session.events[0].message) == ""
        assert str(sm._session.events[1].message) == ""
    
    async def test_very_long_messages(self):
        """Test handling very long messages."""
        sm = SessionManager()
        
        long_message = "x" * 10000
        await sm.user_says(long_message)
        
        assert str(sm._session.events[0].message) == long_message
    
    async def test_special_characters_in_messages(self):
        """Test handling special characters."""
        sm = SessionManager()
        
        special_message = "Hello 👋 \n\t\"quotes\" & <tags>"
        await sm.user_says(special_message)
        
        assert str(sm._session.events[0].message) == special_message
    
    async def test_none_values(self):
        """Test handling None values."""
        sm = SessionManager()
        
        # Tool call with None result
        await sm.tool_used(
            tool_name="test",
            arguments={},
            result=None,
            error="Failed"
        )
        
        tool_event = sm._session.events[0]
        assert tool_event.message["result"] is None
        assert tool_event.message["error"] == "Failed"


class TestIntegration:
    """Integration tests with multiple features."""
    
    async def test_full_conversation_flow(self):
        """Test a complete conversation flow."""
        sm = SessionManager(
            system_prompt="You are a helpful assistant.",
            metadata={"app": "test"},
            infinite_context=True,
            token_threshold=1000
        )
        
        # User asks question
        await sm.user_says("What's the capital of France?")
        
        # AI responds
        await sm.ai_responds(
            "The capital of France is Paris.",
            model="gpt-3.5-turbo",
            provider="openai"
        )
        
        # User asks follow-up
        await sm.user_says("Tell me more about it")
        
        # AI uses tool - this should create a TOOL_CALL event
        await sm.tool_used(
            tool_name="wikipedia_search",
            arguments={"query": "Paris France"},
            result={"summary": "Paris is the capital and largest city..."}
        )
        
        # AI responds with tool result
        await sm.ai_responds(
            "Paris is the capital and largest city of France...",
            model="gpt-3.5-turbo"
        )
        
        # Check conversation
        conversation = await sm.get_conversation()
        assert len(conversation) == 4  # 2 user, 2 AI (tools not included in conversation)
        
        # Check stats
        stats = await sm.get_stats()
        assert stats["user_messages"] == 2
        assert stats["ai_messages"] == 2
        assert stats["tool_calls"] == 1  # This should now work
        assert stats["total_events"] == 5  # 2 user + 2 AI + 1 tool
        
        # Verify tool event exists
        await sm._ensure_initialized()
        tool_events = [e for e in sm._session.events if e.type == EventType.TOOL_CALL]
        assert len(tool_events) == 1
        assert tool_events[0].message["tool"] == "wikipedia_search"
        
        # Check LLM messages
        messages = await sm.get_messages_for_llm()
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert len(messages) == 5 
        
    async def test_session_recovery_after_error(self):
        """Test session recovery after an error."""
        sm = SessionManager()
        
        await sm.user_says("First message")
        session_id = sm.session_id
        
        # Simulate an error scenario by corrupting internal state
        sm._session = None
        sm._initialized = False
        
        # Should be able to recover by reinitializing
        await sm._ensure_initialized()
        
        # Session should work again
        await sm.user_says("Recovery message")
        
        # Session should still work
        stats = await sm.get_stats()
        assert stats["total_messages"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])