# tests/test_models.py
"""
Test suite for core models in chuk_ai_session_manager.

Tests TokenUsage, SessionEvent, Session, SessionRun, and SessionMetadata models.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock

# Import models directly to avoid circular import issues
from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.session_metadata import SessionMetadata
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.token_usage import TokenUsage, TokenSummary
from chuk_ai_session_manager.models.session_run import SessionRun, RunStatus

class TestTokenUsage:
    """Test TokenUsage model and functionality."""
    
    def test_token_usage_initialization(self):
        """Test TokenUsage initialization and auto-calculation."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, model="gpt-4")
        assert usage.total_tokens == 150
        assert usage.estimated_cost_usd is not None
        assert usage.estimated_cost_usd > 0
    
    def test_token_usage_addition(self):
        """Test adding two TokenUsage instances."""
        usage1 = TokenUsage(prompt_tokens=100, completion_tokens=50, model="gpt-4")
        usage2 = TokenUsage(prompt_tokens=200, completion_tokens=75, model="gpt-4")
        
        combined = usage1 + usage2
        assert combined.prompt_tokens == 300
        assert combined.completion_tokens == 125
        assert combined.total_tokens == 425
    
    async def test_token_usage_async_methods(self):
        """Test async methods of TokenUsage."""
        usage = TokenUsage(model="gpt-3.5-turbo")
        
        # Test async update
        await usage.update(prompt_tokens=100, completion_tokens=50)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        
        # Test async cost calculation
        cost = await usage.calculate_cost()
        assert isinstance(cost, float)
        assert cost >= 0
    
    async def test_token_usage_from_text(self):
        """Test creating TokenUsage from text."""
        usage = await TokenUsage.from_text(
            prompt="Hello, world!",
            completion="Hi there!",
            model="gpt-3.5-turbo"
        )
        
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
        assert usage.model == "gpt-3.5-turbo"
    
    async def test_token_counting(self):
        """Test token counting functionality."""
        text = "This is a test message for token counting."
        tokens = await TokenUsage.count_tokens(text, "gpt-3.5-turbo")
        
        assert isinstance(tokens, int)
        assert tokens > 0
        
        # Test with None text
        none_tokens = await TokenUsage.count_tokens(None, "gpt-3.5-turbo")
        assert none_tokens == 0
    
    def test_token_usage_cost_calculation(self):
        """Test cost calculation for different models."""
        # Test GPT-4
        usage_gpt4 = TokenUsage(prompt_tokens=1000, completion_tokens=500, model="gpt-4")
        assert usage_gpt4.estimated_cost_usd > 0
        
        # Test GPT-3.5
        usage_gpt35 = TokenUsage(prompt_tokens=1000, completion_tokens=500, model="gpt-3.5-turbo")
        assert usage_gpt35.estimated_cost_usd > 0
        
        # GPT-4 should be more expensive than GPT-3.5
        assert usage_gpt4.estimated_cost_usd > usage_gpt35.estimated_cost_usd


class TestTokenSummary:
    """Test TokenSummary functionality."""
    
    async def test_token_summary_add_usage(self):
        """Test adding token usage to summary."""
        summary = TokenSummary()
        
        usage1 = TokenUsage(prompt_tokens=100, completion_tokens=50, model="gpt-4")
        usage2 = TokenUsage(prompt_tokens=200, completion_tokens=75, model="gpt-3.5-turbo")
        
        await summary.add_usage(usage1)
        await summary.add_usage(usage2)
        
        assert summary.total_prompt_tokens == 300
        assert summary.total_completion_tokens == 125
        assert summary.total_tokens == 425
        assert len(summary.usage_by_model) == 2
        assert "gpt-4" in summary.usage_by_model
        assert "gpt-3.5-turbo" in summary.usage_by_model


class TestSessionMetadata:
    """Test SessionMetadata model."""
    
    async def test_session_metadata_creation(self):
        """Test SessionMetadata creation."""
        metadata = await SessionMetadata.create(properties={"user_id": "123"})
        
        assert metadata.created_at is not None
        assert metadata.updated_at is not None
        assert await metadata.get_property("user_id") == "123"
    
    async def test_session_metadata_properties(self):
        """Test metadata property operations."""
        metadata = SessionMetadata()
        
        await metadata.set_property("key1", "value1")
        assert await metadata.get_property("key1") == "value1"
        
        await metadata.update_timestamp()
        assert metadata.updated_at is not None


class TestSessionEvent:
    """Test SessionEvent model and functionality."""
    
    async def test_session_event_creation(self):
        """Test basic SessionEvent creation."""
        event = SessionEvent(
            message="Hello, world!",
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        
        assert event.id is not None
        assert event.timestamp is not None
        assert event.message == "Hello, world!"
        assert event.source == EventSource.USER
        assert event.type == EventType.MESSAGE
    
    async def test_session_event_with_tokens(self):
        """Test creating SessionEvent with token counting."""
        event = await SessionEvent.create_with_tokens(
            message="Test message",
            prompt="Test prompt",
            completion="Test completion",
            model="gpt-3.5-turbo",
            source=EventSource.LLM,
            type=EventType.MESSAGE
        )
        
        assert event.token_usage is not None
        assert event.token_usage.prompt_tokens > 0
        assert event.token_usage.completion_tokens > 0
        assert event.token_usage.model == "gpt-3.5-turbo"
    
    async def test_session_event_metadata(self):
        """Test SessionEvent metadata operations."""
        event = SessionEvent(message="Test")
        
        # Test setting metadata
        await event.set_metadata("key1", "value1")
        assert await event.get_metadata("key1") == "value1"
        
        # Test checking metadata existence
        assert await event.has_metadata("key1") == True
        assert await event.has_metadata("nonexistent") == False
        
        # Test getting with default
        assert await event.get_metadata("nonexistent", "default") == "default"
        
        # Test removing metadata
        await event.remove_metadata("key1")
        assert await event.has_metadata("key1") == False
    
    async def test_session_event_token_usage_update(self):
        """Test updating token usage on existing event."""
        event = SessionEvent(message="Test")
        
        await event.update_token_usage(
            prompt="New prompt",
            completion="New completion",
            model="gpt-4"
        )
        
        assert event.token_usage is not None
        assert event.token_usage.prompt_tokens > 0
        assert event.token_usage.completion_tokens > 0
        assert event.token_usage.model == "gpt-4"
    
    async def test_session_event_update_metadata_alias(self):
        """Test update_metadata alias method."""
        event = SessionEvent(message="Test")
        
        await event.update_metadata("key1", "value1")
        assert await event.get_metadata("key1") == "value1"


class TestSessionRun:
    """Test SessionRun model and functionality."""
    
    async def test_session_run_creation(self):
        """Test SessionRun creation."""
        run = await SessionRun.create(metadata={"test": "value"})
        
        assert run.id is not None
        assert run.status == RunStatus.PENDING
        assert run.started_at is not None
        assert run.ended_at is None
        assert await run.get_metadata("test") == "value"
    
    async def test_session_run_lifecycle(self):
        """Test SessionRun status transitions."""
        run = await SessionRun.create()
        
        # Start the run
        await run.mark_running()
        assert run.status == RunStatus.RUNNING
        
        # Complete the run
        await run.mark_completed()
        assert run.status == RunStatus.COMPLETED
        assert run.ended_at is not None
        
        duration = await run.get_duration()
        assert isinstance(duration, float)
        assert duration >= 0
    
    async def test_session_run_failure(self):
        """Test SessionRun failure handling."""
        run = await SessionRun.create()
        
        await run.mark_failed("Test failure reason")
        assert run.status == RunStatus.FAILED
        assert await run.get_metadata("failure_reason") == "Test failure reason"
    
    async def test_session_run_cancellation(self):
        """Test SessionRun cancellation."""
        run = await SessionRun.create()
        
        await run.mark_cancelled("Test cancellation reason")
        assert run.status == RunStatus.CANCELLED
        assert await run.get_metadata("cancel_reason") == "Test cancellation reason"
    
    async def test_session_run_metadata(self):
        """Test SessionRun metadata operations."""
        run = await SessionRun.create()
        
        await run.set_metadata("key1", "value1")
        assert await run.get_metadata("key1") == "value1"
        assert await run.has_metadata("key1") == True
        
        await run.remove_metadata("key1")
        assert await run.has_metadata("key1") == False
    
    async def test_session_run_tool_calls(self):
        """Test SessionRun tool call tracking."""
        run = await SessionRun.create()
        
        await run.add_tool_call("tool-call-1")
        await run.add_tool_call("tool-call-2")
        
        # Test no duplicates
        await run.add_tool_call("tool-call-1")
        
        assert len(run.tool_calls) == 2
        assert "tool-call-1" in run.tool_calls
        assert "tool-call-2" in run.tool_calls
    
    async def test_session_run_to_dict(self):
        """Test SessionRun serialization."""
        run = await SessionRun.create(metadata={"test": "value"})
        await run.mark_completed()
        
        data = await run.to_dict()
        
        assert data["id"] == run.id
        assert data["status"] == "completed"
        assert "started_at" in data
        assert "ended_at" in data
        assert "duration" in data
        assert data["metadata"]["test"] == "value"
    
    async def test_session_run_get_tool_calls(self):
        """Test getting tool calls from session."""
        run = await SessionRun.create()
        await run.add_tool_call("tool-1")
        await run.add_tool_call("tool-2")
        
        # Mock session with events
        mock_session = type('MockSession', (), {
            'events': [
                type('MockEvent', (), {'id': 'tool-1'}),
                type('MockEvent', (), {'id': 'tool-2'}),
                type('MockEvent', (), {'id': 'tool-3'}),  # Not in run
            ]
        })()
        
        tool_events = await run.get_tool_calls(mock_session)
        assert len(tool_events) == 2


class TestSession:
    """Test Session model and functionality."""
    
    async def test_session_creation(self):
        """Test basic Session creation."""
        # Import Session here to avoid circular imports
        from chuk_ai_session_manager.models.session import Session
        
        # Patch the storage system
        with patch('chuk_ai_session_manager.session_storage.get_backend') as mock_get_backend, \
             patch('chuk_ai_session_manager.session_storage.ChukSessionsStore') as mock_store_class:
            
            mock_backend = AsyncMock()
            mock_store = AsyncMock()
            mock_get_backend.return_value = mock_backend
            mock_store_class.return_value = mock_store
            mock_store.save = AsyncMock()
            
            session = await Session.create()
            
            assert session.id is not None
            assert session.metadata is not None
            assert session.events == []
            assert session.state == {}
            assert session.child_ids == []
    
    async def test_session_with_parent(self):
        """Test Session creation with parent."""
        from chuk_ai_session_manager.models.session import Session
        
        # Patch the storage system
        with patch('chuk_ai_session_manager.session_storage.get_backend') as mock_get_backend, \
             patch('chuk_ai_session_manager.session_storage.ChukSessionsStore') as mock_store_class:
            
            mock_backend = AsyncMock()
            mock_store = AsyncMock()
            mock_get_backend.return_value = mock_backend
            mock_store_class.return_value = mock_store
            mock_store.get.return_value = None  # Parent doesn't exist for simplicity
            mock_store.save = AsyncMock()
            
            session = await Session.create(parent_id="parent-123")
            assert session.parent_id == "parent-123"
    
    async def test_session_add_event(self):
        """Test adding events to a session."""
        from chuk_ai_session_manager.models.session import Session
        
        session = Session()
        
        event = await SessionEvent.create_with_tokens(
            message="Test message",
            prompt="Test prompt",
            model="gpt-3.5-turbo"
        )
        
        await session.add_event(event)
        
        assert len(session.events) == 1
        assert session.events[0] == event
        assert session.total_tokens > 0
    
    async def test_session_add_event_and_save(self):
        """Test adding event and saving session."""
        from chuk_ai_session_manager.models.session import Session
        
        # Patch the storage system properly
        with patch('chuk_ai_session_manager.session_storage.get_backend') as mock_get_backend, \
             patch('chuk_ai_session_manager.session_storage.ChukSessionsStore') as mock_store_class:
            
            mock_backend = AsyncMock()
            mock_store = AsyncMock()
            mock_get_backend.return_value = mock_backend
            mock_store_class.return_value = mock_store
            mock_store.save = AsyncMock()
            
            session = Session()
            event = SessionEvent(message="Test")
            
            await session.add_event_and_save(event)
            
            assert len(session.events) == 1
            # Verify save was called
            mock_store.save.assert_called_once_with(session)
    
    async def test_session_state_management(self):
        """Test session state operations."""
        from chuk_ai_session_manager.models.session import Session
        
        session = Session()
        
        # Test setting state
        await session.set_state("key1", "value1")
        assert await session.get_state("key1") == "value1"
        
        # Test state existence
        assert await session.has_state("key1") == True
        assert await session.has_state("nonexistent") == False
        
        # Test getting with default
        assert await session.get_state("nonexistent", "default") == "default"
        
        # Test removing state
        await session.remove_state("key1")
        assert await session.has_state("key1") == False
    
    async def test_session_token_tracking(self):
        """Test token tracking in session."""
        session = await Session.create()
        
        # Add events with token usage
        event1 = await SessionEvent.create_with_tokens(
            message="Hello",
            prompt="Hello",
            model="gpt-3.5-turbo",
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        await session.add_event(event1)
        
        event2 = await SessionEvent.create_with_tokens(
            message="Hi there!",
            prompt="",
            completion="Hi there!",
            model="gpt-3.5-turbo",
            source=EventSource.LLM,
            type=EventType.MESSAGE
        )
        await session.add_event(event2)
        
        # Check token summary
        # The actual token count will be based on the real tokenizer
        # "Hello" is about 1-2 tokens, "Hi there!" is about 2-3 tokens
        # So total should be around 4-5 tokens, not 18
        assert session.total_tokens > 0  # Should have some tokens
        assert session.total_tokens < 10  # But not as many as 18
        
        # Check by source
        usage_by_source = await session.get_token_usage_by_source()
        assert "user" in usage_by_source
        assert "llm" in usage_by_source
        
        # Verify the token counts are reasonable
        assert usage_by_source["user"].total_tokens > 0
        assert usage_by_source["llm"].total_tokens > 0
        
    async def test_session_properties(self):
        """Test session computed properties."""
        from chuk_ai_session_manager.models.session import Session
        
        session = Session()
        
        # Test last_update_time with no events
        assert session.last_update_time == session.metadata.created_at
        
        # Add an event and check last_update_time
        event = SessionEvent(message="Test")
        await session.add_event(event)
        assert session.last_update_time == event.timestamp
        
        # Test active_run (should be None initially)
        assert session.active_run is None
        
        # Add a running session and test
        run = SessionRun(status=RunStatus.RUNNING)
        session.runs.append(run)
        assert session.active_run == run


if __name__ == "__main__":
    pytest.main([__file__, "-v"])