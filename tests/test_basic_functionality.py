# tests/test_basic_functionality.py
"""
Basic functionality tests that work around import issues.

This file tests core functionality without relying on complex imports
that might cause circular dependency issues.
"""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

# Import only the models directly to avoid circular imports
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.token_usage import TokenUsage
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.session_metadata import SessionMetadata


class TestBasicModels:
    """Test basic model functionality."""
    
    def test_event_source_enum(self):
        """Test EventSource enum values."""
        assert EventSource.USER == "user"
        assert EventSource.LLM == "llm"
        assert EventSource.SYSTEM == "system"
    
    def test_event_type_enum(self):
        """Test EventType enum values."""
        assert EventType.MESSAGE == "message"
        assert EventType.SUMMARY == "summary"
        assert EventType.TOOL_CALL == "tool_call"
        assert EventType.REFERENCE == "reference"
        assert EventType.CONTEXT_BRIDGE == "context_bridge"
    
    def test_token_usage_basic(self):
        """Test basic TokenUsage functionality."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            model="test-model"
        )
        
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.model == "test-model"
        assert usage.estimated_cost_usd is not None
    
    async def test_session_event_basic(self):
        """Test basic SessionEvent functionality."""
        event = SessionEvent(
            message="Test message",
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        
        assert event.message == "Test message"
        assert event.source == EventSource.USER
        assert event.type == EventType.MESSAGE
        assert event.id is not None
        assert event.timestamp is not None
    
    async def test_session_event_metadata(self):
        """Test SessionEvent metadata operations."""
        event = SessionEvent(message="Test")
        
        await event.set_metadata("key1", "value1")
        assert await event.get_metadata("key1") == "value1"
        
        assert await event.has_metadata("key1") == True
        assert await event.has_metadata("missing") == False
        
        await event.remove_metadata("key1")
        assert await event.has_metadata("key1") == False
    
    async def test_session_metadata_basic(self):
        """Test basic SessionMetadata functionality."""
        metadata = SessionMetadata()
        
        assert metadata.created_at is not None
        assert metadata.updated_at is not None
        assert isinstance(metadata.properties, dict)
        
        await metadata.set_property("test_key", "test_value")
        assert await metadata.get_property("test_key") == "test_value"


class TestTokenOperations:
    """Test token-related operations."""
    
    async def test_token_counting_with_text(self):
        """Test token counting functionality."""
        # Test basic counting
        tokens = await TokenUsage.count_tokens("Hello world", "gpt-3.5-turbo")
        assert isinstance(tokens, int)
        assert tokens > 0
        
        # Test with None
        none_tokens = await TokenUsage.count_tokens(None, "gpt-3.5-turbo")
        assert none_tokens == 0
        
        # Test with empty string
        empty_tokens = await TokenUsage.count_tokens("", "gpt-3.5-turbo")
        assert empty_tokens == 0
    
    async def test_token_usage_from_text(self):
        """Test creating TokenUsage from text."""
        usage = await TokenUsage.from_text(
            prompt="Test prompt",
            completion="Test completion",
            model="gpt-3.5-turbo"
        )
        
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0
        assert usage.model == "gpt-3.5-turbo"
        assert usage.estimated_cost_usd >= 0
    
    def test_token_usage_cost_calculation(self):
        """Test token usage cost calculation."""
        usage = TokenUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            model="gpt-4"
        )
        
        assert usage.estimated_cost_usd > 0
        
        # Test different model
        usage_cheap = TokenUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            model="gpt-3.5-turbo"
        )
        
        # GPT-4 should be more expensive
        assert usage.estimated_cost_usd > usage_cheap.estimated_cost_usd
    
    def test_token_usage_addition(self):
        """Test adding TokenUsage instances."""
        usage1 = TokenUsage(prompt_tokens=100, completion_tokens=50, model="gpt-4")
        usage2 = TokenUsage(prompt_tokens=200, completion_tokens=100, model="gpt-4")
        
        combined = usage1 + usage2
        assert combined.prompt_tokens == 300
        assert combined.completion_tokens == 150
        assert combined.total_tokens == 450


class TestEventCreation:
    """Test event creation and token tracking."""
    
    async def test_create_event_with_tokens(self):
        """Test creating events with automatic token counting."""
        event = await SessionEvent.create_with_tokens(
            message="Test message",
            prompt="Test prompt",
            completion="Test completion",
            model="gpt-3.5-turbo",
            source=EventSource.LLM,
            type=EventType.MESSAGE
        )
        
        assert event.message == "Test message"
        assert event.source == EventSource.LLM
        assert event.type == EventType.MESSAGE
        assert event.token_usage is not None
        assert event.token_usage.prompt_tokens > 0
        assert event.token_usage.completion_tokens > 0
        assert event.token_usage.model == "gpt-3.5-turbo"
    
    async def test_update_token_usage(self):
        """Test updating token usage on existing event."""
        event = SessionEvent(message="Test")
        
        # Initially no token usage
        assert event.token_usage is None
        
        # Update with token info
        await event.update_token_usage(
            prompt="New prompt",
            completion="New completion",
            model="gpt-4"
        )
        
        assert event.token_usage is not None
        assert event.token_usage.prompt_tokens > 0
        assert event.token_usage.completion_tokens > 0
        assert event.token_usage.model == "gpt-4"


class TestSessionManagementBasics:
    """Test basic session management without complex imports."""
    
    def test_session_creation_isolated(self):
        """Test session creation in isolation."""
        # Import Session only when needed to avoid circular imports
        with patch.dict('sys.modules', {'chuk_sessions': MagicMock()}):
            from chuk_ai_session_manager.models.session import Session
            
            session = Session()
            assert session.id is not None
            assert session.metadata is not None
            assert session.events == []
            assert session.state == {}
            assert session.child_ids == []
    
    async def test_session_event_management(self):
        """Test session event management."""
        with patch.dict('sys.modules', {'chuk_sessions': MagicMock()}):
            from chuk_ai_session_manager.models.session import Session
            
            session = Session()
            
            # Create an event
            event = await SessionEvent.create_with_tokens(
                message="Test message",
                prompt="Test prompt",
                model="gpt-3.5-turbo"
            )
            
            # Add event to session
            await session.add_event(event)
            
            assert len(session.events) == 1
            assert session.events[0] == event
            assert session.total_tokens > 0
    
    async def test_session_state_management(self):
        """Test session state management."""
        with patch.dict('sys.modules', {'chuk_sessions': MagicMock()}):
            from chuk_ai_session_manager.models.session import Session
            
            session = Session()
            
            # Test state operations
            await session.set_state("key1", "value1")
            assert await session.get_state("key1") == "value1"
            
            assert await session.has_state("key1") == True
            assert await session.has_state("missing") == False
            
            await session.remove_state("key1")
            assert await session.has_state("key1") == False


class TestBasicPromptBuilding:
    """Test basic prompt building without complex dependencies."""
    
    async def test_minimal_prompt_logic(self):
        """Test minimal prompt building logic."""
        # Create a simple session structure
        with patch.dict('sys.modules', {'chuk_sessions': MagicMock()}):
            from chuk_ai_session_manager.models.session import Session
            
            session = Session()
            
            # Add user message
            user_event = await SessionEvent.create_with_tokens(
                message="Hello, how are you?",
                prompt="Hello, how are you?",
                source=EventSource.USER,
                type=EventType.MESSAGE
            )
            await session.add_event(user_event)
            
            # Add assistant message
            assistant_event = await SessionEvent.create_with_tokens(
                message="I'm doing well, thank you!",
                prompt="",
                completion="I'm doing well, thank you!",
                source=EventSource.LLM,
                type=EventType.MESSAGE
            )
            await session.add_event(assistant_event)
            
            # Verify basic structure
            assert len(session.events) == 2
            assert session.events[0].source == EventSource.USER
            assert session.events[1].source == EventSource.LLM
            
            # Test basic prompt building logic manually
            user_messages = [e for e in session.events if e.source == EventSource.USER]
            assistant_messages = [e for e in session.events if e.source == EventSource.LLM]
            
            assert len(user_messages) == 1
            assert len(assistant_messages) == 1
            assert user_messages[0].message == "Hello, how are you?"
            assert assistant_messages[0].message == "I'm doing well, thank you!"


class TestErrorHandling:
    """Test basic error handling."""
    
    async def test_token_counting_errors(self):
        """Test token counting with various inputs."""
        # Test with None
        tokens = await TokenUsage.count_tokens(None)
        assert tokens == 0
        
        # Test with empty string
        tokens = await TokenUsage.count_tokens("")
        assert tokens == 0
        
        # Test with non-string input
        tokens = await TokenUsage.count_tokens(123)
        assert tokens >= 0  # Should convert to string and count
    
    def test_token_usage_invalid_model(self):
        """Test TokenUsage with unknown model."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            model="unknown-model-xyz"
        )
        
        # Should still work, just use default pricing
        assert usage.estimated_cost_usd >= 0
        assert usage.total_tokens == 150
    
    async def test_event_metadata_edge_cases(self):
        """Test event metadata with edge cases."""
        event = SessionEvent(message="Test")
        
        # Test setting None value
        await event.set_metadata("none_key", None)
        assert await event.get_metadata("none_key") is None
        
        # Test getting non-existent key with default
        assert await event.get_metadata("missing", "default") == "default"
        
        # Test removing non-existent key (should not error)
        await event.remove_metadata("non_existent")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])