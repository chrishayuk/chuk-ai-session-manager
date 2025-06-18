# tests/test_simple.py
"""
Simple working test for chuk_ai_session_manager.

This test should work with the fixed imports and focuses on actual functionality
rather than implementation details.
"""

import pytest
import asyncio


class TestBasicImports:
    """Test that basic imports work."""
    
    def test_import_enums(self):
        """Test importing enums."""
        from chuk_ai_session_manager import EventSource, EventType
        
        assert EventSource.USER == "user"
        assert EventSource.LLM == "llm"
        assert EventSource.SYSTEM == "system"
        assert EventType.MESSAGE == "message"
        assert EventType.TOOL_CALL == "tool_call"
        assert EventType.SUMMARY == "summary"
    
    def test_import_session_manager(self):
        """Test importing SessionManager."""
        from chuk_ai_session_manager import SessionManager
        
        # Should be able to create instance
        sm = SessionManager()
        assert sm is not None
    
    def test_import_convenience_functions(self):
        """Test importing convenience functions."""
        from chuk_ai_session_manager import (
            track_conversation,
            track_llm_call,
            quick_conversation,
            track_infinite_conversation
        )
        
        assert track_conversation is not None
        assert callable(track_conversation)
        assert track_llm_call is not None
        assert callable(track_llm_call)
        assert quick_conversation is not None
        assert callable(quick_conversation)
        assert track_infinite_conversation is not None
        assert callable(track_infinite_conversation)


class TestBasicFunctionality:
    """Test basic functionality works."""
    
    async def test_session_manager_creation(self):
        """Test creating a SessionManager."""
        from chuk_ai_session_manager import SessionManager
        
        sm = SessionManager()
        
        # Test basic properties
        assert hasattr(sm, 'session_id')
        session_id = sm.session_id
        assert isinstance(session_id, str)
        assert len(session_id) > 0
        
        # Test that session_id is consistent
        assert sm.session_id == session_id
    
    async def test_session_manager_infinite_context(self):
        """Test SessionManager with infinite context."""
        from chuk_ai_session_manager import SessionManager
        
        sm_infinite = SessionManager(infinite_context=True)
        assert sm_infinite.is_infinite == True
        
        sm_regular = SessionManager(infinite_context=False)
        assert sm_regular.is_infinite == False
        
        # Test default is False
        sm_default = SessionManager()
        assert sm_default.is_infinite == False
    
    async def test_user_says_basic(self):
        """Test basic user_says functionality."""
        from chuk_ai_session_manager import SessionManager
        
        sm = SessionManager()
        
        # This should work without errors
        session_id = await sm.user_says("Hello, world!")
        assert isinstance(session_id, str)
        assert len(session_id) > 0
        
        # Session ID should remain consistent
        assert session_id == sm.session_id
    
    async def test_ai_responds_basic(self):
        """Test basic ai_responds functionality."""
        from chuk_ai_session_manager import SessionManager
        
        sm = SessionManager()
        
        # Add user message first
        user_session_id = await sm.user_says("Hello!")
        
        # Then AI response
        ai_session_id = await sm.ai_responds(
            "Hi there! How can I help?",
            model="gpt-4",
            provider="openai"
        )
        
        assert isinstance(ai_session_id, str)
        assert len(ai_session_id) > 0
        
        # Should be same session for both
        assert user_session_id == ai_session_id
    
    async def test_tool_used_basic(self):
        """Test basic tool_used functionality."""
        from chuk_ai_session_manager import SessionManager
        
        sm = SessionManager()
        
        # Use a tool
        session_id = await sm.tool_used(
            tool_name="calculator",
            arguments={"operation": "add", "a": 2, "b": 3},
            result={"answer": 5}
        )
        
        assert isinstance(session_id, str)
        assert len(session_id) > 0
    
    async def test_tool_used_with_error(self):
        """Test tool_used with error."""
        from chuk_ai_session_manager import SessionManager
        
        sm = SessionManager()
        
        # Use a tool that fails
        session_id = await sm.tool_used(
            tool_name="search",
            arguments={"query": "test"},
            result=None,
            error="Connection timeout"
        )
        
        assert isinstance(session_id, str)
        assert len(session_id) > 0
    
    async def test_get_conversation(self):
        """Test getting conversation history."""
        from chuk_ai_session_manager import SessionManager
        
        sm = SessionManager()
        
        # Add some conversation
        await sm.user_says("Hello!")
        await sm.ai_responds("Hi there!")
        await sm.user_says("How are you?")
        await sm.ai_responds("I'm doing well, thanks!")
        
        # Get conversation
        conversation = await sm.get_conversation()
        
        assert len(conversation) == 4
        assert conversation[0]["role"] == "user"
        assert conversation[0]["content"] == "Hello!"
        assert conversation[1]["role"] == "assistant"
        assert conversation[1]["content"] == "Hi there!"
        assert conversation[2]["role"] == "user"
        assert conversation[2]["content"] == "How are you?"
        assert conversation[3]["role"] == "assistant"
        assert conversation[3]["content"] == "I'm doing well, thanks!"
    
    async def test_get_stats(self):
        """Test getting session statistics."""
        from chuk_ai_session_manager import SessionManager
        
        sm = SessionManager()
        
        # Add some conversation
        await sm.user_says("Hello!")
        await sm.ai_responds("Hi there!")
        await sm.tool_used("test_tool", {"arg": "value"}, {"result": "success"})
        
        # Get stats
        stats = await sm.get_stats()
        
        assert "session_id" in stats
        assert "total_events" in stats
        assert "user_messages" in stats
        assert "ai_messages" in stats
        assert "tool_calls" in stats
        assert "total_tokens" in stats
        assert "estimated_cost" in stats
        
        assert stats["user_messages"] == 1
        assert stats["ai_messages"] == 1
        assert stats["tool_calls"] == 1
        assert stats["total_events"] >= 3
    
    async def test_track_conversation_basic(self):
        """Test basic track_conversation functionality."""
        from chuk_ai_session_manager import track_conversation
        
        session_id = await track_conversation(
            user_message="Hello!",
            ai_response="Hi there!",
            model="gpt-4",
            provider="openai"
        )
        
        assert isinstance(session_id, str)
        assert len(session_id) > 0
    
    async def test_track_conversation_infinite(self):
        """Test track_conversation with infinite context."""
        from chuk_ai_session_manager import track_infinite_conversation
        
        session_id = await track_infinite_conversation(
            user_message="Hello!",
            ai_response="Hi there!",
            model="gpt-4",
            provider="openai"
        )
        
        assert isinstance(session_id, str)
        assert len(session_id) > 0
    
    async def test_quick_conversation(self):
        """Test quick_conversation convenience function."""
        from chuk_ai_session_manager import quick_conversation
        
        stats = await quick_conversation(
            user_message="Hello!",
            ai_response="Hi there!"
        )
        
        assert isinstance(stats, dict)
        assert "session_id" in stats
        assert "user_messages" in stats
        assert "ai_messages" in stats
        assert stats["user_messages"] == 1
        assert stats["ai_messages"] == 1


class TestModelsWork:
    """Test that core models work independently."""
    
    def test_token_usage(self):
        """Test TokenUsage model."""
        from chuk_ai_session_manager.models.token_usage import TokenUsage
        
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            model="gpt-3.5-turbo"
        )
        
        assert usage.total_tokens == 150
        assert usage.estimated_cost_usd > 0
        assert usage.model == "gpt-3.5-turbo"
    
    async def test_token_usage_from_text(self):
        """Test TokenUsage.from_text async method."""
        from chuk_ai_session_manager.models.token_usage import TokenUsage
        
        usage = await TokenUsage.from_text(
            prompt="Hello, world!",
            completion="Hi there!",
            model="gpt-3.5-turbo"
        )
        
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0
        assert usage.total_tokens > 0
        assert usage.model == "gpt-3.5-turbo"
    
    async def test_session_event(self):
        """Test SessionEvent model."""
        from chuk_ai_session_manager import SessionEvent, EventSource, EventType
        
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
    
    async def test_session_event_with_tokens(self):
        """Test SessionEvent with token counting."""
        from chuk_ai_session_manager import SessionEvent, EventSource, EventType
        
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
        assert event.token_usage.total_tokens > 0
        assert event.token_usage.model == "gpt-3.5-turbo"
    
    async def test_session_event_metadata(self):
        """Test SessionEvent metadata operations."""
        from chuk_ai_session_manager import SessionEvent, EventSource, EventType
        
        event = SessionEvent(
            message="Test message",
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        
        # Test metadata operations
        await event.set_metadata("test_key", "test_value")
        assert await event.get_metadata("test_key") == "test_value"
        assert await event.has_metadata("test_key") == True
        assert await event.has_metadata("nonexistent") == False
        
        await event.remove_metadata("test_key")
        assert await event.has_metadata("test_key") == False


class TestStorageWorks:
    """Test that storage backend works."""
    
    def test_storage_setup(self):
        """Test storage setup function."""
        from chuk_ai_session_manager import setup_chuk_sessions_storage
        
        # Should not raise an error
        backend = setup_chuk_sessions_storage(
            sandbox_id="test-sandbox",
            default_ttl_hours=1
        )
        
        assert backend is not None
        assert hasattr(backend, 'sandbox_id')
        assert backend.sandbox_id == "test-sandbox"
    
    async def test_session_creation(self):
        """Test creating a session."""
        from chuk_ai_session_manager import Session
        
        session = await Session.create()
        
        assert session.id is not None
        assert session.metadata is not None
        assert session.events == []
        assert session.token_summary is not None
        assert session.total_tokens == 0
        assert session.total_cost == 0.0
    
    async def test_session_with_parent(self):
        """Test creating a session with parent."""
        from chuk_ai_session_manager import Session
        
        parent_session = await Session.create()
        child_session = await Session.create(parent_id=parent_session.id)
        
        assert child_session.parent_id == parent_session.id
        assert child_session.id != parent_session.id


class TestInfiniteContext:
    """Test infinite context functionality."""
    
    async def test_infinite_context_enabled(self):
        """Test that infinite context can be enabled."""
        from chuk_ai_session_manager import SessionManager
        
        sm = SessionManager(
            infinite_context=True,
            token_threshold=100,  # Low threshold for testing
            max_turns_per_segment=3
        )
        
        assert sm.is_infinite == True
        
        # Add several messages
        await sm.user_says("Message 1")
        await sm.ai_responds("Response 1")
        await sm.user_says("Message 2")
        await sm.ai_responds("Response 2")
        
        # Get conversation
        conversation = await sm.get_conversation()
        assert len(conversation) >= 4
        
        # Get session chain
        chain = await sm.get_session_chain()
        assert len(chain) >= 1
    
    async def test_infinite_context_stats(self):
        """Test infinite context statistics."""
        from chuk_ai_session_manager import SessionManager
        
        sm = SessionManager(infinite_context=True)
        
        await sm.user_says("Hello!")
        await sm.ai_responds("Hi there!")
        
        stats = await sm.get_stats()
        
        assert stats["infinite_context"] == True
        assert "session_chain" in stats
        assert stats["user_messages"] == 1
        assert stats["ai_messages"] == 1


class TestErrorHandling:
    """Test error handling works correctly."""
    
    async def test_invalid_session_id_graceful(self):
        """Test handling of invalid session ID gracefully."""
        from chuk_ai_session_manager import SessionManager
        
        # This should not crash, even with invalid session ID
        sm = SessionManager(session_id="nonexistent-session")
        
        # The session manager should handle this gracefully
        # when we try to use it - it should either create a new session
        # or give a clear error
        try:
            session_id = await sm.user_says("Test message")
            # If it doesn't error, that's fine - it means it created a new session
            assert isinstance(session_id, str)
            assert len(session_id) > 0
        except ValueError as e:
            # If it does error, make sure it's a reasonable error
            assert "not found" in str(e).lower()
        except Exception as e:
            # Any other exception should be informative
            assert len(str(e)) > 0
    
    async def test_empty_messages_handled(self):
        """Test handling of empty messages."""
        from chuk_ai_session_manager import SessionManager
        
        sm = SessionManager()
        
        # Empty user message
        session_id = await sm.user_says("")
        assert isinstance(session_id, str)
        
        # Empty AI response
        session_id = await sm.ai_responds("")
        assert isinstance(session_id, str)
    
    async def test_none_values_handled(self):
        """Test handling of None values in tool calls."""
        from chuk_ai_session_manager import SessionManager
        
        sm = SessionManager()
        
        # Tool call with None result
        session_id = await sm.tool_used(
            tool_name="test",
            arguments={},
            result=None
        )
        assert isinstance(session_id, str)


class TestIntegrationScenarios:
    """Test realistic usage scenarios."""
    
    async def test_complete_conversation_flow(self):
        """Test a complete conversation with tools."""
        from chuk_ai_session_manager import SessionManager
        
        sm = SessionManager()
        
        # User asks a question
        await sm.user_says("What's the weather in San Francisco?")
        
        # AI decides to use a tool
        await sm.ai_responds("I'll check the weather for you.")
        
        # Tool is used
        await sm.tool_used(
            tool_name="weather",
            arguments={"location": "San Francisco"},
            result={"temperature": "72°F", "condition": "sunny"}
        )
        
        # AI provides final response
        await sm.ai_responds("The weather in San Francisco is 72°F and sunny!")
        
        # Check the conversation flow
        conversation = await sm.get_conversation()
        assert len(conversation) == 3  # User, AI response, final AI response
        
        stats = await sm.get_stats()
        assert stats["user_messages"] == 1
        assert stats["ai_messages"] == 2
        assert stats["tool_calls"] == 1
    
    async def test_multi_turn_conversation(self):
        """Test a multi-turn conversation."""
        from chuk_ai_session_manager import SessionManager
        
        sm = SessionManager()
        
        # Multiple conversation turns
        turns = [
            ("user", "Hello!"),
            ("ai", "Hi there! How can I help you today?"),
            ("user", "I need help with Python programming."),
            ("ai", "I'd be happy to help with Python! What specific topic?"),
            ("user", "How do I create a class?"),
            ("ai", "Here's how to create a class in Python...")
        ]
        
        for role, message in turns:
            if role == "user":
                await sm.user_says(message)
            else:
                await sm.ai_responds(message)
        
        conversation = await sm.get_conversation()
        assert len(conversation) == 6
        
        # Check conversation structure
        for i, (expected_role, expected_message) in enumerate(turns):
            actual_role = "user" if expected_role == "user" else "assistant"
            assert conversation[i]["role"] == actual_role
            assert conversation[i]["content"] == expected_message
    
    async def test_convenience_function_integration(self):
        """Test integration using convenience functions."""
        from chuk_ai_session_manager import track_conversation, quick_conversation
        
        # Use track_conversation
        session_id1 = await track_conversation(
            "What's 2+2?",
            "2+2 equals 4",
            model="gpt-4"
        )
        
        # Use quick_conversation
        stats = await quick_conversation(
            "Hello!",
            "Hi there!"
        )
        
        assert isinstance(session_id1, str)
        assert isinstance(stats, dict)
        assert stats["user_messages"] == 1
        assert stats["ai_messages"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])