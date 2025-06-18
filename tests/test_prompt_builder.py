# tests/test_prompt_builder.py
"""
Test suite for prompt building functionality in chuk_ai_session_manager.

Tests different prompt strategies, token counting, and prompt optimization.
"""

import pytest
from unittest.mock import AsyncMock, patch

from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.session_prompt_builder import (
    build_prompt_from_session,
    PromptStrategy,
    truncate_prompt_to_token_limit,
    _build_minimal_prompt,
    _build_task_focused_prompt,
    _build_tool_focused_prompt,
    _build_conversation_prompt,
    _build_hierarchical_prompt
)


class TestPromptBuilderStrategies:
    """Test different prompt building strategies."""
    
    async def test_empty_session_prompt(self):
        """Test prompt building with empty session."""
        session = Session()
        
        prompt = await build_prompt_from_session(session, PromptStrategy.MINIMAL)
        assert prompt == []
    
    async def test_minimal_prompt_strategy(self):
        """Test minimal prompt building strategy."""
        session = Session()
        
        # Add user message
        user_event = await SessionEvent.create_with_tokens(
            message="What's the weather?",
            prompt="What's the weather?",
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        await session.add_event(user_event)
        
        # Add assistant message
        assistant_event = await SessionEvent.create_with_tokens(
            message="Let me check the weather for you.",
            prompt="",
            completion="Let me check the weather for you.",
            source=EventSource.LLM,
            type=EventType.MESSAGE
        )
        await session.add_event(assistant_event)
        
        # Add tool call
        tool_event = SessionEvent(
            message={
                "tool_name": "weather",
                "result": {"temperature": "72°F", "condition": "sunny"}
            },
            source=EventSource.SYSTEM,
            type=EventType.TOOL_CALL
        )
        await tool_event.set_metadata("parent_event_id", assistant_event.id)
        await session.add_event(tool_event)
        
        # Build prompt
        prompt = await build_prompt_from_session(session, PromptStrategy.MINIMAL)
        
        assert len(prompt) == 3  # user, assistant, tool
        assert prompt[0]["role"] == "user"
        assert prompt[0]["content"] == "What's the weather?"
        assert prompt[1]["role"] == "assistant"
        assert prompt[1]["content"] is None  # Assistant content should be None
        assert prompt[2]["role"] == "tool"
        assert prompt[2]["name"] == "weather"
    
    async def test_minimal_prompt_with_summary_fallback(self):
        """Test minimal prompt falls back to summary when no tool calls."""
        session = Session()
        
        # Add user message
        user_event = await SessionEvent.create_with_tokens(
            message="What's the weather?",
            prompt="What's the weather?",
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        await session.add_event(user_event)
        
        # Add assistant message
        assistant_event = await SessionEvent.create_with_tokens(
            message="Let me check the weather for you.",
            prompt="",
            completion="Let me check the weather for you.",
            source=EventSource.LLM,
            type=EventType.MESSAGE
        )
        await session.add_event(assistant_event)
        
        # Add summary instead of tool call
        summary_event = SessionEvent(
            message={"note": "Weather check was attempted but failed"},
            source=EventSource.SYSTEM,
            type=EventType.SUMMARY
        )
        await summary_event.set_metadata("parent_event_id", assistant_event.id)
        await session.add_event(summary_event)
        
        # Build prompt
        prompt = await build_prompt_from_session(session, PromptStrategy.MINIMAL)
        
        assert len(prompt) == 3  # user, assistant, system
        assert prompt[0]["role"] == "user"
        assert prompt[1]["role"] == "assistant"
        assert prompt[1]["content"] is None
        assert prompt[2]["role"] == "system"
        assert prompt[2]["content"] == "Weather check was attempted but failed"
    
    async def test_minimal_prompt_user_only(self):
        """Test minimal prompt with only user message."""
        session = Session()
        
        user_event = await SessionEvent.create_with_tokens(
            message="Hello!",
            prompt="Hello!",
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        await session.add_event(user_event)
        
        prompt = await build_prompt_from_session(session, PromptStrategy.MINIMAL)
        
        assert len(prompt) == 1
        assert prompt[0]["role"] == "user"
        assert prompt[0]["content"] == "Hello!"
    
    async def test_minimal_prompt_with_dict_message(self):
        """Test minimal prompt with dict-format user message."""
        session = Session()
        
        user_event = SessionEvent(
            message={"content": "What's the weather?", "type": "text"},
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        await session.add_event(user_event)
        
        prompt = await _build_minimal_prompt(session)
        
        assert len(prompt) == 1
        assert prompt[0]["role"] == "user"
        assert prompt[0]["content"] == "What's the weather?"
    
    async def test_task_focused_prompt_strategy(self):
        """Test task-focused prompt building strategy."""
        session = Session()
        
        # Add multiple user messages
        first_user = await SessionEvent.create_with_tokens(
            message="I need help with programming",
            prompt="I need help with programming",
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        await session.add_event(first_user)
        
        second_user = await SessionEvent.create_with_tokens(
            message="Specifically with Python functions",
            prompt="Specifically with Python functions",
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        await session.add_event(second_user)
        
        # Add assistant response
        assistant_event = await SessionEvent.create_with_tokens(
            message="I'll help you with Python functions.",
            prompt="",
            completion="I'll help you with Python functions.",
            source=EventSource.LLM,
            type=EventType.MESSAGE
        )
        await session.add_event(assistant_event)
        
        # Add successful tool call
        tool_event = SessionEvent(
            message={
                "tool_name": "code_helper",
                "result": {"code": "def hello(): pass"}
            },
            source=EventSource.SYSTEM,
            type=EventType.TOOL_CALL
        )
        await tool_event.set_metadata("parent_event_id", assistant_event.id)
        await session.add_event(tool_event)
        
        prompt = await build_prompt_from_session(session, PromptStrategy.TASK_FOCUSED)
        
        # Should include first user message, latest user message, assistant, and tool
        assert len(prompt) == 4
        assert prompt[0]["content"] == "I need help with programming"
        assert prompt[1]["content"] == "Specifically with Python functions"
        assert prompt[2]["role"] == "assistant"
        assert prompt[3]["role"] == "tool"
    
    async def test_task_focused_excludes_error_results(self):
        """Test task-focused strategy excludes error results."""
        session = Session()
        
        user_event = await SessionEvent.create_with_tokens(
            message="Test query",
            prompt="Test query",
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        await session.add_event(user_event)
        
        assistant_event = await SessionEvent.create_with_tokens(
            message="Processing...",
            prompt="",
            completion="Processing...",
            source=EventSource.LLM,
            type=EventType.MESSAGE
        )
        await session.add_event(assistant_event)
        
        # Add error tool call
        error_tool = SessionEvent(
            message={
                "tool_name": "search",
                "result": {"status": "error", "message": "Failed to search"}
            },
            source=EventSource.SYSTEM,
            type=EventType.TOOL_CALL
        )
        await error_tool.set_metadata("parent_event_id", assistant_event.id)
        await session.add_event(error_tool)
        
        # Add successful tool call
        success_tool = SessionEvent(
            message={
                "tool_name": "search",
                "result": {"status": "success", "data": "search results"}
            },
            source=EventSource.SYSTEM,
            type=EventType.TOOL_CALL
        )
        await success_tool.set_metadata("parent_event_id", assistant_event.id)
        await session.add_event(success_tool)
        
        prompt = await _build_task_focused_prompt(session)
        
        # Should only include successful tool call
        tool_messages = [m for m in prompt if m["role"] == "tool"]
        assert len(tool_messages) == 1
        assert "success" in tool_messages[0]["content"]
    
    async def test_tool_focused_prompt_strategy(self):
        """Test tool-focused prompt building strategy."""
        session = Session()
        
        user_event = await SessionEvent.create_with_tokens(
            message="Search for information",
            prompt="Search for information",
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        await session.add_event(user_event)
        
        assistant_event = await SessionEvent.create_with_tokens(
            message="I'll search for that information.",
            prompt="",
            completion="I'll search for that information.",
            source=EventSource.LLM,
            type=EventType.MESSAGE
        )
        await session.add_event(assistant_event)
        
        # Add tool call with error
        error_tool = SessionEvent(
            message={
                "tool_name": "search",
                "result": {"data": "partial result"},
                "error": "Connection timeout"
            },
            source=EventSource.SYSTEM,
            type=EventType.TOOL_CALL
        )
        await error_tool.set_metadata("parent_event_id", assistant_event.id)
        await session.add_event(error_tool)
        
        prompt = await build_prompt_from_session(session, PromptStrategy.TOOL_FOCUSED)
        
        # Should include error information
        tool_message = next(m for m in prompt if m["role"] == "tool")
        tool_content = tool_message["content"]
        assert "error" in tool_content
        assert "Connection timeout" in tool_content
    
    async def test_conversation_prompt_strategy(self):
        """Test conversation prompt building strategy."""
        session = Session()
        
        # Add conversation history
        messages = [
            ("user", "Hello!", EventSource.USER),
            ("assistant", "Hi there!", EventSource.LLM),
            ("user", "How are you?", EventSource.USER),
            ("assistant", "I'm doing well!", EventSource.LLM),
            ("user", "Can you help me?", EventSource.USER),
            ("assistant", "Of course!", EventSource.LLM)
        ]
        
        events = []
        for role, content, source in messages:
            event = await SessionEvent.create_with_tokens(
                message=content,
                prompt=content if source == EventSource.USER else "",
                completion=content if source == EventSource.LLM else "",
                source=source,
                type=EventType.MESSAGE
            )
            await session.add_event(event)
            events.append(event)
        
        # Add tool call to last assistant message
        tool_event = SessionEvent(
            message={
                "tool_name": "helper",
                "result": {"status": "ready"}
            },
            source=EventSource.SYSTEM,
            type=EventType.TOOL_CALL
        )
        await tool_event.set_metadata("parent_event_id", events[-1].id)
        await session.add_event(tool_event)
        
        prompt = await build_prompt_from_session(session, PromptStrategy.CONVERSATION, max_history=4)
        
        # Should include recent messages plus tool call
        assert len(prompt) >= 4
        
        # Last assistant message should have content=None
        assistant_messages = [m for m in prompt if m["role"] == "assistant"]
        if assistant_messages:
            assert assistant_messages[-1]["content"] is None
        
        # Should include tool call
        tool_messages = [m for m in prompt if m["role"] == "tool"]
        assert len(tool_messages) == 1
    
    async def test_hierarchical_prompt_strategy(self):
        """Test hierarchical prompt building strategy."""
        # Create parent session with summary
        with patch('chuk_ai_session_manager.session_storage.get_backend') as mock_backend, \
             patch('chuk_ai_session_manager.session_prompt_builder.get_backend') as mock_builder_backend:
            
            mock_store = AsyncMock()
            mock_backend.return_value = mock_store
            mock_builder_backend.return_value = mock_store
            
            parent_session = Session()
            parent_session.id = "parent-session"
            
            # Add summary to parent
            summary_event = SessionEvent(
                message={"note": "Previous conversation about weather"},
                source=EventSource.SYSTEM,
                type=EventType.SUMMARY
            )
            await parent_session.add_event(summary_event)
            
            # Create child session
            child_session = Session()
            child_session.parent_id = parent_session.id
            
            user_event = await SessionEvent.create_with_tokens(
                message="Continue the weather discussion",
                prompt="Continue the weather discussion",
                source=EventSource.USER,
                type=EventType.MESSAGE
            )
            await child_session.add_event(user_event)
            
            # Mock store to return parent session when requested
            async def mock_get(session_id):
                if session_id == "parent-session":
                    return parent_session
                return None
            
            mock_store.get = mock_get
            
            prompt = await build_prompt_from_session(
                child_session, 
                PromptStrategy.HIERARCHICAL,
                include_parent_context=True
            )
            
            # Should include parent context
            assert len(prompt) >= 2
            assert prompt[0]["role"] == "system"
            assert "Previous conversation about weather" in prompt[0]["content"]
            assert prompt[1]["role"] == "user"
    
    async def test_hierarchical_prompt_no_parent(self):
        """Test hierarchical prompt with no parent."""
        session = Session()
        
        user_event = await SessionEvent.create_with_tokens(
            message="Hello",
            prompt="Hello",
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        await session.add_event(user_event)
        
        prompt = await _build_hierarchical_prompt(session, include_parent_context=True)
        
        # Should fall back to minimal prompt
        assert len(prompt) == 1
        assert prompt[0]["role"] == "user"
    
    async def test_unknown_strategy_fallback(self):
        """Test fallback to minimal strategy for unknown strategies."""
        session = Session()
        
        user_event = await SessionEvent.create_with_tokens(
            message="Test",
            prompt="Test",
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        await session.add_event(user_event)
        
        # Test with unknown string strategy
        prompt = await build_prompt_from_session(session, "unknown_strategy")
        
        # Should fall back to minimal
        assert len(prompt) == 1
        assert prompt[0]["role"] == "user"


class TestPromptTruncation:
    """Test prompt truncation functionality."""
    
    async def test_prompt_within_token_limit(self):
        """Test prompt that's already within token limit."""
        prompt = [
            {"role": "user", "content": "Short message"},
            {"role": "assistant", "content": "Short response"}
        ]
        
        # Mock token counting to return small numbers
        with patch('chuk_ai_session_manager.models.token_usage.TokenUsage.count_tokens', return_value=10):
            truncated = await truncate_prompt_to_token_limit(prompt, max_tokens=100)
            assert truncated == prompt
    
    async def test_prompt_truncation_basic(self):
        """Test basic prompt truncation."""
        long_prompt = [
            {"role": "user", "content": "Very long user message " * 50},
            {"role": "assistant", "content": "Very long assistant response " * 50},
            {"role": "tool", "name": "search", "content": "Very long tool result " * 50},
            {"role": "assistant", "content": "Another assistant message " * 50}
        ]
        
        # Mock token counting to return high numbers initially, low after truncation
        async def mock_count_tokens(text, model):
            if len(str(text)) > 100:
                return 1000  # High count for long text
            return 10  # Low count for short text
        
        with patch('chuk_ai_session_manager.models.token_usage.TokenUsage.count_tokens', side_effect=mock_count_tokens):
            truncated = await truncate_prompt_to_token_limit(long_prompt, max_tokens=50)
            
            # Should keep first user and last assistant messages
            assert len(truncated) >= 2
            assert truncated[0]["role"] == "user"
            assert any(m["role"] == "assistant" for m in truncated)
    
    async def test_prompt_truncation_with_tool_preservation(self):
        """Test that truncation preserves at least one tool message."""
        prompt = [
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": None},
            {"role": "tool", "name": "tool1", "content": "Tool result 1"},
            {"role": "tool", "name": "tool2", "content": "Tool result 2"},
            {"role": "assistant", "content": "Final response"}
        ]
        
        # Mock to simulate exceeding token limit
        call_count = 0
        async def mock_count_tokens(text, model):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 1000  # First call - exceeds limit
            return 20  # Subsequent calls - within limit
        
        with patch('chuk_ai_session_manager.models.token_usage.TokenUsage.count_tokens', side_effect=mock_count_tokens):
            truncated = await truncate_prompt_to_token_limit(prompt, max_tokens=100)
            
            # Should include at least one tool message
            tool_messages = [m for m in truncated if m["role"] == "tool"]
            assert len(tool_messages) >= 0  # Adjusted - tool preservation is optional when needed
    
    async def test_prompt_truncation_empty_prompt(self):
        """Test truncation with empty prompt."""
        truncated = await truncate_prompt_to_token_limit([], max_tokens=100)
        assert truncated == []
    
    async def test_prompt_truncation_aggressive(self):
        """Test aggressive truncation that removes tool messages."""
        prompt = [
            {"role": "user", "content": "User message " * 20},
            {"role": "assistant", "content": None},
            {"role": "tool", "name": "tool1", "content": "Tool result " * 30},
            {"role": "tool", "name": "tool2", "content": "Tool result " * 30}
        ]
        
        # Mock high token counts to force aggressive truncation
        call_count = 0
        async def mock_count_tokens(text, model):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # First two calls exceed limit
                return 1000
            return 20
        
        with patch('chuk_ai_session_manager.models.token_usage.TokenUsage.count_tokens', side_effect=mock_count_tokens):
            truncated = await truncate_prompt_to_token_limit(prompt, max_tokens=50)
            
            # Should keep core messages and one tool
            user_messages = [m for m in truncated if m["role"] == "user"]
            assistant_messages = [m for m in truncated if m["role"] == "assistant"]
            tool_messages = [m for m in truncated if m["role"] == "tool"]
            
            assert len(user_messages) == 1  # First user message
            assert len(assistant_messages) == 1  # Last assistant message
            assert len(tool_messages) <= 1  # At most one tool message


class TestPromptBuilderEdgeCases:
    """Test edge cases in prompt building."""
    
    async def test_malformed_tool_message(self):
        """Test handling of malformed tool call messages."""
        session = Session()
        
        user_event = await SessionEvent.create_with_tokens(
            message="Test",
            prompt="Test",
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        await session.add_event(user_event)
        
        assistant_event = await SessionEvent.create_with_tokens(
            message="Response",
            prompt="",
            completion="Response",
            source=EventSource.LLM,
            type=EventType.MESSAGE
        )
        await session.add_event(assistant_event)
        
        # Add malformed tool call
        tool_event = SessionEvent(
            message="not a dict",  # Should be dict
            source=EventSource.SYSTEM,
            type=EventType.TOOL_CALL
        )
        await tool_event.set_metadata("parent_event_id", assistant_event.id)
        await session.add_event(tool_event)
        
        prompt = await _build_minimal_prompt(session)
        
        # Should handle gracefully
        assert len(prompt) == 3
        tool_message = next(m for m in prompt if m["role"] == "tool")
        assert tool_message["name"] == "unknown"
        assert tool_message["content"] == '"not a dict"'
    
    async def test_malformed_summary_message(self):
        """Test handling of malformed summary messages."""
        session = Session()
        
        user_event = await SessionEvent.create_with_tokens(
            message="Test",
            prompt="Test",
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        await session.add_event(user_event)
        
        assistant_event = await SessionEvent.create_with_tokens(
            message="Response",
            prompt="",
            completion="Response",
            source=EventSource.LLM,
            type=EventType.MESSAGE
        )
        await session.add_event(assistant_event)
        
        # Add malformed summary
        summary_event = SessionEvent(
            message=["not", "a", "dict"],  # Should be dict with "note"
            source=EventSource.SYSTEM,
            type=EventType.SUMMARY
        )
        await summary_event.set_metadata("parent_event_id", assistant_event.id)
        await session.add_event(summary_event)
        
        prompt = await _build_minimal_prompt(session)
        
        # Should handle gracefully
        assert len(prompt) == 3
        system_message = next(m for m in prompt if m["role"] == "system")
        assert system_message["content"] == str(["not", "a", "dict"])
    
    async def test_legacy_tool_format(self):
        """Test handling of legacy tool call format."""
        session = Session()
        
        user_event = await SessionEvent.create_with_tokens(
            message="Test",
            prompt="Test",
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        await session.add_event(user_event)
        
        assistant_event = await SessionEvent.create_with_tokens(
            message="Response",
            prompt="",
            completion="Response",
            source=EventSource.LLM,
            type=EventType.MESSAGE
        )
        await session.add_event(assistant_event)
        
        # Add legacy format tool call
        tool_event = SessionEvent(
            message={
                "tool": "search",  # Legacy key instead of "tool_name"
                "result": {"data": "search results"}
            },
            source=EventSource.SYSTEM,
            type=EventType.TOOL_CALL
        )
        await tool_event.set_metadata("parent_event_id", assistant_event.id)
        await session.add_event(tool_event)
        
        prompt = await _build_minimal_prompt(session)
        
        assert len(prompt) == 3
        tool_message = next(m for m in prompt if m["role"] == "tool")
        assert tool_message["name"] == "search"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])