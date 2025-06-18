# tests/test_tools.py
"""
Test suite for tool processing functionality in chuk_ai_session_manager.

Tests SessionAwareToolProcessor and sample tools integration.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from chuk_tool_processor.models.tool_result import ToolResult

from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.session_aware_tool_processor import SessionAwareToolProcessor
from chuk_ai_session_manager.exceptions import ToolProcessingError


@pytest.fixture
async def mock_tool_processor():
    """Mock ToolProcessor for testing."""
    mock_processor = MagicMock()
    mock_executor = AsyncMock()
    mock_processor.executor = mock_executor
    return mock_processor


@pytest.fixture
async def mock_session_store():
    """Mock session store for testing."""
    mock_store = AsyncMock()
    
    # Create a mock session object that behaves like Session but allows attribute assignment
    class MockSession:
        def __init__(self):
            self.id = "test-session"
            self.events = []
            self.add_event_and_save = AsyncMock()
    
    mock_session = MockSession()
    mock_store.get.return_value = mock_session
    mock_store.save = AsyncMock()
    return mock_store, mock_session


@pytest.fixture
async def tool_processor(mock_tool_processor):
    """SessionAwareToolProcessor instance with mocked dependencies."""
    # Create a mock session object
    class MockSession:
        def __init__(self):
            self.id = "test-session"
            self.events = []
            self.add_event_and_save = AsyncMock()
    
    mock_session = MockSession()
    
    # The autouse fixture from conftest.py should handle the backend mocking
    # We just need to ensure the store returns our session
    from chuk_ai_session_manager.session_storage import ChukSessionsStore
    
    # Get the mocked store instance from the autouse fixture
    store = ChukSessionsStore(None)  # Backend will be mocked by autouse fixture
    store.get = AsyncMock(return_value=mock_session)
    
    with patch('chuk_ai_session_manager.session_aware_tool_processor.ToolProcessor', return_value=mock_tool_processor):
        processor = SessionAwareToolProcessor(
            session_id="test-session",
            enable_caching=True,
            max_retries=2,
            retry_delay=0.1
        )
        yield processor, mock_session, store


class TestSessionAwareToolProcessor:
    """Test SessionAwareToolProcessor functionality."""
    
    async def test_tool_processor_initialization(self, mock_tool_processor):
        """Test tool processor initialization."""
        with patch('chuk_ai_session_manager.session_aware_tool_processor.ToolProcessor', return_value=mock_tool_processor):
            processor = SessionAwareToolProcessor(
                session_id="test-session",
                enable_caching=False,
                max_retries=3,
                retry_delay=0.5
            )
            
            assert processor.session_id == "test-session"
            assert processor.enable_caching == False
            assert processor.max_retries == 3
            assert processor.retry_delay == 0.5
            assert processor.cache == {}
    
    async def test_tool_processor_missing_executor(self):
        """Test error when ToolProcessor doesn't have executor."""
        mock_processor = MagicMock()
        # Explicitly delete the executor attribute to trigger the error
        if hasattr(mock_processor, 'executor'):
            delattr(mock_processor, 'executor')
        
        with patch('chuk_ai_session_manager.session_aware_tool_processor.ToolProcessor', return_value=mock_processor):
            with pytest.raises(AttributeError, match="missing `.executor`"):
                SessionAwareToolProcessor(session_id="test-session")
    
    async def test_tool_processor_create_method(self):
        """Test the create class method."""
        # Create a mock session object
        class MockSession:
            def __init__(self):
                self.id = "test-session"
                self.events = []
                self.add_event_and_save = AsyncMock()
        
        mock_session = MockSession()
        
        # The autouse fixture should handle backend mocking
        from chuk_ai_session_manager.session_storage import ChukSessionsStore
        store = ChukSessionsStore(None)
        store.get = AsyncMock(return_value=mock_session)
        
        with patch('chuk_ai_session_manager.session_aware_tool_processor.ToolProcessor') as mock_tp:
            mock_tp.return_value.executor = AsyncMock()
            
            processor = await SessionAwareToolProcessor.create("test-session")
            assert processor.session_id == "test-session"
            store.get.assert_called_once_with("test-session")
    
    async def test_tool_processor_create_session_not_found(self):
        """Test create method with nonexistent session."""
        # The autouse fixture should handle backend mocking
        from chuk_ai_session_manager.session_storage import ChukSessionsStore
        store = ChukSessionsStore(None)
        store.get = AsyncMock(return_value=None)  # Session not found
        
        with pytest.raises(ValueError, match="Session test-session not found"):
            await SessionAwareToolProcessor.create("test-session")
    
    async def test_process_llm_message_no_tool_calls(self, tool_processor):
        """Test processing LLM message with no tool calls."""
        processor, mock_session, mock_store = tool_processor
        
        llm_message = {
            "role": "assistant",
            "content": "I'll help you with that."
        }
        
        results = await processor.process_llm_message(llm_message, None)
        
        assert results == []
        # Should still create a parent event
        mock_session.add_event_and_save.assert_called_once()
    
    async def test_process_llm_message_with_tool_calls(self, tool_processor):
        """Test processing LLM message with tool calls."""
        processor, mock_session, mock_store = tool_processor
        
        # Mock successful tool execution
        mock_result = ToolResult(
            tool="calculator",
            result={"answer": 42},
            error=None
        )
        processor._tp.executor.execute.return_value = [mock_result]
        
        llm_message = {
            "role": "assistant",
            "content": "I'll calculate that for you.",
            "tool_calls": [
                {
                    "function": {
                        "name": "calculator",
                        "arguments": '{"operation": "add", "a": 20, "b": 22}'
                    }
                }
            ]
        }
        
        results = await processor.process_llm_message(llm_message, None)
        
        assert len(results) == 1
        assert results[0].tool == "calculator"
        assert results[0].result == {"answer": 42}
        assert results[0].error is None
        
        # Verify tool call was executed
        processor._tp.executor.execute.assert_called_once()
        
        # Verify events were logged
        # Should have at least 2 calls: parent message + tool call event
        assert mock_session.add_event_and_save.call_count >= 2
    
    async def test_tool_call_caching(self, tool_processor):
        """Test tool call result caching."""
        processor, mock_session, mock_store = tool_processor
        
        # Mock tool result
        mock_result = ToolResult(
            tool="calculator",
            result={"answer": 42},
            error=None
        )
        processor._tp.executor.execute.return_value = [mock_result]
        
        tool_call = {
            "function": {
                "name": "calculator",
                "arguments": '{"operation": "add", "a": 20, "b": 22}'
            }
        }
        
        llm_message = {
            "role": "assistant",
            "content": "Calculating...",
            "tool_calls": [tool_call]
        }
        
        # First call
        results1 = await processor.process_llm_message(llm_message, None)
        
        # Second call (should use cache)
        results2 = await processor.process_llm_message(llm_message, None)
        
        assert len(results1) == 1
        assert len(results2) == 1
        assert results1[0].result == results2[0].result
        
        # Executor should only be called once (first time)
        processor._tp.executor.execute.assert_called_once()
    
    async def test_tool_call_retry_mechanism(self, tool_processor):
        """Test tool call retry on failure."""
        processor, mock_session, mock_store = tool_processor
        
        # Mock executor to fail twice, then succeed
        call_count = 0
        async def mock_execute(calls):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Network error")
            return [ToolResult(tool="search", result={"data": "success"}, error=None)]
        
        processor._tp.executor.execute.side_effect = mock_execute
        
        llm_message = {
            "role": "assistant",
            "content": "Searching...",
            "tool_calls": [
                {
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "test"}'
                    }
                }
            ]
        }
        
        results = await processor.process_llm_message(llm_message, None)
        
        assert len(results) == 1
        assert results[0].result == {"data": "success"}
        assert call_count == 3  # Failed twice, succeeded on third try
    
    async def test_tool_call_max_retries_exceeded(self, tool_processor):
        """Test tool call when max retries are exceeded."""
        processor, mock_session, mock_store = tool_processor
        
        # Mock executor to always fail
        processor._tp.executor.execute.side_effect = Exception("Persistent error")
        
        llm_message = {
            "role": "assistant",
            "content": "Searching...",
            "tool_calls": [
                {
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "test"}'
                    }
                }
            ]
        }
        
        results = await processor.process_llm_message(llm_message, None)
        
        assert len(results) == 1
        assert results[0].result is None
        assert results[0].error == "Persistent error"
        
        # Should have tried max_retries + 1 times (2 + 1 = 3)
        assert processor._tp.executor.execute.call_count == 3
    
    async def test_tool_call_invalid_json_arguments(self, tool_processor):
        """Test tool call with invalid JSON arguments."""
        processor, mock_session, mock_store = tool_processor
        
        mock_result = ToolResult(
            tool="calculator",
            result={"processed": True},
            error=None
        )
        processor._tp.executor.execute.return_value = [mock_result]
        
        llm_message = {
            "role": "assistant",
            "content": "Calculating...",
            "tool_calls": [
                {
                    "function": {
                        "name": "calculator",
                        "arguments": "invalid json {"  # Invalid JSON
                    }
                }
            ]
        }
        
        results = await processor.process_llm_message(llm_message, None)
        
        assert len(results) == 1
        # Should handle invalid JSON gracefully
        call_args = processor._tp.executor.execute.call_args[0][0]
        assert call_args[0].arguments == {"raw": "invalid json {"}
    
    async def test_tool_call_event_logging(self, tool_processor):
        """Test that tool calls are properly logged as events."""
        processor, mock_session, mock_store = tool_processor
        
        mock_result = ToolResult(
            tool="weather",
            result={"temperature": "75°F"},
            error=None
        )
        processor._tp.executor.execute.return_value = [mock_result]
        
        llm_message = {
            "role": "assistant",
            "content": "Checking weather...",
            "tool_calls": [
                {
                    "function": {
                        "name": "weather",
                        "arguments": '{"location": "New York"}'
                    }
                }
            ]
        }
        
        await processor.process_llm_message(llm_message, None)
        
        # Check that session.add_event_and_save was called multiple times
        # Should have at least 2 calls: parent event + tool event
        assert mock_session.add_event_and_save.call_count >= 2
        
        # Check the actual events that were added to the session
        event_calls = mock_session.add_event_and_save.call_args_list
        
        # Find the tool call event
        tool_events = []
        for call in event_calls:
            event = call[0][0]  # First argument to add_event_and_save
            if hasattr(event, 'type') and event.type == EventType.TOOL_CALL:
                tool_events.append(event)
        
        assert len(tool_events) >= 1
        tool_event = tool_events[0]
        assert tool_event.message["tool"] == "weather"
        assert tool_event.source == EventSource.SYSTEM
    
    async def test_maybe_await_helper(self, tool_processor):
        """Test the _maybe_await helper method."""
        processor, _, _ = tool_processor
        
        # Test with regular value
        result = await processor._maybe_await("regular_value")
        assert result == "regular_value"
        
        # Test with coroutine
        async def async_func():
            return "async_result"
        
        result = await processor._maybe_await(async_func())
        assert result == "async_result"
    
    async def test_exec_calls_method(self, tool_processor):
        """Test the _exec_calls method."""
        processor, _, _ = tool_processor
        
        mock_result = ToolResult(
            tool="test_tool",
            result="test_result",
            error=None
        )
        processor._tp.executor.execute.return_value = [mock_result]
        
        calls = [
            {
                "function": {
                    "name": "test_tool",
                    "arguments": '{"param": "value"}'
                }
            }
        ]
        
        results = await processor._exec_calls(calls)
        
        assert len(results) == 1
        assert results[0].tool == "test_tool"
        assert results[0].result == "test_result"
        
        # Verify the tool call was properly constructed
        call_args = processor._tp.executor.execute.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0].tool == "test_tool"
        assert call_args[0].arguments == {"param": "value"}
    
    async def test_caching_disabled(self):
        """Test behavior when caching is disabled."""
        # Create a mock session object
        class MockSession:
            def __init__(self):
                self.id = "test-session"
                self.events = []
                self.add_event_and_save = AsyncMock()
        
        mock_session = MockSession()
        
        # The autouse fixture should handle backend mocking
        from chuk_ai_session_manager.session_storage import ChukSessionsStore
        store = ChukSessionsStore(None)
        store.get = AsyncMock(return_value=mock_session)
        
        with patch('chuk_ai_session_manager.session_aware_tool_processor.ToolProcessor') as mock_tp:
            mock_tp.return_value.executor = AsyncMock()
            mock_result = ToolResult(tool="test", result="result", error=None)
            mock_tp.return_value.executor.execute.return_value = [mock_result]
            
            processor = SessionAwareToolProcessor(
                session_id="test-session",
                enable_caching=False
            )
            
            llm_message = {
                "role": "assistant",
                "content": "Testing...",
                "tool_calls": [
                    {
                        "function": {
                            "name": "test",
                            "arguments": '{"param": "value"}'
                        }
                    }
                ]
            }
            
            # Call twice
            await processor.process_llm_message(llm_message, None)
            await processor.process_llm_message(llm_message, None)
            
            # Executor should be called twice (no caching)
            assert mock_tp.return_value.executor.execute.call_count == 2


class TestToolProcessorIntegration:
    """Integration tests for tool processor."""
    
    async def test_multiple_tool_calls_in_message(self, tool_processor):
        """Test processing multiple tool calls in a single message."""
        processor, mock_session, mock_store = tool_processor
        
        # The implementation processes tool calls one by one, not all at once
        # Mock the executor to be called twice, once for each tool
        results_sequence = [
            [ToolResult(tool="calculator", result={"answer": 42}, error=None)],
            [ToolResult(tool="weather", result={"temp": "75°F"}, error=None)]
        ]
        processor._tp.executor.execute.side_effect = results_sequence
        
        llm_message = {
            "role": "assistant",
            "content": "I'll calculate and check weather.",
            "tool_calls": [
                {
                    "function": {
                        "name": "calculator",
                        "arguments": '{"operation": "add", "a": 20, "b": 22}'
                    }
                },
                {
                    "function": {
                        "name": "weather",
                        "arguments": '{"location": "NYC"}'
                    }
                }
            ]
        }
        
        results = await processor.process_llm_message(llm_message, None)
        
        assert len(results) == 2
        assert results[0].tool == "calculator"
        assert results[1].tool == "weather"
        
        # The executor should be called twice (once for each tool call)
        assert processor._tp.executor.execute.call_count == 2
        
        # Should have logged events for each tool call plus the parent message
        # At least 3 calls: parent message + 2 tool calls
        assert mock_session.add_event_and_save.call_count >= 3
    
    async def test_mixed_success_failure_tools(self, tool_processor):
        """Test processing with both successful and failed tool calls."""
        processor, mock_session, mock_store = tool_processor
        
        # Mock one success, one failure per call
        call_count = 0
        async def mock_execute(calls):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [ToolResult(tool="success_tool", result={"data": "ok"}, error=None)]
            else:
                raise Exception("Tool failed")
        
        processor._tp.executor.execute.side_effect = mock_execute
        
        llm_message = {
            "role": "assistant",
            "content": "Running tools...",
            "tool_calls": [
                {
                    "function": {
                        "name": "success_tool",
                        "arguments": '{"param": "value"}'
                    }
                },
                {
                    "function": {
                        "name": "fail_tool",
                        "arguments": '{"param": "value"}'
                    }
                }
            ]
        }
        
        results = await processor.process_llm_message(llm_message, None)
        
        assert len(results) == 2
        assert results[0].error is None  # Success
        assert results[1].error is not None  # Failure
    
    async def test_tool_processor_state_isolation(self):
        """Test that different processor instances have isolated state."""
        # Create mock sessions
        class MockSession:
            def __init__(self, session_id):
                self.id = session_id
                self.events = []
                self.add_event_and_save = AsyncMock()
        
        mock_session1 = MockSession("session-1")
        mock_session2 = MockSession("session-2")
        
        # The autouse fixture should handle backend mocking
        from chuk_ai_session_manager.session_storage import ChukSessionsStore
        store = ChukSessionsStore(None)
        
        def get_session(session_id):
            if session_id == "session-1":
                return mock_session1
            elif session_id == "session-2":
                return mock_session2
            return None
        
        store.get = AsyncMock(side_effect=get_session)
        
        with patch('chuk_ai_session_manager.session_aware_tool_processor.ToolProcessor') as mock_tp:
            mock_tp.return_value.executor = AsyncMock()
            mock_tp.return_value.executor.execute.return_value = [
                ToolResult(tool="test", result="result", error=None)
            ]
            
            processor1 = SessionAwareToolProcessor(session_id="session-1")
            processor2 = SessionAwareToolProcessor(session_id="session-2")
            
            # Add item to processor1 cache
            processor1.cache["key1"] = "value1"
            
            # processor2 cache should be empty
            assert "key1" not in processor2.cache
            assert processor1.session_id != processor2.session_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])