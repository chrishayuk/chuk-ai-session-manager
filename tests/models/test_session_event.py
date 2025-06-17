# tests/session_event.py
import pytest
from datetime import datetime, timezone
import time
from uuid import UUID

from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.token_usage import TokenUsage


def test_default_session_event_fields():
    before = datetime.now(timezone.utc)
    time.sleep(0.001)
    event = SessionEvent(message="hello")
    time.sleep(0.001)
    after = datetime.now(timezone.utc)

    # id is a valid UUID string
    assert isinstance(event.id, str)
    UUID(event.id)  # should not raise

    # timestamp is set automatically and is within bounds
    assert before < event.timestamp < after
    assert event.timestamp.tzinfo == timezone.utc

    # default type and source
    assert event.type == EventType.MESSAGE
    assert event.source == EventSource.LLM

    # default metadata is empty dict
    assert isinstance(event.metadata, dict)
    assert event.metadata == {}

    # message and task_id
    assert event.message == "hello"
    assert event.task_id is None


def test_custom_fields_assignment():
    payload = {"foo": "bar"}
    evt = SessionEvent(
        message=payload,
        task_id="task123",
        type=EventType.TOOL_CALL,
        source=EventSource.USER,
        metadata={"key": 42}
    )

    assert evt.message == payload
    assert evt.task_id == "task123"
    assert evt.type == EventType.TOOL_CALL
    assert evt.source == EventSource.USER
    assert evt.metadata == {"key": 42}


def test_serialization_round_trip():
    evt = SessionEvent(
        message="test",
        task_id="t1",
        type=EventType.SUMMARY,
        source=EventSource.SYSTEM,
        metadata={"m": True}
    )
    data = evt.model_dump()

    # Ensure string values
    assert data["message"] == "test"
    assert data["task_id"] == "t1"
    assert data["type"] == EventType.SUMMARY
    assert data["source"] == EventSource.SYSTEM
    assert data["metadata"] == {"m": True}

    json_str = evt.model_dump_json()
    assert "test" in json_str
    assert "SUMMARY" not in json_str  # uses value, not name
    assert EventType.SUMMARY.value in json_str
    assert EventSource.SYSTEM.value in json_str


def test_invalid_type_and_source_raise_validation_error():
    with pytest.raises(Exception):
        SessionEvent(type="invalid_type")
    with pytest.raises(Exception):
        SessionEvent(source="noone")


@pytest.mark.asyncio
async def test_create_with_tokens():
    message = "Test message"
    prompt = "Test prompt"
    completion = "Test completion"
    model = "gpt-3.5-turbo"
    
    event = await SessionEvent.create_with_tokens(
        message=message,
        prompt=prompt,
        completion=completion,
        model=model,
        source=EventSource.USER,
        type=EventType.MESSAGE
    )
    
    assert event.message == message
    assert event.source == EventSource.USER
    assert event.type == EventType.MESSAGE
    assert event.token_usage is not None
    assert event.token_usage.prompt_tokens > 0
    assert event.token_usage.completion_tokens > 0
    assert event.token_usage.total_tokens > 0
    assert event.token_usage.model == model


@pytest.mark.asyncio
async def test_update_token_usage():
    event = SessionEvent(message="test")
    assert event.token_usage is None
    
    await event.update_token_usage(
        prompt="Test prompt",
        completion="Test completion",
        model="gpt-3.5-turbo"
    )
    
    assert event.token_usage is not None
    assert event.token_usage.prompt_tokens > 0
    assert event.token_usage.completion_tokens > 0
    assert event.token_usage.total_tokens > 0
    assert event.token_usage.model == "gpt-3.5-turbo"


@pytest.mark.asyncio
async def test_metadata_operations():
    event = SessionEvent(message="test")
    
    # Set metadata
    await event.set_metadata("test_key", "test_value")
    assert event.metadata["test_key"] == "test_value"
    
    # Get metadata
    assert await event.get_metadata("test_key") == "test_value"
    assert await event.get_metadata("nonexistent", "default") == "default"
    
    # Has metadata
    assert await event.has_metadata("test_key") is True
    assert await event.has_metadata("nonexistent") is False
    
    # Remove metadata
    await event.remove_metadata("test_key")
    assert await event.has_metadata("test_key") is False


@pytest.mark.asyncio
async def test_update_metadata_alias():
    event = SessionEvent(message="test")
    
    # Use update_metadata (alias for set_metadata)
    await event.update_metadata("test_key", "test_value")
    assert event.metadata["test_key"] == "test_value"
    assert await event.get_metadata("test_key") == "test_value"


@pytest.mark.asyncio
async def test_create_with_tokens_defaults():
    message = "Test message"
    
    # Create with minimal parameters
    event = await SessionEvent.create_with_tokens(
        message=message,
        prompt=message
    )
    
    assert event.message == message
    assert event.source == EventSource.LLM  # default
    assert event.type == EventType.MESSAGE  # default
    assert event.token_usage is not None
    assert event.token_usage.prompt_tokens > 0