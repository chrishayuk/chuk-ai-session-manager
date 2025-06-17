# tests/session_run.py
import pytest
from datetime import datetime, timezone
import time
from uuid import UUID

from chuk_ai_session_manager.models.session_run import SessionRun, RunStatus


def test_default_session_run_fields():
    before = datetime.now(timezone.utc)
    time.sleep(0.001)
    run = SessionRun()
    time.sleep(0.001)
    after = datetime.now(timezone.utc)

    # id is a valid UUID string
    assert isinstance(run.id, str)
    UUID(run.id)  # should not raise

    # started_at is set automatically and within the expected range
    assert isinstance(run.started_at, datetime)
    assert before < run.started_at < after
    assert run.started_at.tzinfo == timezone.utc

    # ended_at default is None
    assert run.ended_at is None

    # status defaults to PENDING
    assert run.status == RunStatus.PENDING

    # metadata defaults to empty dict
    assert isinstance(run.metadata, dict)
    assert run.metadata == {}


@pytest.mark.asyncio
async def test_mark_running_preserves_metadata_and_updates_started_at():
    run = SessionRun()
    original_started = run.started_at
    run.metadata['foo'] = 'bar'
    time.sleep(0.001)

    await run.mark_running()

    # status updated
    assert run.status == RunStatus.RUNNING
    # started_at updated
    assert run.started_at > original_started
    assert run.started_at.tzinfo == timezone.utc
    # ended_at remains None
    assert run.ended_at is None
    # metadata preserved
    assert run.metadata['foo'] == 'bar'


@pytest.mark.asyncio
async def test_mark_completed_sets_status_and_ended_at():
    run = SessionRun()
    time.sleep(0.001)

    await run.mark_completed()

    assert run.status == RunStatus.COMPLETED
    assert isinstance(run.ended_at, datetime)
    assert run.ended_at.tzinfo == timezone.utc
    # ended_at should be after started_at
    assert run.ended_at >= run.started_at


@pytest.mark.asyncio
async def test_mark_failed_sets_status_and_ended_at():
    run = SessionRun()
    time.sleep(0.001)

    await run.mark_failed()

    assert run.status == RunStatus.FAILED
    assert isinstance(run.ended_at, datetime)
    assert run.ended_at.tzinfo == timezone.utc
    assert run.ended_at >= run.started_at


@pytest.mark.asyncio
async def test_mark_cancelled_sets_status_and_ended_at():
    run = SessionRun()
    time.sleep(0.001)

    await run.mark_cancelled()

    assert run.status == RunStatus.CANCELLED
    assert isinstance(run.ended_at, datetime)
    assert run.ended_at.tzinfo == timezone.utc
    assert run.ended_at >= run.started_at


@pytest.mark.asyncio
async def test_mark_failed_with_reason():
    run = SessionRun()
    reason = "Test failure reason"
    
    await run.mark_failed(reason)
    
    assert run.status == RunStatus.FAILED
    assert await run.get_metadata("failure_reason") == reason


@pytest.mark.asyncio
async def test_mark_cancelled_with_reason():
    run = SessionRun()
    reason = "Test cancel reason"
    
    await run.mark_cancelled(reason)
    
    assert run.status == RunStatus.CANCELLED
    assert await run.get_metadata("cancel_reason") == reason


@pytest.mark.parametrize("status", list(RunStatus))
def test_runstatus_enum_values_and_roundtrip(status):
    # Ensure enum .value is a string
    assert isinstance(status.value, str)
    # Round-trip via constructor
    assert RunStatus(status.value) is status


def test_serialization_to_dict_and_json():
    run = SessionRun()
    d = run.model_dump()
    # Keys present
    assert 'id' in d
    assert 'started_at' in d
    assert 'status' in d
    assert 'metadata' in d

    j = run.model_dump_json()
    assert isinstance(j, str)
    # JSON string should contain the run id and status value
    assert run.id in j
    assert RunStatus.PENDING.value in j


@pytest.mark.asyncio
async def test_metadata_operations():
    run = SessionRun()
    
    # Set and get metadata
    await run.set_metadata("test_key", "test_value")
    assert await run.get_metadata("test_key") == "test_value"
    
    # Default value for nonexistent key
    assert await run.get_metadata("nonexistent", "default") == "default"
    
    # Has metadata
    assert await run.has_metadata("test_key") is True
    assert await run.has_metadata("nonexistent") is False
    
    # Remove metadata
    await run.remove_metadata("test_key")
    assert await run.has_metadata("test_key") is False


@pytest.mark.asyncio
async def test_get_duration():
    run = SessionRun()
    
    # No duration before the run is ended
    assert await run.get_duration() is None
    
    # Duration after completion
    await run.mark_completed()
    duration = await run.get_duration()
    assert isinstance(duration, float)
    assert duration >= 0.0


@pytest.mark.asyncio
async def test_create_class_method():
    metadata = {"test_key": "test_value"}
    run = await SessionRun.create(metadata)
    
    assert isinstance(run, SessionRun)
    assert run.metadata == metadata
    assert run.status == RunStatus.PENDING


@pytest.mark.asyncio
async def test_tool_call_association():
    run = SessionRun()
    
    # Add tool call
    tool_call_id = "test_tool_call_id"
    await run.add_tool_call(tool_call_id)
    
    assert tool_call_id in run.tool_calls
    
    # Adding same tool call again doesn't duplicate
    await run.add_tool_call(tool_call_id)
    assert run.tool_calls.count(tool_call_id) == 1


@pytest.mark.asyncio
async def test_to_dict():
    run = SessionRun()
    await run.mark_running()
    await run.set_metadata("test_key", "test_value")
    
    result = await run.to_dict()
    
    assert result["id"] == run.id
    assert result["status"] == RunStatus.RUNNING.value
    assert "started_at" in result
    assert "metadata" in result
    assert result["metadata"]["test_key"] == "test_value"
    assert "tool_calls" in result