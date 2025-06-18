# tests/test_session.py
import pytest
import time
from uuid import UUID
from datetime import datetime, timezone

# session
from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_metadata import SessionMetadata
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.session_run import SessionRun, RunStatus
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.session_storage import ChukSessionsStore, setup_chuk_sessions_storage

MessageT = str  # simple alias for tests

@pytest.fixture
def chuk_store():
    """Create and register a CHUK Sessions store for each test."""
    backend = setup_chuk_sessions_storage(sandbox_id="test-session")
    return ChukSessionsStore(backend)


def test_default_fields_and_metadata():
    sess = Session[MessageT]()
    # id is a valid UUID
    assert isinstance(sess.id, str)
    UUID(sess.id)
    # metadata is default SessionMetadata
    assert isinstance(sess.metadata, SessionMetadata)
    # no children, runs, events, or state
    assert sess.child_ids == []
    assert sess.runs == []
    assert sess.events == []
    assert sess.state == {}


def test_last_update_time_without_events():
    sess = Session[MessageT]()
    assert sess.last_update_time == sess.metadata.created_at


def test_last_update_time_with_events():
    sess = Session[MessageT]()
    e1 = SessionEvent(message="m1")
    time.sleep(0.001)
    e2 = SessionEvent(message="m2")
    sess.events = [e1, e2]
    assert sess.last_update_time == max(e1.timestamp, e2.timestamp)


def test_active_run_selection():
    sess = Session[MessageT]()
    r1 = SessionRun(status=RunStatus.COMPLETED)
    r2 = SessionRun(status=RunStatus.RUNNING)
    r3 = SessionRun(status=RunStatus.COMPLETED)
    sess.runs = [r1, r2, r3]
    assert sess.active_run is r2
    # without running
    sess.runs = [r1, r3]
    assert sess.active_run is None


@pytest.mark.asyncio
async def test_add_and_remove_child(chuk_store):
    sess = Session[MessageT]()
    await sess.add_child('c1')
    assert sess.child_ids == ['c1']
    # duplicate
    await sess.add_child('c1')
    assert sess.child_ids == ['c1']
    await sess.remove_child('c1')
    assert sess.child_ids == []


@pytest.mark.asyncio
async def test_hierarchy_sync_and_ancestors(chuk_store):
    # create parent and save
    parent = Session[MessageT]()
    await chuk_store.save(parent)
    
    # create child with parent_id
    child = Session[MessageT](parent_id=parent.id)
    await child.async_init()  # Need to call this explicitly now
    
    # Check parent's children
    parent = await chuk_store.get(parent.id)  # Reload parent
    assert child.id in parent.child_ids
    
    # ancestors
    anc = await child.ancestors()
    assert [s.id for s in anc] == [parent.id]


@pytest.mark.asyncio
async def test_descendants(chuk_store):
    # build root->child->grand
    root = Session[MessageT]()
    child = Session[MessageT]()
    grand = Session[MessageT]()
    
    await chuk_store.save(root)
    await chuk_store.save(child)
    await chuk_store.save(grand)
    
    root.child_ids = [child.id]
    child.child_ids = [grand.id]
    
    # Save updated sessions
    await chuk_store.save(root)
    await chuk_store.save(child)
    
    desc = await root.descendants()
    ids = [s.id for s in desc]
    assert child.id in ids and grand.id in ids
    
    # child descendants
    child_desc = await child.descendants()
    assert [s.id for s in child_desc] == [grand.id]


@pytest.mark.asyncio
async def test_sync_nonexistent_parent_does_not_error():
    sess = Session[MessageT](parent_id='nope')
    # This should not raise an error
    await sess.async_init()
    assert sess.parent_id == 'nope'


@pytest.mark.asyncio
async def test_add_event():
    sess = Session[MessageT]()
    event = SessionEvent(message="test")
    
    await sess.add_event(event)
    assert event in sess.events


@pytest.mark.asyncio
async def test_add_event_and_save(chuk_store):
    sess = Session[MessageT]()
    await chuk_store.save(sess)
    
    event = SessionEvent(message="test")
    await sess.add_event_and_save(event)
    
    # Verify event is added to session
    assert event in sess.events
    
    # Verify session was saved
    saved_sess = await chuk_store.get(sess.id)
    assert event in saved_sess.events


@pytest.mark.asyncio
async def test_token_usage_by_source():
    sess = Session[MessageT]()
    
    # Create events with token usage
    event1 = await SessionEvent.create_with_tokens(
        message="user message",
        prompt="user message",
        model="gpt-3.5-turbo",
        source=EventSource.USER
    )
    
    # Fix: Include prompt parameter
    event2 = await SessionEvent.create_with_tokens(
        message="system message",
        prompt="",  # Empty prompt
        completion="system message",
        model="gpt-3.5-turbo",
        source=EventSource.SYSTEM
    )
    
    # Add events to session
    await sess.add_event(event1)
    await sess.add_event(event2)
    
    # Get token usage by source
    usage_by_source = await sess.get_token_usage_by_source()
    
    # Check that both sources are present
    assert EventSource.USER.value in usage_by_source
    assert EventSource.SYSTEM.value in usage_by_source
    
    # Check token counts
    assert usage_by_source[EventSource.USER.value].total_prompt_tokens > 0
    assert usage_by_source[EventSource.SYSTEM.value].total_completion_tokens > 0


@pytest.mark.asyncio
async def test_token_usage_by_run():
    sess = Session[MessageT]()
    
    # Create events with token usage and task_id
    event1 = await SessionEvent.create_with_tokens(
        message="task 1 message",
        prompt="task 1 message",
        model="gpt-3.5-turbo",
        task_id="task1"
    )
    
    event2 = await SessionEvent.create_with_tokens(
        message="task 2 message",
        prompt="task 2 message",
        model="gpt-3.5-turbo",
        task_id="task2"
    )
    
    # Add events to session
    await sess.add_event(event1)
    await sess.add_event(event2)
    
    # Get token usage by run
    usage_by_run = await sess.get_token_usage_by_run()
    
    # Check that both runs are present
    assert "task1" in usage_by_run
    assert "task2" in usage_by_run
    
    # Check token counts
    assert usage_by_run["task1"].total_tokens > 0
    assert usage_by_run["task2"].total_tokens > 0


@pytest.mark.asyncio
async def test_count_message_tokens():
    sess = Session[MessageT]()
    
    # Test with string message
    str_tokens = await sess.count_message_tokens("This is a test message", "gpt-3.5-turbo")
    assert str_tokens > 0
    
    # Test with dictionary message
    dict_tokens = await sess.count_message_tokens({"content": "This is a test message"}, "gpt-3.5-turbo")
    assert dict_tokens > 0
    
    # String and dict should have the same token count
    assert str_tokens == dict_tokens


@pytest.mark.asyncio
async def test_state_management():
    sess = Session[MessageT]()
    
    # Set state
    await sess.set_state("test_key", "test_value")
    assert sess.state["test_key"] == "test_value"
    
    # Get state
    value = await sess.get_state("test_key")
    assert value == "test_value"
    
    # Default value for nonexistent key
    assert await sess.get_state("nonexistent", "default") == "default"
    
    # Has state
    assert await sess.has_state("test_key") is True
    assert await sess.has_state("nonexistent") is False
    
    # Remove state
    await sess.remove_state("test_key")
    assert await sess.has_state("test_key") is False


@pytest.mark.asyncio
async def test_create_class_method(chuk_store):
    parent_sess = Session[MessageT]()
    await chuk_store.save(parent_sess)
    
    # Create a session with parent
    sess = await Session.create(parent_id=parent_sess.id)
    
    assert isinstance(sess, Session)
    assert sess.parent_id == parent_sess.id
    
    # Verify parent-child relationship
    parent = await chuk_store.get(parent_sess.id)
    assert sess.id in parent.child_ids
    
    # Verify session was saved
    saved_sess = await chuk_store.get(sess.id)
    assert saved_sess is not None