# tests/test_infinite_conversation.py
"""
Async tests for the basic InfiniteConversationManager flows.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch

from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.storage import SessionStoreProvider, InMemorySessionStore
from chuk_ai_session_manager.infinite_conversation import InfiniteConversationManager


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

@pytest_asyncio.fixture
async def store():
    mem = InMemorySessionStore()
    SessionStoreProvider.set_store(mem)
    return mem


@pytest_asyncio.fixture
async def llm_callback():
    async def _cb(messages, model="gpt-4"):
        return "ok"
    return AsyncMock(side_effect=_cb)


@pytest_asyncio.fixture
async def manager(store):
    return InfiniteConversationManager(token_threshold=1000)


@pytest_asyncio.fixture
async def session(store):
    s = Session()
    for i, src in enumerate(
        [EventSource.USER, EventSource.LLM, EventSource.USER, EventSource.LLM]
    ):
        await s.add_event(
            SessionEvent(message=f"m{i}", source=src, type=EventType.MESSAGE)
        )
    await store.save(s)
    return s


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_process_message_no_segmentation(manager, session, llm_callback):
    sid = session.id
    nid = await manager.process_message(sid, "hi", EventSource.USER, llm_callback)
    assert nid == sid
    updated = await SessionStoreProvider.get_store().get(sid)
    assert updated.events[-1].message == "hi"


@pytest.mark.asyncio
async def test_process_message_with_segmentation(manager, session, llm_callback):
    sid = session.id
    with patch.object(manager, "_should_create_new_segment", return_value=True):
        nid = await manager.process_message(sid, "seg", EventSource.USER, llm_callback)
    assert nid != sid
    # summary recorded on original segment
    original = await SessionStoreProvider.get_store().get(sid)
    assert any(e.type == EventType.SUMMARY for e in original.events)


@pytest.mark.asyncio
async def test_build_context_for_llm_basic(manager, session):
    ctx = await manager.build_context_for_llm(session.id)
    assert len(ctx) == 4
    assert ctx[0]["role"] == "user"
    assert ctx[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_build_context_with_ancestors(manager, store):
    parent = Session()
    await parent.add_event(
        SessionEvent(message="prev-sum", source=EventSource.SYSTEM, type=EventType.SUMMARY)
    )
    await store.save(parent)

    child = Session(parent_id=parent.id)
    await child.add_event(
        SessionEvent(message="child-msg", source=EventSource.USER, type=EventType.MESSAGE)
    )
    await store.save(child)

    ctx = await manager.build_context_for_llm(child.id, include_summaries=True)
    # first message is parent-summary, second is child-user
    assert ctx[0]["role"] == "system"
    assert ctx[1]["role"] == "user"
    assert ctx[1]["content"] == "child-msg"


@pytest.mark.asyncio
async def test_full_conversation_history(store, llm_callback):
    mgr = InfiniteConversationManager(token_threshold=2)
    root = Session()
    await store.save(root)
    cur = root.id

    # two messages → segmentation because threshold=2
    with patch.object(mgr, "_should_create_new_segment", side_effect=[False, True]):
        cur = await mgr.process_message(cur, "u", EventSource.USER, llm_callback)
        cur = await mgr.process_message(cur, "a", EventSource.LLM, llm_callback)

    history = await mgr.get_full_conversation_history(cur)
    roles = [r for r, *_ in history]
    assert "user" in roles and "assistant" in roles


@pytest.mark.asyncio
async def test_empty_session(manager, store, llm_callback):
    s = Session()
    await store.save(s)
    await manager.process_message(s.id, "hi", EventSource.USER, llm_callback)
    updated = await store.get(s.id)
    assert len(updated.events) == 1


@pytest.mark.asyncio
async def test_get_session_chain(store):
    mgr = InfiniteConversationManager()
    root = Session(); await store.save(root)
    child = Session(parent_id=root.id); await store.save(child)
    grand = Session(parent_id=child.id); await store.save(grand)

    chain = await mgr.get_session_chain(grand.id)
    ids = [s.id for s in chain]

    # ensure relative order root → child → grandchild appears
    assert ids[-3:] == [root.id, child.id, grand.id]
