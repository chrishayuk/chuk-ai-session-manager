# tests/test_infinite_conversation_advanced.py
"""
Advanced tests for the InfiniteConversationManager.
"""

import pytest
import pytest_asyncio
import asyncio
import time
from unittest.mock import AsyncMock, patch

from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.storage import SessionStoreProvider, InMemorySessionStore
from chuk_ai_session_manager.infinite_conversation import InfiniteConversationManager


@pytest_asyncio.fixture
async def store():
    s = InMemorySessionStore()
    SessionStoreProvider.set_store(s)
    return s


@pytest.fixture
def llm_callback():
    async def cb(messages, model="gpt-4"):
        return "reply"
    return AsyncMock(side_effect=cb)


@pytest.fixture
def manager(store, llm_callback):
    return InfiniteConversationManager(token_threshold=1000, max_turns_per_segment=20)


@pytest_asyncio.fixture
async def large_session(store):
    sess = Session()
    for i in range(50):
        await sess.add_event(SessionEvent(
            message=f"u{i}",
            source=EventSource.USER,
            type=EventType.MESSAGE
        ))
        await sess.add_event(SessionEvent(
            message=f"a{i}",
            source=EventSource.LLM,
            type=EventType.MESSAGE
        ))
    await store.save(sess)
    return sess


@pytest_asyncio.fixture
async def multi_segment_hierarchy(store, llm_callback):
    mgr = InfiniteConversationManager(token_threshold=1)
    root = Session()
    await store.save(root)

    child = Session(parent_id=root.id)
    await store.save(child)

    # record the child as a true child
    await root.add_child(child.id)
    await store.save(root)

    chain = [child.id]
    with patch.object(mgr, "_should_create_new_segment", return_value=True):
        cur = child.id
        for i in range(3):
            nxt = await mgr.process_message(
                cur,
                f"m{i}",
                EventSource.USER,
                llm_callback
            )
            if nxt != cur:
                chain.append(nxt)
                cur = nxt

    return {"root_id": root.id, "chain_ids": chain}


@pytest.mark.asyncio
async def test_large_session_performance(manager, large_session, llm_callback):
    start = time.time()
    with patch.object(manager, "_should_create_new_segment", return_value=False):
        nid = await manager.process_message(
            large_session.id,
            "ping",
            EventSource.USER,
            llm_callback
        )
    assert nid == large_session.id
    assert time.time() - start < 5.0

    upd = await SessionStoreProvider.get_store().get(large_session.id)
    # 100 original events + 1 new
    assert len(upd.events) == 101


@pytest.mark.asyncio
async def test_complex_hierarchy_navigation(multi_segment_hierarchy):
    info = multi_segment_hierarchy
    mgr = InfiniteConversationManager()
    chain = await mgr.get_session_chain(info["chain_ids"][-1])
    ids = {s.id for s in chain}
    assert set(info["chain_ids"]).issubset(ids)
    assert chain[-1].id == info["chain_ids"][-1]


@pytest.mark.asyncio
async def test_concurrent_message_processing(store, llm_callback):
    mgr = InfiniteConversationManager(token_threshold=1000)
    sess = Session()
    await store.save(sess)

    tasks = [
        asyncio.create_task(
            mgr.process_message(sess.id, f"c{i}", EventSource.USER, llm_callback)
        )
        for i in range(5)
    ]
    await asyncio.gather(*tasks)

    upd = await store.get(sess.id)
    assert len(upd.events) == 5


@pytest.mark.asyncio
async def test_error_handling_in_llm_callback(store):
    """Test that LLM callback errors are properly propagated during summarization."""
    mgr = InfiniteConversationManager(token_threshold=1, max_turns_per_segment=1)
    sess = Session()
    await store.save(sess)

    async def bad(*args):
        raise RuntimeError("fail")

    # With very low thresholds, even the first message triggers summarization
    with pytest.raises(RuntimeError):
        await mgr.process_message(sess.id, "first", EventSource.USER, bad)

    # The message should still have been added before summarization failed
    upd = await store.get(sess.id)
    assert len(upd.events) == 1


@pytest.mark.asyncio
async def test_zero_and_high_threshold(store, llm_callback):
    # zero threshold → always segment
    mgr0 = InfiniteConversationManager(token_threshold=0)
    s0 = Session()
    await store.save(s0)
    nid0 = await mgr0.process_message(s0.id, "hi", EventSource.USER, llm_callback)
    assert nid0 != s0.id

    # very high threshold → never segment
    mgrH = InfiniteConversationManager(token_threshold=10**6)
    sH = Session()
    await store.save(sH)
    for _ in range(5):
        nidH = await mgrH.process_message(sH.id, "hi", EventSource.USER, llm_callback)
        assert nidH == sH.id

    upd = await store.get(sH.id)
    assert len(upd.events) == 5
