# tests/test_infinite_conversation_advanced.py
"""
Advanced tests for the InfiniteConversationManager.
"""

import pytest
import pytest_asyncio
import asyncio
import time
from unittest.mock import AsyncMock, patch, MagicMock

from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.session_storage import get_backend, ChukSessionsStore
from chuk_ai_session_manager.infinite_conversation import InfiniteConversationManager


class MockSessionStore:
    """Mock store for testing that mimics CHUK Sessions behavior."""
    
    def __init__(self):
        self._sessions = {}
        
    async def get(self, session_id: str):
        return self._sessions.get(session_id)
    
    async def save(self, session: Session):
        self._sessions[session.id] = session
    
    async def delete(self, session_id: str):
        self._sessions.pop(session_id, None)
    
    async def list_sessions(self, prefix: str = ""):
        if prefix:
            return [sid for sid in self._sessions.keys() if sid.startswith(prefix)]
        return list(self._sessions.keys())
    
    def clear(self):
        self._sessions.clear()


@pytest_asyncio.fixture
async def mock_store():
    """Create a mock store for testing."""
    store = MockSessionStore()
    return store


@pytest_asyncio.fixture
async def store_setup(mock_store):
    """Set up the storage backend for testing."""
    # Mock the get_backend function to return our mock store
    with patch('chuk_ai_session_manager.session_storage.get_backend') as mock_get_backend, \
         patch('chuk_ai_session_manager.infinite_conversation.get_backend') as mock_inf_get_backend:
        
        # Create a mock backend that returns our mock store
        mock_backend = MagicMock()
        mock_chuk_store = ChukSessionsStore(mock_backend)
        
        # Override the ChukSessionsStore methods to use our mock store
        async def mock_get(session_id):
            return await mock_store.get(session_id)
        
        async def mock_save(session):
            return await mock_store.save(session)
        
        async def mock_delete(session_id):
            return await mock_store.delete(session_id)
        
        async def mock_list(prefix=""):
            return await mock_store.list_sessions(prefix)
        
        mock_chuk_store.get = mock_get
        mock_chuk_store.save = mock_save
        mock_chuk_store.delete = mock_delete
        mock_chuk_store.list_sessions = mock_list
        
        # Make get_backend return our mock backend
        mock_get_backend.return_value = mock_backend
        mock_inf_get_backend.return_value = mock_backend
        
        # Patch ChukSessionsStore constructor to return our mock
        with patch('chuk_ai_session_manager.session_storage.ChukSessionsStore', return_value=mock_chuk_store), \
             patch('chuk_ai_session_manager.infinite_conversation.ChukSessionsStore', return_value=mock_chuk_store):
            
            yield mock_store


@pytest.fixture
def llm_callback():
    async def cb(messages, model="gpt-4"):
        return "reply"
    return AsyncMock(side_effect=cb)


@pytest.fixture
def manager():
    return InfiniteConversationManager(token_threshold=1000, max_turns_per_segment=20)


@pytest_asyncio.fixture
async def large_session(store_setup):
    store = store_setup
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
async def multi_segment_hierarchy(store_setup, llm_callback):
    store = store_setup
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
async def test_large_session_performance(manager, large_session, llm_callback, store_setup):
    store = store_setup
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

    upd = await store.get(large_session.id)
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
async def test_concurrent_message_processing(store_setup, llm_callback):
    store = store_setup
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
async def test_error_handling_in_llm_callback(store_setup):
    """Test that LLM callback errors are properly propagated during summarization."""
    store = store_setup
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
async def test_zero_and_high_threshold(store_setup, llm_callback):
    store = store_setup
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


@pytest.mark.asyncio
async def test_session_creation_with_parent(store_setup):
    """Test that new sessions are properly created with parent relationships."""
    store = store_setup
    mgr = InfiniteConversationManager(token_threshold=1)  # Very low threshold
    
    # Create initial session
    parent_session = Session()
    await store.save(parent_session)
    
    # Mock LLM callback
    async def mock_llm_callback(*args):
        return "Summary of conversation"
    
    # Process a message that should trigger segmentation
    with patch.object(mgr, "_should_create_new_segment", return_value=True):
        new_session_id = await mgr.process_message(
            parent_session.id,
            "test message",
            EventSource.USER,
            mock_llm_callback
        )
    
    # Verify new session was created
    assert new_session_id != parent_session.id
    
    # Verify parent session has summary
    updated_parent = await store.get(parent_session.id)
    summary_events = [e for e in updated_parent.events if e.type == EventType.SUMMARY]
    assert len(summary_events) == 1
    assert summary_events[0].message == "Summary of conversation"


@pytest.mark.asyncio
async def test_session_chain_ordering(store_setup):
    """Test that session chains are returned in correct order (root to leaf)."""
    store = store_setup
    
    # Create a chain: root -> child1 -> child2
    root = Session()
    await store.save(root)
    
    child1 = Session(parent_id=root.id)
    await store.save(child1)
    await root.add_child(child1.id)
    await store.save(root)
    
    child2 = Session(parent_id=child1.id)
    await store.save(child2)
    await child1.add_child(child2.id)
    await store.save(child1)
    
    # Get session chain
    mgr = InfiniteConversationManager()
    chain = await mgr.get_session_chain(child2.id)
    
    # Verify order: root -> child1 -> child2
    assert len(chain) == 3
    assert chain[0].id == root.id
    assert chain[1].id == child1.id
    assert chain[2].id == child2.id
    
    # Verify parent relationships
    assert chain[0].parent_id is None
    assert chain[1].parent_id == root.id
    assert chain[2].parent_id == child1.id