# tests/storage/providers/test_memory.py
"""
Tests for the in-memory session store.
"""
import pytest
import asyncio

from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.storage.providers.memory import InMemorySessionStore
from tests.storage.test_base import create_test_session


class TestInMemorySessionStore:
    """Tests for the InMemorySessionStore class."""
    
    @pytest.fixture
    def store(self):
        """Create a new in-memory store for each test."""
        return InMemorySessionStore()
    
    @pytest.mark.asyncio
    async def test_save_and_get(self, store):
        """Test saving and retrieving a session."""
        # Create a session
        session = await create_test_session()
        
        # Save the session
        await store.save(session)
        
        # Retrieve the session
        retrieved = await store.get(session.id)
        
        # Check that we got the expected data back
        assert retrieved is not None
        assert retrieved.id == session.id
        assert len(retrieved.events) == 2
        assert retrieved.events[0].message == "Test message 1"
        assert retrieved.events[1].message == "Test message 2"
        
        # Should be the same object instance (cached)
        assert retrieved is session
    
    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        """Test retrieving a non-existent session."""
        # Try to get a session that doesn't exist
        nonexistent = await store.get("does-not-exist")
        
        # Should return None
        assert nonexistent is None
    
    @pytest.mark.asyncio
    async def test_delete(self, store):
        """Test deleting a session."""
        # Create and save a session
        session = await create_test_session()
        await store.save(session)
        
        # Delete the session
        await store.delete(session.id)
        
        # Check it's gone
        assert await store.get(session.id) is None
    
    @pytest.mark.asyncio
    async def test_list_sessions(self, store):
        """Test listing sessions."""
        # Create and save multiple sessions
        sessions = [await create_test_session() for _ in range(3)]
        for session in sessions:
            await store.save(session)
        
        # List all sessions
        session_ids = await store.list_sessions()
        
        # Check we got all the sessions
        assert len(session_ids) == 3
        for session in sessions:
            assert session.id in session_ids
    
    @pytest.mark.asyncio
    async def test_list_sessions_with_prefix(self, store):
        """Test listing sessions with a prefix filter."""
        # Create and save sessions with different prefixes
        prefixed_sessions = []
        for i in range(3):
            session = Session()
            session.id = f"test_{i}"
            await store.save(session)
            prefixed_sessions.append(session)
        
        # Create some other sessions
        for i in range(2):
            session = Session()
            session.id = f"other_{i}"
            await store.save(session)
        
        # List sessions with the "test_" prefix
        session_ids = await store.list_sessions(prefix="test_")
        
        # Check we got only the test sessions
        assert len(session_ids) == 3
        for session in prefixed_sessions:
            assert session.id in session_ids
    
    @pytest.mark.asyncio
    async def test_update_session(self, store):
        """Test updating an existing session."""
        # Create and save a session
        session = await create_test_session()
        await store.save(session)
        
        # Create a new event asynchronously
        new_event = await SessionEvent.create_with_tokens(
            message="Test message 3",
            prompt="Test message 3",
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        
        # Add the event to the session
        await session.add_event(new_event)
        
        # Save again
        await store.save(session)
        
        # Retrieve and check
        retrieved = await store.get(session.id)
        assert retrieved is not None
        assert len(retrieved.events) == 3
        assert retrieved.events[2].message == "Test message 3"
    
    @pytest.mark.asyncio
    async def test_clear(self, store):
        """Test clearing all sessions."""
        # Create and save multiple sessions
        sessions = [await create_test_session() for _ in range(3)]
        for session in sessions:
            await store.save(session)
        
        # Clear the store
        await store.clear()
        
        # List all sessions - should be empty
        session_ids = await store.list_sessions()
        assert len(session_ids) == 0
        
        # Check each session - should be gone
        for session in sessions:
            assert await store.get(session.id) is None
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, store):
        """Test concurrent operations on the store."""
        # Create sessions
        sessions = [await create_test_session() for _ in range(10)]
        
        # Concurrently save all sessions
        await asyncio.gather(*[store.save(session) for session in sessions])
        
        # Concurrently get all sessions
        retrieved_sessions = await asyncio.gather(*[store.get(session.id) for session in sessions])
        
        # All sessions should be retrieved
        for i, session in enumerate(sessions):
            assert retrieved_sessions[i] is not None
            assert retrieved_sessions[i].id == session.id
    
    @pytest.mark.asyncio
    async def test_get_by_property(self, store):
        """Test finding sessions by metadata property."""
        # Create sessions with different properties
        category_a_sessions = []
        for i in range(3):
            session = await create_test_session()
            await session.metadata.set_property("category", "A")
            await store.save(session)
            category_a_sessions.append(session)
        
        category_b_sessions = []
        for i in range(2):
            session = await create_test_session()
            await session.metadata.set_property("category", "B")
            await store.save(session)
            category_b_sessions.append(session)
        
        # Find sessions by property
        category_a_results = await store.get_by_property("category", "A")
        
        # Check results
        assert len(category_a_results) == 3
        for session in category_a_results:
            assert session.id in [s.id for s in category_a_sessions]
        
        # Find sessions with a different property
        category_b_results = await store.get_by_property("category", "B")
        assert len(category_b_results) == 2
        
        # Find sessions with a non-existent property
        no_results = await store.get_by_property("category", "C")
        assert len(no_results) == 0
    
    @pytest.mark.asyncio
    async def test_get_by_state(self, store):
        """Test finding sessions by state value."""
        # Create sessions with different states
        active_sessions = []
        for i in range(3):
            session = await create_test_session()
            await session.set_state("status", "active")
            await store.save(session)
            active_sessions.append(session)
        
        inactive_sessions = []
        for i in range(2):
            session = await create_test_session()
            await session.set_state("status", "inactive")
            await store.save(session)
            inactive_sessions.append(session)
        
        # Find sessions by state
        active_results = await store.get_by_state("status", "active")
        
        # Check results
        assert len(active_results) == 3
        for session in active_results:
            assert session.id in [s.id for s in active_sessions]
        
        # Find sessions with a different state
        inactive_results = await store.get_by_state("status", "inactive")
        assert len(inactive_results) == 2
        
        # Find sessions with a non-existent state
        no_results = await store.get_by_state("status", "pending")
        assert len(no_results) == 0
    
    @pytest.mark.asyncio  
    async def test_count(self, store):
        """Test counting the number of sessions."""
        # Initially empty
        count = await store.count()
        assert count == 0
        
        # Add some sessions
        sessions = [await create_test_session() for _ in range(5)]
        for session in sessions:
            await store.save(session)
        
        # Check count
        count = await store.count()
        assert count == 5
        
        # Delete a session
        await store.delete(sessions[0].id)
        
        # Check count again
        count = await store.count()
        assert count == 4