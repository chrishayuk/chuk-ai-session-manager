# tests/storage/providers/test_file.py
"""
Tests for the file-based session store.
"""
import json
import os
import pytest
import shutil
import tempfile
import asyncio
from pathlib import Path

from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.storage.providers.file import FileSessionStore, create_file_session_store
from tests.storage.test_base import create_test_session


class TestFileSessionStore:
    """Tests for the FileSessionStore class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        # Create a temp directory
        temp_dir = tempfile.mkdtemp()
        
        # Return it for the test to use
        yield temp_dir
        
        # Clean up after the test
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def store(self, temp_dir):
        """Create a file-based store using the temp directory."""
        return FileSessionStore(temp_dir)
    
    @pytest.mark.asyncio
    async def test_save_and_get(self, store, temp_dir):
        """Test saving and retrieving a session."""
        # Create a session
        session = await create_test_session()
        
        # Save the session
        await store.save(session)
        
        # Check that the file was created
        file_path = Path(temp_dir) / f"{session.id}.json"
        assert file_path.exists()
        
        # Clear cache to ensure we're reading from file
        await store.clear_cache()
        
        # Retrieve the session
        retrieved = await store.get(session.id)
        
        # Check that we got the expected data back
        assert retrieved is not None
        assert retrieved.id == session.id
        assert len(retrieved.events) == 2
        assert retrieved.events[0].message == "Test message 1"
        assert retrieved.events[1].message == "Test message 2"
    
    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        """Test retrieving a non-existent session."""
        # Try to get a session that doesn't exist
        nonexistent = await store.get("does-not-exist")
        
        # Should return None
        assert nonexistent is None
    
    @pytest.mark.asyncio
    async def test_delete(self, store, temp_dir):
        """Test deleting a session."""
        # Create and save a session
        session = await create_test_session()
        await store.save(session)
        
        # Check it's there
        file_path = Path(temp_dir) / f"{session.id}.json"
        assert file_path.exists()
        
        # Delete the session
        await store.delete(session.id)
        
        # Check it's gone from both the store and filesystem
        assert await store.get(session.id) is None
        assert not file_path.exists()
    
    @pytest.mark.asyncio
    async def test_list_sessions(self, store, temp_dir):
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
            session = await create_test_session()
            session.id = f"test_{i}"
            await store.save(session)
            prefixed_sessions.append(session)
        
        # Create some other sessions
        for i in range(2):
            session = await create_test_session()
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
        
        # Clear cache to ensure we're reading from file
        await store.clear_cache()
        
        # Retrieve and check
        retrieved = await store.get(session.id)
        assert retrieved is not None
        assert len(retrieved.events) == 3
        assert retrieved.events[2].message == "Test message 3"
    
    @pytest.mark.asyncio
    async def test_file_content(self, store, temp_dir):
        """Test the structure of the saved JSON file."""
        # Create and save a session
        session = await create_test_session()
        await store.save(session)
        
        # Read the file directly
        file_path = Path(temp_dir) / f"{session.id}.json"
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check the structure
        assert data["id"] == session.id
        assert len(data["events"]) == 2
        assert data["events"][0]["message"] == "Test message 1"
        assert data["events"][1]["message"] == "Test message 2"
    
    @pytest.mark.asyncio
    async def test_auto_save_false(self, temp_dir):
        """Test store with auto_save=False."""
        # Create store with auto_save disabled
        store = FileSessionStore(temp_dir, auto_save=False)
        
        # Create and save a session
        session = await create_test_session()
        await store.save(session)
        
        # The file should not exist yet
        file_path = Path(temp_dir) / f"{session.id}.json"
        assert not file_path.exists()
        
        # But we should still be able to get it from the store (from cache)
        retrieved = await store.get(session.id)
        assert retrieved is not None
        assert retrieved.id == session.id
        
        # Now flush the store
        await store.flush()
        
        # The file should now exist
        assert file_path.exists()
    
    @pytest.mark.asyncio
    async def test_persistence_across_stores(self, temp_dir):
        """Test that sessions persist between different store instances."""
        # Create a store and save a session
        store1 = FileSessionStore(temp_dir)
        session = await create_test_session()
        await store1.save(session)
        
        # Create a new store instance pointing to the same directory
        store2 = FileSessionStore(temp_dir)
        
        # We should be able to retrieve the session
        retrieved = await store2.get(session.id)
        assert retrieved is not None
        assert retrieved.id == session.id
        assert len(retrieved.events) == 2
    
    @pytest.mark.asyncio
    async def test_corrupted_file(self, store, temp_dir):
        """Test handling of corrupted JSON files."""
        # Create a corrupted file
        session_id = "corrupted"
        file_path = Path(temp_dir) / f"{session_id}.json"
        with open(file_path, 'w') as f:
            f.write("{not valid json")
        
        # Trying to get the session should return None
        assert await store.get(session_id) is None
    
    @pytest.mark.asyncio
    async def test_vacuum(self, temp_dir):
        """Test the vacuum functionality."""
        store = FileSessionStore(temp_dir)
        
        # Create some temporary files
        temp1 = Path(temp_dir) / "temp1.tmp"
        with open(temp1, 'w') as f:
            f.write("temp file 1")
            
        corrupt1 = Path(temp_dir) / "corrupt1.json"
        with open(corrupt1, 'w') as f:
            f.write("{invalid json")
        
        # Run vacuum
        fixed = await store.vacuum()
        
        # Should have fixed 2 files
        assert fixed == 2
        
        # Temp file should be deleted
        assert not temp1.exists()
        
        # Corrupt file should be renamed
        assert not corrupt1.exists()
        assert (Path(temp_dir) / "corrupt1.corrupt").exists()
    
    @pytest.mark.asyncio
    async def test_clear_cache(self, store, temp_dir):
        """Test clearing the cache."""
        # Create and save a session
        session = await create_test_session()
        await store.save(session)
        
        # Get it once to cache it
        cached = await store.get(session.id)
        assert cached is session  # Same instance
        
        # Clear the cache
        await store.clear_cache()
        
        # Get it again - should be a new instance
        fresh = await store.get(session.id)
        assert fresh is not session  # Different instance
        assert fresh.id == session.id  # But same content
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, temp_dir):
        """Test concurrent file operations."""
        store = FileSessionStore(temp_dir)
        
        # Create multiple sessions
        sessions = [await create_test_session() for _ in range(5)]
        
        # Save them concurrently
        await asyncio.gather(*[store.save(session) for session in sessions])
        
        # Check all files exist
        for session in sessions:
            file_path = Path(temp_dir) / f"{session.id}.json"
            assert file_path.exists()
        
        # Retrieve them concurrently
        retrieved = await asyncio.gather(*[store.get(session.id) for session in sessions])
        
        # All should be retrieved successfully
        for i, session in enumerate(sessions):
            assert retrieved[i] is not None
            assert retrieved[i].id == session.id
    
    @pytest.mark.asyncio
    async def test_create_file_session_store_factory(self, temp_dir):
        """Test the factory function."""
        # Create using the factory
        store = await create_file_session_store(temp_dir)
        
        assert isinstance(store, FileSessionStore)
        
        # Should work normally
        session = await create_test_session()
        await store.save(session)
        
        retrieved = await store.get(session.id)
        assert retrieved is not None
        assert retrieved.id == session.id