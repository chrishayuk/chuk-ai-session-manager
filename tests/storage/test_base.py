# tests/storage/test_base.py
"""
Tests for the base session store functionality.
"""
import pytest

from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.storage.base import SessionStoreProvider
from chuk_ai_session_manager.storage.providers.memory import InMemorySessionStore


class TestSessionStoreProvider:
    """Tests for the SessionStoreProvider class."""
    
    def test_default_store(self):
        """Test that the default store is an InMemorySessionStore."""
        # Reset the provider to use the default store
        SessionStoreProvider._store = None
        
        # Get the store and check its type
        store = SessionStoreProvider.get_store()
        assert isinstance(store, InMemorySessionStore)
    
    def test_set_store(self):
        """Test setting a custom store."""
        # Create a custom store
        custom_store = InMemorySessionStore()
        
        # Set it as the global store
        SessionStoreProvider.set_store(custom_store)
        
        # Get the store and check it's our custom instance
        store = SessionStoreProvider.get_store()
        assert store is custom_store


async def create_test_session():
    """Create a test session with events asynchronously."""
    session = Session()
    
    # Create events with async token counting
    event1 = await SessionEvent.create_with_tokens(
        message="Test message 1",
        prompt="Test message 1",
        source=EventSource.USER,
        type=EventType.MESSAGE
    )
    
    event2 = await SessionEvent.create_with_tokens(
        message="Test message 2",
        prompt="",
        completion="Test message 2",
        source=EventSource.LLM,
        type=EventType.MESSAGE
    )
    
    # Add events to session
    await session.add_event(event1)
    await session.add_event(event2)
    
    return session