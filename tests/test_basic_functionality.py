# test_basic_functionality.py
"""
Basic functionality test to verify the fixes work.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock

# Test the imports work correctly
def test_basic_imports():
    """Test that basic imports work without errors."""
    # Core models
    from chuk_ai_session_manager.models.event_source import EventSource
    from chuk_ai_session_manager.models.event_type import EventType
    from chuk_ai_session_manager.models.session import Session
    from chuk_ai_session_manager.models.session_event import SessionEvent
    
    # Storage
    from chuk_ai_session_manager.session_storage import SessionStorage, get_backend, ChukSessionsStore
    
    # Main package
    import chuk_ai_session_manager
    
    # Test compatibility alias
    assert hasattr(chuk_ai_session_manager, 'chuk_sessions_storage')
    assert hasattr(chuk_ai_session_manager.chuk_sessions_storage, 'SessionStorage')
    assert hasattr(chuk_ai_session_manager.chuk_sessions_storage, 'ChukSessionManager')
    
    print("✅ All imports successful!")


def test_datetime_serialization_fix():
    """Test that datetime objects can be serialized properly."""
    from chuk_ai_session_manager.models.session import Session
    from datetime import datetime, timezone
    
    # Create a session (which has datetime fields)
    session = Session()
    
    # Test that model_dump_json works (this was failing before)
    json_str = session.model_dump_json()
    assert isinstance(json_str, str)
    assert len(json_str) > 0
    
    # Test that we can parse it back
    import json
    data = json.loads(json_str)
    assert 'metadata' in data
    assert 'created_at' in data['metadata']
    
    # Test that we can recreate the session
    restored_session = Session.model_validate(data)
    assert restored_session.id == session.id
    assert isinstance(restored_session.metadata.created_at, datetime)
    
    print("✅ DateTime serialization works!")


@pytest.mark.asyncio
async def test_session_aware_tool_processor_imports():
    """Test that SessionAwareToolProcessor can be imported and used."""
    from chuk_ai_session_manager.session_aware_tool_processor import SessionAwareToolProcessor
    from chuk_ai_session_manager.models.session import Session
    from unittest.mock import AsyncMock, patch
    
    # Mock the CHUK Sessions dependencies
    with patch('chuk_ai_session_manager.session_storage.get_backend') as mock_backend:
        mock_store = AsyncMock()
        mock_session = Session()
        mock_store.get.return_value = mock_session
        
        with patch('chuk_ai_session_manager.session_storage.ChukSessionsStore') as mock_store_class:
            mock_store_class.return_value = mock_store
            mock_backend.return_value = MagicMock()
            
            # This should not raise NameError: name 'SessionStoreProvider' is not defined
            processor = SessionAwareToolProcessor("test-session-id")
            assert processor.session_id == "test-session-id"
            
            print("✅ SessionAwareToolProcessor works!")


@pytest.mark.asyncio 
async def test_simple_api_basic():
    """Test that SimpleAPI basic functionality works."""
    # Import from the correct path
    try:
        from chuk_ai_session_manager.simple_api import SessionManager
    except ImportError:
        # If simple_api doesn't exist yet, skip this test
        pytest.skip("simple_api module not available")
    
    from unittest.mock import patch, AsyncMock, MagicMock
    
    # Mock the storage layer
    with patch('chuk_ai_session_manager.session_storage.get_backend') as mock_backend, \
         patch('chuk_ai_session_manager.models.session.get_backend') as mock_session_backend:
        
        mock_store = AsyncMock()
        mock_store_instance = AsyncMock()
        
        # Mock session creation
        with patch('chuk_ai_session_manager.models.session.Session.create') as mock_create:
            mock_session = MagicMock()
            mock_session.id = "test-session-123"
            mock_session.add_event_and_save = AsyncMock()
            mock_create.return_value = mock_session
            
            sm = SessionManager()
            session_id = await sm.user_says("Hello!")
            
            assert session_id == "test-session-123"
            print("✅ Simple API works!")


if __name__ == "__main__":
    # Run the synchronous tests
    test_basic_imports()
    test_datetime_serialization_fix()
    
    # Run the async tests
    asyncio.run(test_session_aware_tool_processor_imports())
    asyncio.run(test_simple_api_basic())
    
    print("🎉 All basic functionality tests passed!")