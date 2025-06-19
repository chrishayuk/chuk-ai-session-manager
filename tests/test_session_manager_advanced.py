# tests/test_session_manager_advanced.py
"""
Advanced test suite for SessionManager.

Tests advanced features and edge cases including:
- Complex infinite context scenarios
- Session chain management
- Custom stores and callbacks
- Performance and memory usage
- Error recovery
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, call
from uuid import uuid4

from chuk_ai_session_manager import SessionManager
from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.token_usage import TokenUsage
from chuk_ai_session_manager.session_storage import ChukSessionsStore


class TestInfiniteContextAdvanced:
    """Advanced tests for infinite context functionality."""
    
    async def test_load_session_chain(self):
        """Test loading and reconstructing session chain."""
        # Use a shared mock store
        from chuk_ai_session_manager.session_storage import ChukSessionsStore
        
        # Create a mock store that persists sessions
        sessions_db = {}
        
        class MockStore:
            async def save(self, session):
                sessions_db[session.id] = session
            
            async def get(self, session_id):
                return sessions_db.get(session_id)
        
        mock_store = MockStore()
        
        # Create a chain of sessions manually with the same store
        sm1 = SessionManager(store=mock_store)
        await sm1.user_says("Session 1 message")
        await sm1.ai_responds("Session 1 response")
        session1_id = sm1.session_id
        
        sm2 = SessionManager(parent_id=session1_id, store=mock_store)
        await sm2.user_says("Session 2 message")
        await sm2.ai_responds("Session 2 response")
        session2_id = sm2.session_id
        
        sm3 = SessionManager(parent_id=session2_id, store=mock_store)
        await sm3.user_says("Session 3 message")
        await sm3.ai_responds("Session 3 response")
        session3_id = sm3.session_id
        
        # Create manager with infinite context and load chain
        sm_infinite = SessionManager(
            session_id=session3_id,
            infinite_context=True,
            store=mock_store
        )
        
        # Initialize and load the chain
        await sm_infinite._ensure_initialized()
        await sm_infinite.load_session_chain()
        
        # Should have reconstructed the chain
        chain = await sm_infinite.get_session_chain()
        assert len(chain) >= 1  # At least current session
        
        # Full conversation should include messages from the chain
        conversation = await sm_infinite.get_conversation(include_all_segments=True)
        assert len(conversation) == 6 
        
    async def test_summary_with_llm_callback(self):
        """Test using LLM callback for summary generation."""
        # Mock LLM callback
        async def mock_llm_summarizer(messages):
            # Simulate LLM summarization
            topics = []
            for msg in messages:
                if msg["role"] == "user":
                    # Extract key parts of the message
                    content = msg["content"].lower()
                    if "quantum" in content:
                        topics.append("quantum computing")
                    elif len(content) > 10:
                        topics.append(content[:20])
            
            return f"AI Summary: Discussed {len(topics)} topics including {', '.join(topics)}"
        
        sm = SessionManager(
            infinite_context=True,
            max_turns_per_segment=2
        )
        
        await sm.user_says("Tell me about quantum computing")
        await sm.ai_responds("Quantum computing uses quantum bits...")
        
        # Generate summary with callback
        summary = await sm._create_summary(mock_llm_summarizer)
        
        assert "AI Summary:" in summary
        assert "quantum computing" in summary
    
    async def test_segment_transition_preserves_context(self):
        """Test that context is preserved across segment transitions."""
        sm = SessionManager(
            system_prompt="You are a helpful assistant.",
            metadata={"session_type": "support"},
            infinite_context=True,
            max_turns_per_segment=2
        )
        
        # First segment
        await sm.user_says("Hello")
        await sm.ai_responds("Hi there!")
        first_segment_id = sm.session_id
        
        # Trigger new segment
        await sm.user_says("New segment")
        second_segment_id = sm.session_id
        
        assert first_segment_id != second_segment_id
        
        # System prompt should be preserved
        assert sm.system_prompt == "You are a helpful assistant."
        
        # New segment should have parent reference
        assert sm._session.parent_id == first_segment_id
        
        # Full conversation should include all messages
        full_conv = await sm.get_conversation(include_all_segments=True)
        assert len(full_conv) == 3
    
    async def test_complex_session_chain_stats(self):
        """Test statistics across complex session chains."""
        # Need to mock the store to track all sessions
        with patch('chuk_ai_session_manager.session_storage.ChukSessionsStore') as mock_store_class:
            sessions_db = {}
            
            async def mock_save(session):
                sessions_db[session.id] = session
            
            async def mock_get(session_id):
                return sessions_db.get(session_id)
            
            mock_store = AsyncMock()
            mock_store.save = mock_save
            mock_store.get = mock_get
            mock_store_class.return_value = mock_store
            
            sm = SessionManager(
                infinite_context=True,
                token_threshold=200,  # Reasonable threshold
                max_turns_per_segment=2  # Force segmentation every 2 turns
            )
            
            # Create multiple segments with different content
            messages = [
                ("What is AI?", "AI is artificial intelligence..."),
                ("Tell me more", "AI involves machine learning..."),  # This triggers segment 2
                ("What about deep learning?", "Deep learning is a subset..."),
                ("Give examples", "Examples include image recognition..."),  # This triggers segment 3
            ]
            
            for user_msg, ai_msg in messages:
                await sm.user_says(user_msg)
                await sm.ai_responds(ai_msg, model="gpt-3.5-turbo")
            
            # Get comprehensive stats
            stats = await sm.get_stats(include_all_segments=True)
            
            assert stats["user_messages"] == 4
            assert stats["ai_messages"] == 4
            assert stats["session_segments"] >= 2  # Should have at least 2 segments
            assert stats["total_messages"] == 8
            assert len(stats["session_chain"]) >= 2
    
    async def test_segment_limit_handling(self):
        """Test handling of segment limits."""
        # Test with very low limits
        sm = SessionManager(
            infinite_context=True,
            token_threshold=10,  # Extremely low
            max_turns_per_segment=1  # One turn per segment
        )
        
        # Each message should create a new segment
        for i in range(5):
            await sm.user_says(f"Message {i}")
        
        chain = await sm.get_session_chain()
        assert len(chain) >= 5  # At least one segment per message


class TestCustomStoreIntegration:
    """Test integration with custom session stores."""
    
    async def test_custom_store_operations(self):
        """Test all operations with custom store."""
        # Create mock store with full tracking
        saved_sessions = {}
        
        class MockStore:
            async def save(self, session):
                saved_sessions[session.id] = session
            
            async def get(self, session_id):
                return saved_sessions.get(session_id)
            
            async def delete(self, session_id):
                saved_sessions.pop(session_id, None)
            
            async def list_sessions(self, prefix=""):
                return [sid for sid in saved_sessions if sid.startswith(prefix)]
        
        store = MockStore()
        sm = SessionManager(store=store)
        
        # Test save
        await sm.user_says("Test message")
        assert sm.session_id in saved_sessions
        
        # Test load
        sm2 = SessionManager(session_id=sm.session_id, store=store)
        await sm2.user_says("Another message")
        
        # Both messages should be in the session
        conversation = await sm2.get_conversation()
        assert len(conversation) == 2
    
    async def test_store_error_handling(self):
        """Test error handling with store operations."""
        # Create store that fails
        failing_store = AsyncMock()
        failing_store.save.side_effect = Exception("Storage error")
        failing_store.get.return_value = None
        
        sm = SessionManager(store=failing_store)
        
        # Should raise the storage error
        with pytest.raises(Exception, match="Storage error"):
            await sm.user_says("This will fail to save")


class TestPerformanceAndMemory:
    """Test performance and memory usage."""
    
    async def test_large_conversation_handling(self):
        """Test handling very large conversations."""
        sm = SessionManager()
        
        # Add many messages
        message_count = 100
        for i in range(message_count):
            if i % 2 == 0:
                await sm.user_says(f"User message {i}")
            else:
                await sm.ai_responds(f"AI response {i}", model="gpt-3.5-turbo")
        
        # Should handle large conversations
        stats = await sm.get_stats()
        assert stats["total_messages"] == message_count
        assert stats["user_messages"] == 50
        assert stats["ai_messages"] == 50
        
        # Get conversation should work
        conversation = await sm.get_conversation()
        assert len(conversation) == message_count
    
    async def test_memory_efficiency_with_infinite_context(self):
        """Test memory efficiency in infinite context mode."""
        sm = SessionManager(
            infinite_context=True,
            token_threshold=500,
            max_turns_per_segment=10
        )
        
        # Add many messages across segments
        for i in range(50):
            await sm.user_says(f"Message {i} with some content")
            await sm.ai_responds(f"Response {i} with detailed information")
        
        # Check memory usage indirectly through data structures
        # Full conversation should be tracked
        assert len(sm._full_conversation) == 100
        
        # But individual sessions should be smaller
        assert len(sm._session.events) < 100  # Current segment only
    
    async def test_concurrent_session_managers(self):
        """Test multiple SessionManagers running concurrently."""
        # Create multiple managers
        managers = [SessionManager() for _ in range(10)]
        
        async def process_conversation(sm, index):
            for i in range(5):
                await sm.user_says(f"Manager {index}, message {i}")
                await sm.ai_responds(f"Response to manager {index}, message {i}")
            return sm.session_id
        
        # Run all conversations concurrently
        tasks = [
            process_conversation(sm, i) 
            for i, sm in enumerate(managers)
        ]
        session_ids = await asyncio.gather(*tasks)
        
        # All should have unique session IDs
        assert len(set(session_ids)) == 10
        
        # Each should have correct message count
        for sm in managers:
            stats = await sm.get_stats()
            assert stats["total_messages"] == 10


class TestErrorRecoveryAndResilience:
    """Test error recovery and resilience."""
    
    async def test_recovery_from_corrupt_session(self):
        """Test recovery when session data is corrupted."""
        sm = SessionManager()
        await sm.user_says("Initial message")
        
        # Corrupt the session
        sm._session.events = None  # This would normally break things
        
        # Try to continue - should handle gracefully
        try:
            await sm.ai_responds("Response")
        except AttributeError:
            # Expected if events is None
            pass
        
        # Should be able to reinitialize
        sm._initialized = False
        sm._session = None
        await sm._ensure_initialized()
        
        # New session should work
        await sm.user_says("Recovery message")
        stats = await sm.get_stats()
        assert stats["total_messages"] >= 1
    
    async def test_partial_event_data(self):
        """Test handling events with partial data."""
        sm = SessionManager()
        
        # Create event with minimal data
        from chuk_ai_session_manager.models.session_event import SessionEvent
        
        minimal_event = SessionEvent(
            message="Minimal",
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        
        # Add directly to session
        await sm._ensure_initialized()
        sm._session.events.append(minimal_event)
        
        # Should handle gracefully
        conversation = await sm.get_conversation()
        assert len(conversation) == 1
        assert conversation[0]["content"] == "Minimal"
    
    async def test_concurrent_modifications(self):
        """Test handling concurrent modifications to session."""
        sm = SessionManager()
        
        async def add_user_messages():
            for i in range(10):
                await sm.user_says(f"User {i}")
                await asyncio.sleep(0.01)
        
        async def add_ai_messages():
            for i in range(10):
                await sm.ai_responds(f"AI {i}")
                await asyncio.sleep(0.01)
        
        # Run concurrently
        await asyncio.gather(
            add_user_messages(),
            add_ai_messages()
        )
        
        # Should have all messages
        stats = await sm.get_stats()
        assert stats["total_messages"] == 20


class TestMetadataAndProperties:
    """Test metadata and property handling."""
    
    async def test_comprehensive_metadata(self):
        """Test comprehensive metadata handling."""
        metadata = {
            "user_id": "user-123",
            "session_type": "support",
            "priority": "high",
            "tags": ["urgent", "billing"],
            "custom_data": {
                "department": "sales",
                "region": "north"
            }
        }
        
        sm = SessionManager(metadata=metadata)
        await sm.user_says("Help needed")
        
        # All metadata should be preserved
        for key, value in metadata.items():
            assert sm._session.metadata.properties.get(key) == value
    
    async def test_metadata_updates(self):
        """Test updating metadata after creation."""
        sm = SessionManager()
        await sm.user_says("Initial message")
        
        # Update metadata
        sm._session.metadata.properties["status"] = "active"
        sm._session.metadata.properties["priority"] = "high"
        
        # Should be reflected in session
        assert sm._session.metadata.properties.get("status") == "active"
        assert sm._session.metadata.properties.get("priority") == "high"
    
    async def test_event_specific_metadata(self):
        """Test event-specific metadata."""
        sm = SessionManager()
        
        # Add message with specific metadata
        await sm.user_says(
            "Question about billing",
            category="billing",
            sentiment="negative",
            confidence=0.85
        )
        
        event = sm._session.events[0]
        assert event.metadata.get("category") == "billing"
        assert event.metadata.get("sentiment") == "negative"
        assert event.metadata.get("confidence") == 0.85


class TestSpecialScenarios:
    """Test special scenarios and use cases."""
    
    async def test_multi_modal_content(self):
        """Test handling multi-modal content (text with metadata about images, etc)."""
        sm = SessionManager()
        
        # User sends message with image reference
        await sm.user_says(
            "Look at this chart",
            attachments=["chart.png"],
            content_type="text_with_image"
        )
        
        # AI analyzes
        await sm.ai_responds(
            "I can see the chart shows an upward trend",
            analyzed_content=["chart.png"],
            confidence=0.92
        )
        
        # Check metadata was stored
        user_event = sm._session.events[0]
        assert user_event.metadata.get("attachments") == ["chart.png"]
        
        ai_event = sm._session.events[1]
        assert ai_event.metadata.get("analyzed_content") == ["chart.png"]
    
    async def test_conversation_branching(self):
        """Test conversation branching scenarios."""
        # Main conversation
        sm_main = SessionManager()
        await sm_main.user_says("Main topic")
        await sm_main.ai_responds("Main response")
        main_id = sm_main.session_id
        
        # Branch 1
        sm_branch1 = SessionManager(parent_id=main_id)
        await sm_branch1.user_says("Branch 1 exploration")
        await sm_branch1.ai_responds("Branch 1 response")
        
        # Branch 2 (alternative path)
        sm_branch2 = SessionManager(parent_id=main_id)
        await sm_branch2.user_says("Branch 2 exploration")
        await sm_branch2.ai_responds("Branch 2 response")
        
        # Branches should be independent
        branch1_conv = await sm_branch1.get_conversation()
        branch2_conv = await sm_branch2.get_conversation()
        
        assert len(branch1_conv) == 2
        assert len(branch2_conv) == 2
        assert branch1_conv[0]["content"] != branch2_conv[0]["content"]
    
    async def test_session_tagging_and_search(self):
        """Test session tagging for later search/retrieval."""
        # Create multiple tagged sessions
        sessions = []
        
        for i in range(5):
            sm = SessionManager(
                metadata={
                    "category": "support" if i % 2 == 0 else "sales",
                    "tags": [f"tag{i}", "common"],
                    "date": datetime.now().isoformat()
                }
            )
            await sm.user_says(f"Session {i} message")
            # Ensure metadata is properly set
            await sm._ensure_initialized()
            sessions.append(sm)
        
        # Verify metadata was properly set
        for i, sm in enumerate(sessions):
            expected_category = "support" if i % 2 == 0 else "sales"
            actual_category = sm._session.metadata.properties.get("category")
            assert actual_category == expected_category, f"Session {i} has wrong category: {actual_category}"
        
        # In a real implementation, you could search sessions by metadata
        support_sessions = [
            sm for sm in sessions 
            if sm._session and sm._session.metadata.properties.get("category") == "support"
        ]
        
        assert len(support_sessions) == 3  # 0, 2, 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])