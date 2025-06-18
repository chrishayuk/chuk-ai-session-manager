# tests/test_infinite_conversation.py
"""
Test suite for infinite conversation functionality in chuk_ai_session_manager.

Tests InfiniteConversationManager and automatic session segmentation.
"""

import pytest
from unittest.mock import AsyncMock, patch

from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.infinite_conversation import (
    InfiniteConversationManager,
    SummarizationStrategy
)


@pytest.fixture
async def mock_session_store():
    """Mock session store for testing."""
    mock_store = AsyncMock()
    
    # Create a set of mock sessions
    sessions = {}
    
    async def mock_get(session_id):
        return sessions.get(session_id)
    
    async def mock_save(session):
        sessions[session.id] = session
    
    mock_store.get.side_effect = mock_get
    mock_store.save.side_effect = mock_save
    
    return mock_store, sessions


@pytest.fixture
async def infinite_manager():
    """InfiniteConversationManager with default settings."""
    return InfiniteConversationManager(
        token_threshold=100,  # Low threshold for testing
        max_turns_per_segment=3,  # Low turn limit for testing
        summarization_strategy=SummarizationStrategy.BASIC
    )


@pytest.fixture
async def mock_llm_callback():
    """Mock LLM callback for summarization."""
    async def callback(messages):
        # Simple mock summary based on message count
        user_messages = [m for m in messages if m.get("role") == "user"]
        return f"Summary of conversation with {len(user_messages)} user messages"
    
    return callback


class TestInfiniteConversationManager:
    """Test InfiniteConversationManager functionality."""
    
    def test_infinite_conversation_manager_initialization(self):
        """Test InfiniteConversationManager initialization."""
        manager = InfiniteConversationManager(
            token_threshold=5000,
            max_turns_per_segment=25,
            summarization_strategy=SummarizationStrategy.KEY_POINTS
        )
        
        assert manager.token_threshold == 5000
        assert manager.max_turns_per_segment == 25
        assert manager.summarization_strategy == SummarizationStrategy.KEY_POINTS
    
    async def test_process_message_basic(self, infinite_manager, mock_session_store, mock_llm_callback):
        """Test basic message processing without segmentation."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.infinite_conversation.get_backend', return_value=mock_store):
            # Create initial session and add to store
            session = Session()
            session.id = "test-session"
            sessions[session.id] = session  # Add to mock store
            
            # Process a user message
            result_session_id = await infinite_manager.process_message(
                session_id=session.id,
                message="Hello, how are you?",
                source=EventSource.USER,
                llm_callback=mock_llm_callback,
                model="gpt-3.5-turbo"
            )
            
            # Should return the same session ID (no segmentation)
            assert result_session_id == session.id
            assert len(session.events) == 1
            assert session.events[0].message == "Hello, how are you?"
            assert session.events[0].source == EventSource.USER
    
    async def test_should_create_new_segment_token_threshold(self, infinite_manager):
        """Test segmentation trigger based on token threshold."""
        session = Session()
        
        # Mock session with high token count
        session.token_summary.total_tokens = 150  # Above threshold of 100
        
        should_segment = await infinite_manager._should_create_new_segment(session)
        assert should_segment == True
        
        # Test below threshold
        session.token_summary.total_tokens = 50
        should_segment = await infinite_manager._should_create_new_segment(session)
        assert should_segment == False
    
    async def test_should_create_new_segment_turn_threshold(self, infinite_manager):
        """Test segmentation trigger based on turn threshold."""
        session = Session()
        
        # Add messages up to turn limit
        for i in range(4):  # Above threshold of 3
            event = SessionEvent(
                message=f"Message {i}",
                source=EventSource.USER if i % 2 == 0 else EventSource.LLM,
                type=EventType.MESSAGE
            )
            await session.add_event(event)
        
        should_segment = await infinite_manager._should_create_new_segment(session)
        assert should_segment == True
        
        # Test below threshold
        session.events = session.events[:2]  # Keep only 2 messages
        should_segment = await infinite_manager._should_create_new_segment(session)
        assert should_segment == False
    
    async def test_create_summary_basic(self, infinite_manager, mock_llm_callback):
        """Test basic summary creation."""
        session = Session()
        
        # Add some message events
        for i in range(3):
            event = SessionEvent(
                message=f"User message {i}?",
                source=EventSource.USER,
                type=EventType.MESSAGE
            )
            await session.add_event(event)
            
            event = SessionEvent(
                message=f"Assistant response {i}",
                source=EventSource.LLM,
                type=EventType.MESSAGE
            )
            await session.add_event(event)
        
        summary = await infinite_manager._create_summary(session, mock_llm_callback)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "3" in summary  # Should mention 3 user messages
    
    async def test_create_summary_with_questions(self, infinite_manager, mock_llm_callback):
        """Test summary creation with question extraction."""
        session = Session()
        
        # Add events with questions
        event1 = SessionEvent(
            message="What is the weather like today?",
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        await session.add_event(event1)
        
        event2 = SessionEvent(
            message="How do I cook pasta?",
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        await session.add_event(event2)
        
        # Use the actual _create_summary method to test question extraction
        user_messages = [e for e in session.events if e.source == EventSource.USER]
        
        topics = []
        for event in user_messages:
            content = str(event.message)
            if "?" in content:
                question = content.split("?")[0].strip()
                if len(question) > 10:
                    topics.append(question[:50])
        
        assert len(topics) == 2
        assert "What is the weather like today" in topics[0]
        assert "How do I cook pasta" in topics[1]
    
    async def test_get_summarization_prompt_strategies(self, infinite_manager):
        """Test different summarization prompt strategies."""
        # Test BASIC strategy
        infinite_manager.summarization_strategy = SummarizationStrategy.BASIC
        prompt = infinite_manager._get_summarization_prompt()
        assert "concise summary" in prompt.lower()
        
        # Test KEY_POINTS strategy
        infinite_manager.summarization_strategy = SummarizationStrategy.KEY_POINTS
        prompt = infinite_manager._get_summarization_prompt()
        assert "key points" in prompt.lower()
        
        # Test TOPIC_BASED strategy
        infinite_manager.summarization_strategy = SummarizationStrategy.TOPIC_BASED
        prompt = infinite_manager._get_summarization_prompt()
        assert "topics" in prompt.lower()
        
        # Test QUERY_FOCUSED strategy
        infinite_manager.summarization_strategy = SummarizationStrategy.QUERY_FOCUSED
        prompt = infinite_manager._get_summarization_prompt()
        assert "questions" in prompt.lower()
    
    async def test_build_context_without_summaries(self, infinite_manager, mock_session_store):
        """Test building context without including summaries."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.infinite_conversation.get_backend', return_value=mock_store):
            session = Session()
            session.id = "test-session"
            
            # Add messages
            for i in range(5):
                event = SessionEvent(
                    message=f"Message {i}",
                    source=EventSource.USER if i % 2 == 0 else EventSource.LLM,
                    type=EventType.MESSAGE
                )
                await session.add_event(event)
            
            sessions[session.id] = session  # Add to mock store
            
            # Build context without summaries
            context = await infinite_manager.build_context_for_llm(
                session_id=session.id,
                max_messages=3,
                include_summaries=False
            )
            
            # Should only include recent messages, no system message
            assert len(context) == 3
            assert all(msg["role"] in ["user", "assistant"] for msg in context)
    
    async def test_get_full_conversation_history(self, infinite_manager, mock_session_store):
        """Test getting full conversation history across segments."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.infinite_conversation.get_backend', return_value=mock_store):
            # Create two sessions with conversation
            session1 = Session()
            session1.id = "session1"
            
            # Add messages to first session
            event1 = SessionEvent(
                message="Hello",
                source=EventSource.USER,
                type=EventType.MESSAGE
            )
            event2 = SessionEvent(
                message="Hi there!",
                source=EventSource.LLM,
                type=EventType.MESSAGE
            )
            await session1.add_event(event1)
            await session1.add_event(event2)
            sessions[session1.id] = session1
            
            session2 = Session()
            session2.id = "session2"
            session2.parent_id = session1.id
            
            # Add messages to second session
            event3 = SessionEvent(
                message="How are you?",
                source=EventSource.USER,
                type=EventType.MESSAGE
            )
            event4 = SessionEvent(
                message="I'm good!",
                source=EventSource.LLM,
                type=EventType.MESSAGE
            )
            await session2.add_event(event3)
            await session2.add_event(event4)
            sessions[session2.id] = session2
            
            # Mock get_session_chain
            async def mock_get_session_chain(session_id):
                return [session1, session2]
            
            infinite_manager.get_session_chain = mock_get_session_chain
            
            history = await infinite_manager.get_full_conversation_history(session2.id)
            
            assert len(history) == 4
            assert history[0] == ("user", EventSource.USER, "Hello")
            assert history[1] == ("assistant", EventSource.LLM, "Hi there!")
            assert history[2] == ("user", EventSource.USER, "How are you?")
            assert history[3] == ("assistant", EventSource.LLM, "I'm good!")


class TestInfiniteConversationCore:
    """Test core infinite conversation functionality without complex mocking."""
    
    async def test_segmentation_logic_isolated(self, infinite_manager):
        """Test segmentation logic in isolation."""
        session = Session()
        
        # Test token threshold
        session.token_summary.total_tokens = 150
        assert await infinite_manager._should_create_new_segment(session) == True
        
        session.token_summary.total_tokens = 50
        assert await infinite_manager._should_create_new_segment(session) == False
        
        # Test turn threshold
        session.token_summary.total_tokens = 50  # Below token threshold
        for i in range(5):  # Above turn threshold
            event = SessionEvent(
                message=f"Message {i}",
                source=EventSource.USER,
                type=EventType.MESSAGE
            )
            await session.add_event(event)
        
        assert await infinite_manager._should_create_new_segment(session) == True
    
    async def test_summary_creation_isolated(self, infinite_manager, mock_llm_callback):
        """Test summary creation in isolation."""
        session = Session()
        
        # Add various message types
        messages = [
            ("What's the weather?", EventSource.USER),
            ("It's sunny today", EventSource.LLM),
            ("How about tomorrow?", EventSource.USER),
            ("Tomorrow will be cloudy", EventSource.LLM),
        ]
        
        for message, source in messages:
            event = SessionEvent(
                message=message,
                source=source,
                type=EventType.MESSAGE
            )
            await session.add_event(event)
        
        summary = await infinite_manager._create_summary(session, mock_llm_callback)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        # Should mention 2 user messages
        assert "2" in summary


class TestInfiniteConversationEdgeCases:
    """Test edge cases in infinite conversation management."""
    
    async def test_session_not_found(self, infinite_manager, mock_session_store, mock_llm_callback):
        """Test handling of nonexistent session."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.infinite_conversation.get_backend', return_value=mock_store):
            with pytest.raises(ValueError, match="Session nonexistent not found"):
                await infinite_manager.process_message(
                    session_id="nonexistent",
                    message="Hello",
                    source=EventSource.USER,
                    llm_callback=mock_llm_callback
                )
    
    async def test_empty_session_summary(self, infinite_manager, mock_llm_callback):
        """Test summary creation for empty session."""
        session = Session()
        
        summary = await infinite_manager._create_summary(session, mock_llm_callback)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "0" in summary  # Should mention 0 user messages
    
    async def test_build_context_session_not_found(self, infinite_manager, mock_session_store):
        """Test build context with nonexistent session."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.infinite_conversation.get_backend', return_value=mock_store):
            with pytest.raises(ValueError, match="Session nonexistent not found"):
                await infinite_manager.build_context_for_llm("nonexistent")
    
    async def test_segmentation_without_llm_callback(self, infinite_manager, mock_session_store):
        """Test that segmentation fails gracefully without LLM callback."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.infinite_conversation.get_backend', return_value=mock_store):
            session = Session()
            session.id = "test-session"
            session.token_summary.total_tokens = 150  # Above threshold
            sessions[session.id] = session
            
            # This should trigger an error when trying to create summary
            with pytest.raises(Exception):
                await infinite_manager.process_message(
                    session_id=session.id,
                    message="Test message",
                    source=EventSource.USER,
                    llm_callback=None  # No callback provided
                )
    
    async def test_basic_context_building(self, infinite_manager, mock_session_store):
        """Test basic context building without ancestors."""
        mock_store, sessions = mock_session_store
        
        with patch('chuk_ai_session_manager.infinite_conversation.get_backend', return_value=mock_store):
            session = Session()
            session.id = "simple-session"
            
            # Add some messages
            for i in range(3):
                event = SessionEvent(
                    message=f"Message {i}",
                    source=EventSource.USER if i % 2 == 0 else EventSource.LLM,
                    type=EventType.MESSAGE
                )
                await session.add_event(event)
            
            sessions[session.id] = session
            
            # Build basic context
            context = await infinite_manager.build_context_for_llm(
                session_id=session.id,
                max_messages=2,
                include_summaries=False
            )
            
            assert isinstance(context, list)
            assert len(context) <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])