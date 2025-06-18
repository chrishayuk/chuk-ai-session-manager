# chuk_ai_session_manager/infinite_conversation.py
"""
InfiniteConversationManager for handling conversations that exceed token limits.

This module provides support for managing conversations that span multiple
session segments, with automatic summarization and context building.
"""
from __future__ import annotations
from enum import Enum
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
import logging

from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.session_storage import get_backend, ChukSessionsStore

# Type for LLM function callbacks
LLMCallbackAsync = Callable[[List[Dict[str, str]], str], Any]

logger = logging.getLogger(__name__)


class SummarizationStrategy(str, Enum):
    """Different strategies for summarizing conversation segments."""
    BASIC = "basic"               # General overview of the conversation
    KEY_POINTS = "key_points"     # Focus on key information points
    TOPIC_BASED = "topic_based"   # Organize by topics discussed
    QUERY_FOCUSED = "query_focused"  # Focus on user's questions


class InfiniteConversationManager:
    """
    Manages conversations that can theoretically be infinite in length.
    
    This manager automatically segments conversations that exceed token limits
    by creating a chain of sessions with summaries that provide context.
    """
    
    def __init__(
        self,
        token_threshold: int = 3000,
        max_turns_per_segment: int = 20,
        summarization_strategy: SummarizationStrategy = SummarizationStrategy.BASIC
    ):
        """
        Initialize the infinite conversation manager.
        
        Args:
            token_threshold: Maximum tokens before creating a new segment
            max_turns_per_segment: Maximum conversation turns per segment
            summarization_strategy: Strategy to use for summarization
        """
        self.token_threshold = token_threshold
        self.max_turns_per_segment = max_turns_per_segment
        self.summarization_strategy = summarization_strategy
    
    async def process_message(
        self,
        session_id: str,
        message: str,
        source: EventSource,
        llm_callback: LLMCallbackAsync,
        model: str = "gpt-3.5-turbo"
    ) -> str:
        """
        Process a new message in the conversation.
        
        This method:
        1. Adds the message to the current session
        2. Checks if token threshold is exceeded
        3. If needed, creates a summary and starts a new session
        
        Args:
            session_id: ID of the current session
            message: The message content
            source: Source of the message (USER or LLM)
            llm_callback: Async callback for LLM calls
            model: The model to use for token counting
            
        Returns:
            The current session ID (may be a new one if threshold was exceeded)
        """
        # Get the store
        backend = get_backend()
        store = ChukSessionsStore(backend)
        
        # Get the current session
        session = await store.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Add the message to the session
        event = await SessionEvent.create_with_tokens(
            message=message,
            prompt=message if source == EventSource.USER else "",
            completion=message if source == EventSource.LLM else "",
            model=model,
            source=source,
            type=EventType.MESSAGE
        )
        await session.add_event_and_save(event)
        
        # Check if we've exceeded the token threshold
        if await self._should_create_new_segment(session):
            logger.info(f"Token threshold exceeded for session {session_id}. Creating new segment.")
            
            # Create a summary of the current session
            summary = await self._create_summary(session, llm_callback)
            
            # Add the summary to the current session
            summary_event = SessionEvent(
                message=summary,
                source=EventSource.SYSTEM,
                type=EventType.SUMMARY
            )
            await session.add_event_and_save(summary_event)
            
            # Create a new session with the current as parent
            new_session = await Session.create(parent_id=session_id)
            
            # Return the new session ID
            return new_session.id
        
        # No new segment needed, return the current session ID
        return session_id
    
    async def _should_create_new_segment(self, session: Session) -> bool:
        """
        Determine if we should create a new session segment.
        
        Args:
            session: The current session
            
        Returns:
            True if a new segment should be created
        """
        # Check token count
        if session.total_tokens >= self.token_threshold:
            return True
        
        # Check turn count
        message_events = [e for e in session.events if e.type == EventType.MESSAGE]
        if len(message_events) >= self.max_turns_per_segment:
            return True
        
        return False
    
    async def _create_summary(
        self,
        session: Session,
        llm_callback: LLMCallbackAsync
    ) -> str:
        """
        Create a summary of the session.
        
        Args:
            session: The session to summarize
            llm_callback: Async callback for LLM calls
            
        Returns:
            A summary string
        """
        # Get message events
        message_events = [e for e in session.events if e.type == EventType.MESSAGE]
        
        # Create a conversation history for the LLM
        messages = []
        
        # Add system prompt based on summarization strategy
        system_prompt = self._get_summarization_prompt()
        messages.append({"role": "system", "content": system_prompt})
        
        # Add the conversation history
        for event in message_events:
            role = "user" if event.source == EventSource.USER else "assistant"
            content = event.message
            messages.append({"role": role, "content": content})
        
        # Call the LLM to generate a summary
        summary = await llm_callback(messages)
        return summary
    
    def _get_summarization_prompt(self) -> str:
        """
        Get the prompt for summarization based on the selected strategy.
        
        Returns:
            A prompt string
        """
        if self.summarization_strategy == SummarizationStrategy.BASIC:
            return "Please provide a concise summary of this conversation. Focus on the main topic and key information exchanged."
            
        elif self.summarization_strategy == SummarizationStrategy.KEY_POINTS:
            return "Summarize this conversation by identifying and listing the key points discussed. Focus on the most important information exchanged."
            
        elif self.summarization_strategy == SummarizationStrategy.TOPIC_BASED:
            return "Create a summary of this conversation organized by topics discussed. Identify the main subject areas and the key points within each."
            
        elif self.summarization_strategy == SummarizationStrategy.QUERY_FOCUSED:
            return "Summarize this conversation by focusing on the user's main questions and the key answers provided. Prioritize what the user was seeking to learn."
            
        else:
            return "Please provide a brief summary of this conversation."
    
    async def build_context_for_llm(
        self,
        session_id: str,
        max_messages: int = 10,
        include_summaries: bool = True
    ) -> List[Dict[str, str]]:
        """
        Build context for an LLM call from the current session and its ancestors.
        
        Args:
            session_id: ID of the current session
            max_messages: Maximum number of recent messages to include
            include_summaries: Whether to include summaries from parent sessions
            
        Returns:
            A list of messages suitable for an LLM call
        """
        # Get the store
        backend = get_backend()
        store = ChukSessionsStore(backend)
        
        # Get the current session
        session = await store.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Initialize context
        context = []
        
        # Add summaries from ancestor sessions if requested
        if include_summaries:
            # Get all ancestors
            ancestors = await session.ancestors()
            
            # Get summaries from ancestors (most distant to most recent)
            summaries = []
            for ancestor in ancestors:
                summary_event = next(
                    (e for e in reversed(ancestor.events) if e.type == EventType.SUMMARY),
                    None
                )
                if summary_event:
                    summaries.append(summary_event.message)
            
            # If we have summaries, add them as a system message
            if summaries:
                context.append({
                    "role": "system",
                    "content": "Previous conversation context: " + " ".join(summaries)
                })
        
        # Get recent messages from the current session
        message_events = [e for e in session.events if e.type == EventType.MESSAGE]
        recent_messages = message_events[-max_messages:] if len(message_events) > max_messages else message_events
        
        # Add messages to context
        for event in recent_messages:
            role = "user" if event.source == EventSource.USER else "assistant"
            content = event.message
            context.append({"role": role, "content": content})
        
        return context
    
    async def get_session_chain(self, session_id: str) -> List[Session]:
        """
        Return sessions from root → … → current.

        The `Session.ancestors()` helper usually returns the chain in
        *reverse* (closest parent first).  Tests expect root-first order,
        so we reverse it and then append the current session.
        """
        backend = get_backend()
        store = ChukSessionsStore(backend)
        session = await store.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        ancestors = await session.ancestors()
        # ensure order root → … → parent
        ancestors = list(reversed(ancestors))
        return ancestors + [session]

    async def get_full_conversation_history(
        self,
        session_id: str
    ) -> List[Tuple[str, EventSource, str]]:
        """
        Get the full conversation history across all session segments.
        
        Args:
            session_id: ID of the current session
            
        Returns:
            A list of (role, source, content) tuples representing the conversation
        """
        # Get the session chain
        sessions = await self.get_session_chain(session_id)
        
        # Initialize history
        history = []
        
        # Process each session in the chain
        for session in sessions:
            # Get message events from this session
            message_events = [e for e in session.events if e.type == EventType.MESSAGE]
            
            # Add to history
            for event in message_events:
                role = "user" if event.source == EventSource.USER else "assistant"
                content = event.message
                history.append((role, event.source, content))
        
        return history