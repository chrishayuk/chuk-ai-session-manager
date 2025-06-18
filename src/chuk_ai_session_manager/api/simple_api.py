# chuk_ai_session_manager/api/simple_api.py
"""
Unified SessionManager with built-in infinite context support.

Usage:
    from chuk_ai_session_manager import SessionManager
    
    # Regular session
    sm = SessionManager()
    
    # Infinite context session  
    sm = SessionManager(infinite_context=True)
    
    # Everything else is identical
    await sm.user_says("Hello!")
    await sm.ai_responds("Hi there!", model="gpt-4")
"""

from __future__ import annotations
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime

from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.session_storage import get_backend, ChukSessionsStore

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Unified session manager with built-in infinite context support.
    
    Automatically handles session segmentation, summarization, and context
    preservation when infinite_context=True is enabled.
    """
    
    def __init__(
        self, 
        session_id: Optional[str] = None,
        infinite_context: bool = False,
        token_threshold: int = 4000,
        max_turns_per_segment: int = 20
    ):
        """
        Initialize a session manager.
        
        Args:
            session_id: Use existing session or create new one
            infinite_context: Enable automatic infinite context handling
            token_threshold: Token limit before creating new session (infinite mode)
            max_turns_per_segment: Turn limit before creating new session (infinite mode)
        """
        self._session: Optional[Session] = None
        self._session_id = session_id
        self._is_new = session_id is None
        
        # Infinite context settings
        self._infinite_context = infinite_context
        self._token_threshold = token_threshold
        self._max_turns_per_segment = max_turns_per_segment
        
        # Infinite context state
        self._session_chain: List[str] = []
        self._full_conversation: List[Dict[str, Any]] = []
        self._total_segments = 1
    
    @property
    def session_id(self) -> str:
        """Get the current session ID."""
        if self._session:
            return self._session.id
        elif self._session_id:
            return self._session_id
        else:
            import uuid
            self._session_id = str(uuid.uuid4())
            return self._session_id
    
    @property
    def is_infinite(self) -> bool:
        """Check if infinite context is enabled."""
        return self._infinite_context
    
    async def _ensure_session(self) -> Session:
        """Ensure we have a session, creating one if needed."""
        if self._session is None:
            backend = get_backend()
            store = ChukSessionsStore(backend)
            
            if self._is_new:
                self._session = await Session.create()
                self._session_id = self._session.id
                
                # Always save new sessions immediately
                await store.save(self._session)
                
                # Initialize session chain for infinite context
                if self._infinite_context:
                    self._session_chain = [self._session_id]
            else:
                self._session = await store.get(self._session_id)
                if self._session is None:
                    raise ValueError(f"Session {self._session_id} not found")
        return self._session
    
    async def _should_create_new_segment(self) -> bool:
        """Check if we should create a new session segment."""
        if not self._infinite_context:
            return False
        
        session = await self._ensure_session()
        
        # Check token threshold
        if session.total_tokens >= self._token_threshold:
            return True
        
        # Check turn threshold
        message_events = [e for e in session.events if e.type == EventType.MESSAGE]
        if len(message_events) >= self._max_turns_per_segment:
            return True
        
        return False
    
    async def _create_summary(self) -> str:
        """Create a summary of the current session."""
        session = await self._ensure_session()
        message_events = [e for e in session.events if e.type == EventType.MESSAGE]
        
        # Simple summary generation
        user_messages = [e for e in message_events if e.source == EventSource.USER]
        
        topics = []
        for event in user_messages:
            content = str(event.message)
            if "?" in content:
                question = content.split("?")[0].strip()
                if len(question) > 10:
                    topics.append(question[:50])
        
        if topics:
            summary = f"User discussed: {'; '.join(topics[:3])}"
            if len(topics) > 3:
                summary += f" and {len(topics) - 3} other topics"
        else:
            summary = f"Conversation with {len(user_messages)} user messages and {len(message_events) - len(user_messages)} responses"
        
        return summary
    
    async def _create_new_segment(self) -> str:
        """Create a new session segment with summary."""
        # Create summary of current session
        summary = await self._create_summary()
        
        # Add summary to current session
        summary_event = SessionEvent(
            message=summary,
            source=EventSource.SYSTEM,
            type=EventType.SUMMARY
        )
        current_session = await self._ensure_session()
        await current_session.add_event_and_save(summary_event)
        
        # Create new session with current as parent
        new_session = await Session.create(parent_id=self._session_id)
        
        # Update our state
        old_session_id = self._session_id
        self._session_id = new_session.id
        self._session = new_session
        self._session_chain.append(self._session_id)
        self._total_segments += 1
        
        logger.info(f"Created new session segment: {old_session_id} -> {self._session_id}")
        return self._session_id
    
    async def user_says(self, message: str, **metadata) -> str:
        """
        Track a user message.
        
        Args:
            message: What the user said
            **metadata: Optional metadata to attach
            
        Returns:
            The current session ID (may change in infinite mode)
        """
        # Check for segmentation before adding message
        if await self._should_create_new_segment():
            await self._create_new_segment()
        
        session = await self._ensure_session()
        
        # Create and add the event
        event = await SessionEvent.create_with_tokens(
            message=message,
            prompt=message,
            model="gpt-4o-mini",
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        
        # Add metadata
        for key, value in metadata.items():
            await event.set_metadata(key, value)
        
        await session.add_event_and_save(event)
        
        # Track in full conversation for infinite context
        if self._infinite_context:
            self._full_conversation.append({
                "role": "user",
                "content": message,
                "timestamp": event.timestamp.isoformat(),
                "session_id": self._session_id
            })
        
        return self._session_id
    
    async def ai_responds(
        self, 
        response: str,
        model: str = "unknown",
        provider: str = "unknown",
        **metadata
    ) -> str:
        """
        Track an AI response.
        
        Args:
            response: The AI's response
            model: Model name
            provider: Provider name
            **metadata: Optional metadata
            
        Returns:
            The current session ID (may change in infinite mode)
        """
        # Check for segmentation before adding message
        if await self._should_create_new_segment():
            await self._create_new_segment()
        
        session = await self._ensure_session()
        
        # Create and add the event
        event = await SessionEvent.create_with_tokens(
            message=response,
            prompt="",
            completion=response,
            model=model,
            source=EventSource.LLM,
            type=EventType.MESSAGE
        )
        
        # Add metadata
        full_metadata = {
            "model": model,
            "provider": provider,
            "timestamp": datetime.now().isoformat(),
            **metadata
        }
        
        for key, value in full_metadata.items():
            await event.set_metadata(key, value)
        
        await session.add_event_and_save(event)
        
        # Track in full conversation for infinite context
        if self._infinite_context:
            self._full_conversation.append({
                "role": "assistant",
                "content": response,
                "timestamp": event.timestamp.isoformat(),
                "session_id": self._session_id,
                "model": model,
                "provider": provider
            })
        
        return self._session_id
    
    async def tool_used(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        error: Optional[str] = None,
        **metadata
    ) -> str:
        """Track a tool call."""
        session = await self._ensure_session()
        
        tool_message = {
            "tool": tool_name,
            "arguments": arguments,
            "result": result,
            "error": error,
            "success": error is None
        }
        
        event = SessionEvent(
            message=tool_message,
            source=EventSource.SYSTEM,
            type=EventType.TOOL_CALL
        )
        
        for key, value in metadata.items():
            await event.set_metadata(key, value)
        
        await session.add_event_and_save(event)
        return self._session_id
    
    async def get_conversation(self, include_all_segments: bool = None) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Args:
            include_all_segments: Include all segments (defaults to infinite_context setting)
            
        Returns:
            List of conversation turns
        """
        if include_all_segments is None:
            include_all_segments = self._infinite_context
        
        if self._infinite_context and include_all_segments:
            # Return full conversation across all segments
            return self._full_conversation.copy()
        else:
            # Return current session only
            session = await self._ensure_session()
            conversation = []
            for event in session.events:
                if event.type == EventType.MESSAGE:
                    turn = {
                        "role": "user" if event.source == EventSource.USER else "assistant",
                        "content": event.message,
                        "timestamp": event.timestamp.isoformat()
                    }
                    conversation.append(turn)
            
            return conversation
    
    async def get_session_chain(self) -> List[str]:
        """Get the chain of session IDs (infinite context only)."""
        if self._infinite_context:
            return self._session_chain.copy()
        else:
            return [self._session_id]
    
    async def get_stats(self, include_all_segments: bool = None) -> Dict[str, Any]:
        """
        Get conversation statistics.
        
        Args:
            include_all_segments: Include all segments (defaults to infinite_context setting)
            
        Returns:
            Dictionary with conversation stats
        """
        if include_all_segments is None:
            include_all_segments = self._infinite_context
        
        session = await self._ensure_session()
        
        if self._infinite_context and include_all_segments:
            # Calculate stats across all segments
            user_messages = len([t for t in self._full_conversation if t["role"] == "user"])
            ai_messages = len([t for t in self._full_conversation if t["role"] == "assistant"])
            
            # Get token/cost stats by loading all sessions in chain
            total_tokens = 0
            total_cost = 0.0
            total_events = 0
            
            backend = get_backend()
            store = ChukSessionsStore(backend)
            
            for session_id in self._session_chain:
                try:
                    sess = await store.get(session_id)
                    if sess:
                        total_tokens += sess.total_tokens
                        total_cost += sess.total_cost
                        total_events += len(sess.events)
                except Exception:
                    # Skip if can't load session
                    pass
            
            return {
                "session_id": self._session_id,
                "session_segments": self._total_segments,
                "session_chain": self._session_chain,
                "total_events": total_events,
                "user_messages": user_messages,
                "ai_messages": ai_messages,
                "tool_calls": 0,  # TODO: Track tools in full conversation
                "total_tokens": total_tokens,
                "estimated_cost": total_cost,
                "created_at": session.metadata.created_at.isoformat(),
                "last_update": session.last_update_time.isoformat(),
                "infinite_context": True
            }
        else:
            # Current session stats only
            user_messages = sum(1 for e in session.events 
                               if e.type == EventType.MESSAGE and e.source == EventSource.USER)
            ai_messages = sum(1 for e in session.events 
                             if e.type == EventType.MESSAGE and e.source == EventSource.LLM)
            tool_calls = sum(1 for e in session.events if e.type == EventType.TOOL_CALL)
            
            return {
                "session_id": session.id,
                "session_segments": 1,
                "total_events": len(session.events),
                "user_messages": user_messages,
                "ai_messages": ai_messages,
                "tool_calls": tool_calls,
                "total_tokens": session.total_tokens,
                "estimated_cost": session.total_cost,
                "created_at": session.metadata.created_at.isoformat(),
                "last_update": session.last_update_time.isoformat(),
                "infinite_context": self._infinite_context
            }


# Convenience functions remain the same but simpler
async def track_conversation(
    user_message: str,
    ai_response: str,
    model: str = "unknown",
    provider: str = "unknown",
    infinite_context: bool = False,
    token_threshold: int = 4000
) -> str:
    """Quick way to track a single conversation turn."""
    sm = SessionManager(
        infinite_context=infinite_context,
        token_threshold=token_threshold
    )
    await sm.user_says(user_message)
    session_id = await sm.ai_responds(ai_response, model=model, provider=provider)
    return session_id

async def track_llm_call(
    user_input: str,
    llm_function: Callable[[str], Union[str, Any]],
    model: str = "unknown",
    provider: str = "unknown",
    session_manager: Optional[SessionManager] = None,
    infinite_context: bool = False,
    token_threshold: int = 4000
) -> tuple[str, str]:
    """Track an LLM call automatically."""
    if session_manager is None:
        session_manager = SessionManager(
            infinite_context=infinite_context,
            token_threshold=token_threshold
        )
    
    await session_manager.user_says(user_input)
    
    if asyncio.iscoroutinefunction(llm_function):
        ai_response = await llm_function(user_input)
    else:
        ai_response = llm_function(user_input)
    
    # Handle different response formats
    if isinstance(ai_response, dict) and "choices" in ai_response:
        response_text = ai_response["choices"][0]["message"]["content"]
    elif hasattr(ai_response, "content"):
        response_text = ai_response.content
    else:
        response_text = str(ai_response)
    
    session_id = await session_manager.ai_responds(
        response_text, model=model, provider=provider
    )
    
    return response_text, session_id

async def quick_conversation(
    user_message: str,
    ai_response: str,
    infinite_context: bool = False
) -> Dict[str, Any]:
    """Quickest way to track a conversation and get basic stats."""
    session_id = await track_conversation(
        user_message, ai_response, infinite_context=infinite_context
    )
    sm = SessionManager(session_id, infinite_context=infinite_context)
    return await sm.get_stats()

async def track_infinite_conversation(
    user_message: str,
    ai_response: str,
    model: str = "unknown",
    provider: str = "unknown",
    token_threshold: int = 4000
) -> str:
    """Track a conversation with infinite context support."""
    return await track_conversation(
        user_message, ai_response, model=model, provider=provider,
        infinite_context=True, token_threshold=token_threshold
    )