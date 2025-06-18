# chuk_ai_session_manager/api/simple_api.py
"""
Super simple developer API for session management with any LLM.

Usage:
    from chuk_ai_session_manager.simple_api import SessionManager
    
    # Start a conversation
    sm = SessionManager()
    
    # Track interactions easily
    await sm.user_says("Hello!")
    response = await sm.ai_responds("Hi there! How can I help?")
    
    # Get conversation history
    history = await sm.get_conversation()
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
    Super simple API for session management with any LLM.
    
    This class provides a dead-simple interface for tracking conversations
    while hiding all the complexity of the underlying session management.
    """
    
    def __init__(
        self, 
        session_id: Optional[str] = None,
        auto_save: bool = True,
        store: Optional[Any] = None
    ):
        """
        Initialize a session manager.
        
        Args:
            session_id: Use existing session or create new one
            auto_save: Automatically save after each operation
            store: Custom storage backend (defaults to CHUK Sessions)
        """
        self.auto_save = auto_save
        self._session: Optional[Session] = None
        self._session_id = session_id
        self._is_new_session = session_id is None  # Track if this is a new session
        
        # If no session_id provided, generate one now for convenience
        if not self._session_id:
            import uuid
            self._session_id = str(uuid.uuid4())
    
    async def _ensure_session(self) -> Session:
        """Ensure we have a session, creating one if needed."""
        if self._session is None:
            if self._is_new_session:
                # This is a new session, create it
                self._session = await Session.create()
                self._session_id = self._session.id
            else:
                # Try to load existing session
                backend = get_backend()
                store = ChukSessionsStore(backend)
                self._session = await store.get(self._session_id)
                if self._session is None:
                    raise ValueError(f"Session {self._session_id} not found")
        
        return self._session
    
    @property
    def session_id(self) -> str:
        """Get the current session ID."""
        if self._session:
            return self._session.id
        elif self._session_id:
            return self._session_id
        else:
            # This shouldn't happen with the new __init__, but just in case
            import uuid
            self._session_id = str(uuid.uuid4())
            return self._session_id
    
    async def user_says(
        self, 
        message: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Track a user message.
        
        Args:
            message: What the user said
            metadata: Optional metadata to attach
            
        Returns:
            The session ID for this conversation
        """
        session = await self._ensure_session()
        
        event = await SessionEvent.create_with_tokens(
            message=message,
            prompt=message,
            model="gpt-4o-mini",  # Default model for token counting
            source=EventSource.USER,
            type=EventType.MESSAGE
        )
        
        # Add metadata if provided
        if metadata:
            for key, value in metadata.items():
                await event.set_metadata(key, value)
        
        if self.auto_save:
            await session.add_event_and_save(event)
        else:
            await session.add_event(event)
        
        return session.id
    
    async def ai_responds(
        self, 
        response: str,
        model: str = "unknown",
        provider: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Track an AI response.
        
        Args:
            response: The AI's response
            model: Model name (e.g., "gpt-4o-mini")
            provider: Provider name (e.g., "openai")
            metadata: Optional metadata to attach
            
        Returns:
            The session ID for this conversation
        """
        session = await self._ensure_session()
        
        full_metadata = {
            "model": model,
            "provider": provider,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {})
        }
        
        event = await SessionEvent.create_with_tokens(
            message=response,
            prompt="",  # No prompt for AI response
            completion=response,
            model=model,
            source=EventSource.LLM,
            type=EventType.MESSAGE
        )
        
        # Add metadata
        for key, value in full_metadata.items():
            await event.set_metadata(key, value)
        
        if self.auto_save:
            await session.add_event_and_save(event)
        else:
            await session.add_event(event)
        
        return session.id
    
    async def tool_called(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Track a tool call.
        
        Args:
            tool_name: Name of the tool that was called
            arguments: Arguments passed to the tool
            result: Result from the tool
            error: Error message if tool failed
            metadata: Optional metadata to attach
            
        Returns:
            The session ID for this conversation
        """
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
        
        # Add metadata if provided
        if metadata:
            for key, value in metadata.items():
                await event.set_metadata(key, value)
        
        if self.auto_save:
            await session.add_event_and_save(event)
        else:
            await session.add_event(event)
        
        return session.id
    
    async def get_conversation(self, include_metadata: bool = False) -> List[Dict[str, Any]]:
        """
        Get the conversation history in a simple format.
        
        Args:
            include_metadata: Whether to include event metadata
            
        Returns:
            List of conversation turns as dicts
        """
        session = await self._ensure_session()
        
        conversation = []
        for event in session.events:
            if event.type == EventType.MESSAGE:
                turn = {
                    "role": "user" if event.source == EventSource.USER else "assistant",
                    "content": event.message,
                    "timestamp": event.timestamp.isoformat()
                }
                
                if include_metadata and event.metadata:
                    turn["metadata"] = event.metadata
                
                conversation.append(turn)
        
        return conversation
    
    async def get_tools_used(self) -> List[Dict[str, Any]]:
        """
        Get all tools that were used in this conversation.
        
        Returns:
            List of tool usage information
        """
        session = await self._ensure_session()
        
        tools = []
        for event in session.events:
            if event.type == EventType.TOOL_CALL and isinstance(event.message, dict):
                tools.append({
                    "tool": event.message.get("tool", "unknown"),
                    "arguments": event.message.get("arguments", {}),
                    "result": event.message.get("result"),
                    "success": event.message.get("success", True),
                    "error": event.message.get("error"),
                    "timestamp": event.timestamp.isoformat()
                })
        
        return tools
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get conversation statistics.
        
        Returns:
            Dictionary with conversation stats
        """
        session = await self._ensure_session()
        
        user_messages = sum(1 for e in session.events 
                           if e.type == EventType.MESSAGE and e.source == EventSource.USER)
        ai_messages = sum(1 for e in session.events 
                         if e.type == EventType.MESSAGE and e.source == EventSource.LLM)
        tool_calls = sum(1 for e in session.events if e.type == EventType.TOOL_CALL)
        
        return {
            "session_id": session.id,
            "total_events": len(session.events),
            "user_messages": user_messages,
            "ai_messages": ai_messages,
            "tool_calls": tool_calls,
            "created_at": session.metadata.created_at.isoformat(),
            "last_update": session.last_update_time.isoformat(),
            "total_tokens": session.total_tokens,
            "estimated_cost": session.total_cost
        }
    
    async def save(self) -> None:
        """Manually save the session (if auto_save is False)."""
        if self._session:
            backend = get_backend()
            store = ChukSessionsStore(backend)
            await store.save(self._session)
    
    async def clear(self) -> None:
        """Clear the current conversation and start fresh."""
        self._session = await Session.create()
        self._session_id = self._session.id


# Convenience functions for even simpler usage
async def quick_conversation(
    user_message: str,
    ai_response: str,
    model: str = "unknown",
    provider: str = "unknown"
) -> str:
    """
    Super quick way to track a single conversation turn.
    
    Returns the session ID for further use.
    """
    sm = SessionManager()
    await sm.user_says(user_message)
    session_id = await sm.ai_responds(ai_response, model=model, provider=provider)
    return session_id

async def track_llm_call(
    user_input: str,
    llm_function: Callable[[str], str],
    model: str = "unknown",
    provider: str = "unknown",
    session_manager: Optional[SessionManager] = None
) -> tuple[str, str]:
    """
    Track an LLM call automatically.
    
    Args:
        user_input: The user's input
        llm_function: Function that takes user input and returns AI response
        model: Model name
        provider: Provider name
        session_manager: Existing session manager (creates new if None)
    
    Returns:
        Tuple of (ai_response, session_id)
    """
    if session_manager is None:
        session_manager = SessionManager()
    
    # Track user input
    await session_manager.user_says(user_input)
    
    # Call the LLM
    if asyncio.iscoroutinefunction(llm_function):
        ai_response = await llm_function(user_input)
    else:
        ai_response = llm_function(user_input)
    
    # Track AI response
    session_id = await session_manager.ai_responds(
        ai_response, 
        model=model, 
        provider=provider
    )
    
    return ai_response, session_id