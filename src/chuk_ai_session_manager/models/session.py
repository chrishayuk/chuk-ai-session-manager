# src/chuk_ai_session_manager/models/session.py
"""
Session model for the chuk session manager with improved async support.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Generic, TypeVar, Union
from uuid import uuid4
from pydantic import BaseModel, Field, model_validator
import asyncio

# Import models that Session depends on
from chuk_ai_session_manager.models.session_metadata import SessionMetadata
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.token_usage import TokenUsage, TokenSummary
# Import SessionRun and RunStatus directly to avoid circular import
from chuk_ai_session_manager.models.session_run import SessionRun, RunStatus

MessageT = TypeVar('MessageT')

class Session(BaseModel, Generic[MessageT]):
    """A standalone conversation session with hierarchical support and async methods."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    metadata: SessionMetadata = Field(default_factory=SessionMetadata)

    parent_id: Optional[str] = None
    child_ids: List[str] = Field(default_factory=list)

    task_ids: List[str] = Field(default_factory=list)
    runs: List[SessionRun] = Field(default_factory=list)
    events: List[SessionEvent[MessageT]] = Field(default_factory=list)
    state: Dict[str, Any] = Field(default_factory=dict)
    
    # Token tracking
    token_summary: TokenSummary = Field(default_factory=TokenSummary)

    @model_validator(mode="after")
    def _sync_hierarchy(cls, model: Session) -> Session:
        """After creation, sync this session with its parent in the store.
        
        Note: This is synchronous for compatibility with Pydantic. 
        For async parent syncing, use async_init() after creation.
        """
        # This validator will be called during model creation,
        # but won't actually sync with storage - that requires async
        return model
    
    async def async_init(self) -> None:
        """
        Initialize async components of the session.
        
        Call this after creating a new session to properly set up
        parent-child relationships in the async storage.
        """
        if self.parent_id:
            # Import here to avoid circular import
            from chuk_ai_session_manager.session_storage import get_backend, ChukSessionsStore
            backend = get_backend()
            store = ChukSessionsStore(backend)
            parent = await store.get(self.parent_id)
            if parent and self.id not in parent.child_ids:
                parent.child_ids.append(self.id)
                await store.save(parent)

    @property
    def last_update_time(self) -> datetime:
        """Return timestamp of most recent event, or session creation."""
        if not self.events:
            return self.metadata.created_at
        return max(evt.timestamp for evt in self.events)

    @property
    def active_run(self) -> Optional[SessionRun]:
        """Return the currently running SessionRun, if any."""
        for run in reversed(self.runs):
            if run.status == RunStatus.RUNNING:
                return run
        return None
    
    @property
    def total_tokens(self) -> int:
        """Return the total number of tokens used in this session."""
        return self.token_summary.total_tokens
    
    @property
    def total_cost(self) -> float:
        """Return the total estimated cost of this session."""
        return self.token_summary.total_estimated_cost_usd

    async def add_child(self, child_id: str) -> None:
        """Add a child session ID and save the session."""
        if child_id not in self.child_ids:
            self.child_ids.append(child_id)
            # Save the updated session
            from chuk_ai_session_manager.session_storage import get_backend, ChukSessionsStore
            backend = get_backend()
            store = ChukSessionsStore(backend)
            await store.save(self)

    async def remove_child(self, child_id: str) -> None:
        """Remove a child session ID and save the session."""
        if child_id in self.child_ids:
            self.child_ids.remove(child_id)
            # Save the updated session
            from chuk_ai_session_manager.session_storage import get_backend, ChukSessionsStore
            backend = get_backend()
            store = ChukSessionsStore(backend)
            await store.save(self)

    async def ancestors(self) -> List[Session]:
        """Fetch ancestor sessions from store asynchronously."""
        result: List[Session] = []
        current = self.parent_id
        
        # Import here to avoid circular import
        from chuk_ai_session_manager.session_storage import get_backend, ChukSessionsStore
        backend = get_backend()
        store = ChukSessionsStore(backend)
        
        while current:
            parent = await store.get(current)
            if not parent:
                break
            result.append(parent)
            current = parent.parent_id
        return result

    async def descendants(self) -> List[Session]:
        """Fetch all descendant sessions from store in DFS order asynchronously."""
        result: List[Session] = []
        stack = list(self.child_ids)
        
        # Import here to avoid circular import
        from chuk_ai_session_manager.session_storage import get_backend, ChukSessionsStore
        backend = get_backend()
        store = ChukSessionsStore(backend)
        
        while stack:
            cid = stack.pop()
            child = await store.get(cid)
            if child:
                result.append(child)
                stack.extend(child.child_ids)
        return result
    
    async def add_event(self, event: SessionEvent[MessageT]) -> None:
        """
        Add an event to the session and update token tracking asynchronously.
        
        Args:
            event: The event to add
        """
        # Add the event
        self.events.append(event)
        
        # Update token summary if this event has token usage
        if event.token_usage:
            await self.token_summary.add_usage(event.token_usage)
    
    async def add_event_and_save(self, event: SessionEvent[MessageT]) -> None:
        """
        Add an event to the session, update token tracking, and save the session.
        
        Args:
            event: The event to add
        """
        # Add the event asynchronously
        await self.add_event(event)
        
        # Save the session
        from chuk_ai_session_manager.session_storage import get_backend, ChukSessionsStore
        backend = get_backend()
        store = ChukSessionsStore(backend)
        await store.save(self)
    
    async def get_token_usage_by_source(self) -> Dict[str, TokenSummary]:
        """
        Get token usage statistics grouped by event source asynchronously.
        
        Returns:
            A dictionary mapping event sources to token summaries
        """
        result: Dict[str, TokenSummary] = {}
        
        for event in self.events:
            if not event.token_usage:
                continue
                
            # Use the string value of the enum for the key
            source = event.source.value if hasattr(event.source, 'value') else str(event.source)
            if source not in result:
                result[source] = TokenSummary()
                
            await result[source].add_usage(event.token_usage)
            
        return result
    
    async def get_token_usage_by_run(self) -> Dict[str, TokenSummary]:
        """
        Get token usage statistics grouped by run asynchronously.
        
        Returns:
            A dictionary mapping run IDs to token summaries
        """
        result: Dict[str, TokenSummary] = {}
        
        # Add an entry for events without a run
        result["no_run"] = TokenSummary()
        
        for event in self.events:
            if not event.token_usage:
                continue
                
            run_id = event.task_id or "no_run"
            if run_id not in result:
                result[run_id] = TokenSummary()
                
            await result[run_id].add_usage(event.token_usage)
            
        return result
    
    async def count_message_tokens(
        self, 
        message: Union[str, Dict[str, Any]], 
        model: str = "gpt-3.5-turbo"
    ) -> int:
        """
        Count tokens in a message asynchronously.
        
        Args:
            message: The message to count tokens for (string or dict)
            model: The model to use for counting
            
        Returns:
            The number of tokens in the message
        """
        # If message is already a string, count directly
        if isinstance(message, str):
            return await TokenUsage.count_tokens(message, model)
        
        # If it's a dict (like OpenAI messages), extract content
        if isinstance(message, dict) and "content" in message:
            return await TokenUsage.count_tokens(message["content"], model)
            
        # If it's some other object, convert to string and count
        return await TokenUsage.count_tokens(str(message), model)
    
    async def set_state(self, key: str, value: Any) -> None:
        """
        Set a state value asynchronously.
        
        Args:
            key: The state key to set
            value: The value to set
        """
        self.state[key] = value
        
        # Auto-save if needed (could be added as an option)
        # from chuk_ai_session_manager.chuk_sessions_storage import get_backend, ChukSessionsStore
        # backend = get_backend()
        # store = ChukSessionsStore(backend)
        # await store.save(self)
    
    async def get_state(self, key: str, default: Any = None) -> Any:
        """
        Get a state value asynchronously.
        
        Args:
            key: The state key to retrieve
            default: Default value to return if key not found
            
        Returns:
            The state value or default if not found
        """
        return self.state.get(key, default)
    
    async def has_state(self, key: str) -> bool:
        """
        Check if a state key exists asynchronously.
        
        Args:
            key: The state key to check
            
        Returns:
            True if the key exists in state
        """
        return key in self.state
    
    async def remove_state(self, key: str) -> None:
        """
        Remove a state key-value pair asynchronously.
        
        Args:
            key: The state key to remove
        """
        if key in self.state:
            del self.state[key]
            
            # Auto-save if needed (could be added as an option)
            # from chuk_ai_session_manager.chuk_sessions_storage import get_backend, ChukSessionsStore
            # backend = get_backend()
            # store = ChukSessionsStore(backend)
            # await store.save(self)

    @classmethod
    async def create(cls, session_id: Optional[str] = None, parent_id: Optional[str] = None, **kwargs) -> Session:
        """
        Create a new session asynchronously, handling parent-child relationships.
        
        Args:
            session_id: Optional session ID to use (if not provided, generates a new one)
            parent_id: Optional parent session ID
            **kwargs: Additional arguments for Session initialization
            
        Returns:
            A new Session instance with parent-child relationships set up
        """
        # Allow passing a specific session ID
        if session_id:
            session = cls(id=session_id, parent_id=parent_id, **kwargs)
        else:
            session = cls(parent_id=parent_id, **kwargs)
        
        await session.async_init()
        
        # Save the new session
        from chuk_ai_session_manager.session_storage import get_backend, ChukSessionsStore
        backend = get_backend()
        store = ChukSessionsStore(backend)
        await store.save(session)
        
        return session