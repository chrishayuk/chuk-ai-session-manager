# src/chuk_ai_session_manager/models/session_event.py
"""
Session event model for the chuk session manager with improved async support.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, Generic, Optional, TypeVar
from uuid import uuid4
from pydantic import BaseModel, Field, ConfigDict

from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.token_usage import TokenUsage

MessageT = TypeVar('MessageT')

class SessionEvent(BaseModel, Generic[MessageT]):
    """
    A single event within a session.
    
    Events track all interactions in a session including messages,
    tool calls, summaries, and other activities.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    message: MessageT
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Make source and type have defaults for backward compatibility with tests
    source: EventSource = Field(default=EventSource.SYSTEM)
    type: EventType = Field(default=EventType.MESSAGE)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parent_event_id: Optional[str] = None
    task_id: Optional[str] = None
    
    # Token tracking
    token_usage: Optional[TokenUsage] = None
    
    @classmethod
    async def create_with_tokens(
        cls,
        message: MessageT,
        prompt: str,
        completion: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        source: EventSource = EventSource.SYSTEM,
        type: EventType = EventType.MESSAGE,
        **kwargs
    ) -> SessionEvent[MessageT]:
        """
        Create a new SessionEvent with automatic token counting.
        
        Args:
            message: The message content
            prompt: The prompt text for token counting
            completion: Optional completion text for token counting
            model: The model to use for token counting
            source: The event source
            type: The event type
            **kwargs: Additional fields for the event
            
        Returns:
            A new SessionEvent instance with token usage calculated
        """
        # Create token usage
        token_usage = await TokenUsage.from_text(
            prompt=prompt,
            completion=completion,
            model=model
        )
        
        # Create the event
        event = cls(
            message=message,
            source=source,
            type=type,
            token_usage=token_usage,
            **kwargs
        )
        
        return event
    
    async def update_token_usage(
        self, 
        prompt: Optional[str] = None,
        completion: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        model: str = "gpt-3.5-turbo"
    ) -> None:
        """
        Update the token usage for this event.
        
        This method supports two modes:
        1. Pass prompt/completion strings to calculate tokens
        2. Pass prompt_tokens/completion_tokens directly
        
        Args:
            prompt: Optional prompt text to calculate tokens from
            completion: Optional completion text to calculate tokens from
            prompt_tokens: Optional number of prompt tokens (if already calculated)
            completion_tokens: Optional number of completion tokens (if already calculated)
            model: The model used for token calculation
        """
        if prompt is not None or completion is not None:
            # Calculate tokens from text
            self.token_usage = await TokenUsage.from_text(
                prompt=prompt or "",
                completion=completion,
                model=model
            )
        elif prompt_tokens is not None or completion_tokens is not None:
            # Use provided token counts
            if not self.token_usage:
                self.token_usage = TokenUsage(model=model)
            
            if prompt_tokens is not None:
                self.token_usage.prompt_tokens = prompt_tokens
            if completion_tokens is not None:
                self.token_usage.completion_tokens = completion_tokens
            
            # Update total
            self.token_usage.total_tokens = self.token_usage.prompt_tokens + self.token_usage.completion_tokens
            
            # Recalculate cost
            self.token_usage.estimated_cost_usd = self.token_usage._calculate_cost_sync()
    
    async def set_metadata(self, key: str, value: Any) -> None:
        """
        Set a metadata value asynchronously.
        
        Args:
            key: The metadata key
            value: The value to set
        """
        self.metadata[key] = value
    
    async def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get a metadata value asynchronously.
        
        Args:
            key: The metadata key
            default: Default value if key not found
            
        Returns:
            The metadata value or default
        """
        return self.metadata.get(key, default)
    
    async def has_metadata(self, key: str) -> bool:
        """
        Check if a metadata key exists asynchronously.
        
        Args:
            key: The metadata key to check
            
        Returns:
            True if the key exists
        """
        return key in self.metadata
    
    async def remove_metadata(self, key: str) -> None:
        """
        Remove a metadata key-value pair asynchronously.
        
        Args:
            key: The metadata key to remove
        """
        if key in self.metadata:
            del self.metadata[key]
    
    async def update_metadata(self, key: str, value: Any) -> None:
        """
        Update or add a metadata value asynchronously.
        
        Args:
            key: The metadata key
            value: The new value
        """
        self.metadata[key] = value
    
    async def merge_metadata(self, new_metadata: Dict[str, Any]) -> None:
        """
        Merge new metadata with existing metadata asynchronously.
        
        Args:
            new_metadata: Dictionary of metadata to merge
        """
        self.metadata.update(new_metadata)
    
    async def clear_metadata(self) -> None:
        """Clear all metadata asynchronously."""
        self.metadata.clear()
    
    async def calculate_tokens(self, model: str = "gpt-3.5-turbo") -> int:
        """
        Calculate tokens for this event's message asynchronously.
        
        Args:
            model: The model to use for token counting
            
        Returns:
            The number of tokens in the message
        """
        if self.token_usage:
            return self.token_usage.total_tokens
        
        # Calculate tokens from message
        message_str = str(self.message) if not isinstance(self.message, str) else self.message
        return await TokenUsage.count_tokens(message_str, model)
    
    def is_child_of(self, parent_event_id: str) -> bool:
        """
        Check if this event is a child of another event.
        
        Args:
            parent_event_id: The ID of the potential parent event
            
        Returns:
            True if this event is a child of the specified event
        """
        return self.parent_event_id == parent_event_id
    
    def is_part_of_task(self, task_id: str) -> bool:
        """
        Check if this event is part of a specific task.
        
        Args:
            task_id: The task ID to check
            
        Returns:
            True if this event is part of the specified task
        """
        return self.task_id == task_id
    
    async def to_dict(self) -> Dict[str, Any]:
        """
        Convert the event to a dictionary asynchronously.
        
        Returns:
            Dictionary representation of the event
        """
        result = {
            "id": self.id,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source.value,
            "type": self.type.value,
            "metadata": self.metadata
        }
        
        if self.parent_event_id:
            result["parent_event_id"] = self.parent_event_id
        
        if self.task_id:
            result["task_id"] = self.task_id
        
        if self.token_usage:
            result["token_usage"] = {
                "prompt_tokens": self.token_usage.prompt_tokens,
                "completion_tokens": self.token_usage.completion_tokens,
                "total_tokens": self.token_usage.total_tokens,
                "model": self.token_usage.model,
                "estimated_cost_usd": self.token_usage.estimated_cost_usd
            }
        
        return result