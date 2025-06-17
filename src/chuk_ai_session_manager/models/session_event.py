# chuk_ai_session_manager/models/session_event.py
"""
Session event model for the chuk session manager with improved async support.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, Generic, Optional, TypeVar, Union
from uuid import uuid4
from pydantic import BaseModel, Field, ConfigDict

# session manager
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.token_usage import TokenUsage

# Generic type for event message content
MessageT = TypeVar('MessageT')

class SessionEvent(BaseModel, Generic[MessageT]):
    """An event in a session."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    message: Optional[MessageT] = None
    task_id: Optional[str] = None
    type: EventType = EventType.MESSAGE
    source: EventSource = EventSource.LLM
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Field for token usage tracking
    token_usage: Optional[TokenUsage] = None
    
    @classmethod
    async def create_with_tokens(
        cls,
        message: MessageT,
        prompt: str,
        completion: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        source: EventSource = EventSource.LLM,
        type: EventType = EventType.MESSAGE,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SessionEvent[MessageT]:
        """
        Create a session event with automatic token counting asynchronously.
        
        Args:
            message: The message content
            prompt: The prompt text used (for token counting)
            completion: The completion text (for token counting)
            model: The model used for this interaction
            source: The source of this event
            type: The type of this event
            task_id: Optional task ID this event is associated with
            metadata: Optional additional metadata
            
        Returns:
            A new SessionEvent with token usage information
        """
        # Use the async method of TokenUsage
        token_usage = await TokenUsage.from_text(prompt, completion, model)
        
        # Create the event
        event = cls(
            message=message,
            task_id=task_id,
            type=type,
            source=source,
            metadata=metadata or {},
            token_usage=token_usage
        )
        
        return event
    
    async def update_token_usage(
        self, 
        prompt: Optional[str] = None, 
        completion: Optional[str] = None,
        model: Optional[str] = None
    ) -> None:
        """
        Update token usage information for this event.
        
        Args:
            prompt: The prompt text to count tokens for
            completion: The completion text to count tokens for
            model: The model to use for counting tokens
        """
        # If we don't have token_usage yet, create it
        if self.token_usage is None:
            self.token_usage = TokenUsage(model=model or "")
        
        # If model is provided, update it
        if model and not self.token_usage.model:
            self.token_usage.model = model
            
        # Calculate tokens if text is provided
        if prompt:
            # Use async method for token counting
            prompt_tokens = await TokenUsage.count_tokens(prompt, self.token_usage.model)
            self.token_usage.prompt_tokens = prompt_tokens
            
        if completion:
            # Use async method for token counting
            completion_tokens = await TokenUsage.count_tokens(completion, self.token_usage.model)
            self.token_usage.completion_tokens = completion_tokens
            
        # Recalculate totals
        self.token_usage.total_tokens = self.token_usage.prompt_tokens + self.token_usage.completion_tokens
        if self.token_usage.model:
            # Use async method for cost calculation
            self.token_usage.estimated_cost_usd = await self.token_usage.calculate_cost()
            
    # Metadata async methods with clean names
    async def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a metadata value.
        
        Args:
            key: The metadata key to retrieve
            default: Default value to return if key not found
            
        Returns:
            The metadata value or default if not found
        """
        return self.metadata.get(key, default)

    async def set_metadata(self, key: str, value: Any) -> None:
        """Set a metadata value.
        
        Args:
            key: The metadata key to set
            value: The value to set
        """
        self.metadata[key] = value
        
    async def has_metadata(self, key: str) -> bool:
        """Check if a metadata key exists.
        
        Args:
            key: The metadata key to check
            
        Returns:
            True if the key exists in metadata
        """
        return key in self.metadata
        
    async def remove_metadata(self, key: str) -> None:
        """Remove a metadata key-value pair.
        
        Args:
            key: The metadata key to remove
        """
        if key in self.metadata:
            del self.metadata[key]
    
    # Alternative async method for updating metadata for backward compatibility
    async def update_metadata(self, key: str, value: Any) -> None:
        """Update a metadata value (alias for set_metadata).
        
        Args:
            key: The metadata key to set
            value: The value to set
        """
        await self.set_metadata(key, value)