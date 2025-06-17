# chuk_ai_session_manager/session_run.py
"""
Session run model for the chuk session manager with improved async support.
"""
from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, List
from uuid import uuid4
from pydantic import BaseModel, Field, ConfigDict


class RunStatus(str, Enum):
    """Status of a session run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SessionRun(BaseModel):
    """A single execution or "run" within a session."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: Optional[datetime] = None
    status: RunStatus = RunStatus.PENDING
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tool_calls: List[str] = Field(default_factory=list)  # IDs of associated tool call events

    @classmethod
    async def create(cls, metadata: Optional[Dict[str, Any]] = None) -> SessionRun:
        """Create a new session run asynchronously."""
        return cls(
            status=RunStatus.PENDING,
            metadata=metadata or {}
        )

    async def mark_running(self) -> None:
        """Mark the run as started/running asynchronously."""
        self.status = RunStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)

    async def mark_completed(self) -> None:
        """Mark the run as completed successfully asynchronously."""
        self.status = RunStatus.COMPLETED
        self.ended_at = datetime.now(timezone.utc)

    async def mark_failed(self, reason: Optional[str] = None) -> None:
        """Mark the run as failed asynchronously."""
        self.status = RunStatus.FAILED
        self.ended_at = datetime.now(timezone.utc)
        if reason:
            await self.set_metadata("failure_reason", reason)

    async def mark_cancelled(self, reason: Optional[str] = None) -> None:
        """Mark the run as cancelled asynchronously."""
        self.status = RunStatus.CANCELLED
        self.ended_at = datetime.now(timezone.utc)
        if reason:
            await self.set_metadata("cancel_reason", reason)
        
    async def set_metadata(self, key: str, value: Any) -> None:
        """Set a metadata value asynchronously."""
        self.metadata[key] = value
        
    async def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a metadata value asynchronously."""
        return self.metadata.get(key, default)
        
    async def has_metadata(self, key: str) -> bool:
        """Check if a metadata key exists asynchronously."""
        return key in self.metadata
        
    async def remove_metadata(self, key: str) -> None:
        """Remove a metadata key-value pair asynchronously."""
        if key in self.metadata:
            del self.metadata[key]
        
    async def get_duration(self) -> Optional[float]:
        """Get the duration of the run in seconds asynchronously."""
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()
    
    async def add_tool_call(self, tool_call_id: str) -> None:
        """Associate a tool call event with this run asynchronously."""
        if tool_call_id not in self.tool_calls:
            self.tool_calls.append(tool_call_id)
            
    async def get_tool_calls(self, session: Any) -> List[Any]:  
        """Get all tool call events associated with this run asynchronously."""
        # We use Any type to avoid circular imports
        return [
            event for event in session.events 
            if event.id in self.tool_calls
        ]
        
    async def to_dict(self) -> Dict[str, Any]:
        """Convert the run to a dictionary asynchronously."""
        result = {
            "id": self.id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "metadata": self.metadata,
            "tool_calls": self.tool_calls
        }
        
        if self.ended_at:
            result["ended_at"] = self.ended_at.isoformat()
            result["duration"] = await self.get_duration()
            
        return result