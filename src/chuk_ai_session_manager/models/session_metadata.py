# chuk_ai_session_manager/models/session_metadata.py
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class SessionMetadata(BaseModel):
    """Core metadata associated with a session."""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Free-form properties for session-level identifiers and custom data
    properties: Dict[str, Any] = Field(default_factory=dict)
    
    async def set_property(self, key: str, value: Any) -> None:
        """Add or update a custom metadata property asynchronously."""
        self.properties[key] = value
        self.updated_at = datetime.now(timezone.utc)

    async def get_property(self, key: str) -> Any:
        """Retrieve a metadata property by key asynchronously."""
        return self.properties.get(key)
        
    async def update_timestamp(self) -> None:
        """Update the updated_at timestamp asynchronously."""
        self.updated_at = datetime.now(timezone.utc)
        
    @classmethod
    async def create(cls, properties: Optional[Dict[str, Any]] = None) -> SessionMetadata:
        """Create a new SessionMetadata instance asynchronously."""
        now = datetime.now(timezone.utc)
        return cls(
            created_at=now,
            updated_at=now,
            properties=properties or {}
        )