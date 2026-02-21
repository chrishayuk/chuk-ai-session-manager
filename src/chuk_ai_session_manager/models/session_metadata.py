# chuk_ai_session_manager/models/session_metadata.py
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class SessionMetadata(BaseModel):
    """Core metadata associated with a session."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Free-form properties for session-level identifiers and custom data
    properties: dict[str, Any] = Field(default_factory=dict)

    async def set_property(self, key: str, value: Any) -> None:
        """Add or update a custom metadata property asynchronously."""
        self.properties[key] = value
        self.updated_at = datetime.now(UTC)

    async def get_property(self, key: str) -> Any:
        """Retrieve a metadata property by key asynchronously."""
        return self.properties.get(key)

    async def update_timestamp(self) -> None:
        """Update the updated_at timestamp asynchronously."""
        self.updated_at = datetime.now(UTC)

    @classmethod
    async def create(cls, properties: dict[str, Any] | None = None) -> SessionMetadata:
        """Create a new SessionMetadata instance asynchronously."""
        now = datetime.now(UTC)
        return cls(created_at=now, updated_at=now, properties=properties or {})
