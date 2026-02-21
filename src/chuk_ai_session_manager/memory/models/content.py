# chuk_ai_session_manager/memory/models/content.py
"""Content models, tool definitions, and formatted output for the VM system."""

from typing import Any

from pydantic import BaseModel, Field

from chuk_ai_session_manager.memory.models.enums import (
    CompressionLevel,
    Modality,
    StorageTier,
    ToolType,
)

# =============================================================================
# Content Models (for tool results)
# =============================================================================


class TextContent(BaseModel):
    """Text content representation."""

    text: str = Field(default="")


class ImageContent(BaseModel):
    """Image content representation."""

    caption: str | None = Field(default=None)
    url: str | None = Field(default=None)
    base64: str | None = Field(default=None)
    embedding: list[float] | None = Field(default=None)


class AudioContent(BaseModel):
    """Audio content representation."""

    transcript: str | None = Field(default=None)
    timestamps: list[dict[str, Any]] | None = Field(default=None)
    duration_seconds: float | None = Field(default=None)


class VideoContent(BaseModel):
    """Video content representation."""

    scenes: list[dict[str, Any]] = Field(default_factory=list)
    transcript: str | None = Field(default=None)
    duration_seconds: float | None = Field(default=None)


class StructuredContent(BaseModel):
    """Structured data content representation."""

    data: dict[str, Any] = Field(default_factory=dict)
    schema_name: str | None = Field(default=None)


# Union type for all content types
PageContent = TextContent | ImageContent | AudioContent | VideoContent | StructuredContent


class PageMeta(BaseModel):
    """Metadata for a page in tool results."""

    source_tier: StorageTier | None = Field(default=None)
    mime_type: str | None = Field(default=None)
    size_bytes: int | None = Field(default=None)
    dimensions: list[int] | None = Field(default=None)
    duration_seconds: float | None = Field(default=None)
    latency_ms: float | None = Field(default=None)


class PageData(BaseModel):
    """Page data in tool result envelope."""

    page_id: str
    modality: Modality
    level: CompressionLevel
    tier: StorageTier
    content: PageContent
    meta: PageMeta = Field(default_factory=PageMeta)


class FaultEffects(BaseModel):
    """Side effects of a fault operation."""

    promoted_to_working_set: bool = Field(default=False)
    tokens_est: int = Field(default=0)
    evictions: list[str] | None = Field(default=None)


class SearchResultEntry(BaseModel):
    """Single entry in search results."""

    page_id: str
    modality: Modality
    tier: StorageTier
    levels: list[CompressionLevel] = Field(default_factory=list)
    hint: str = Field(default="")
    relevance: float = Field(default=0.0)


# =============================================================================
# Tool Definition Models
# =============================================================================


class ToolParameter(BaseModel):
    """Single parameter in a tool definition."""

    type: str
    description: str
    enum: list[str] | None = Field(default=None)
    minimum: int | None = Field(default=None)
    maximum: int | None = Field(default=None)
    default: Any | None = Field(default=None)


class ToolParameters(BaseModel):
    """Parameters schema for a tool."""

    type: str = Field(default="object")
    properties: dict[str, ToolParameter] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class ToolFunction(BaseModel):
    """Function definition within a tool."""

    name: str
    description: str
    parameters: ToolParameters


class ToolDefinition(BaseModel):
    """Complete tool definition for Chat Completions API."""

    type: ToolType = Field(default=ToolType.FUNCTION)
    function: ToolFunction


# =============================================================================
# Formatted Output Models
# =============================================================================


class FormattedPage(BaseModel):
    """Result of formatting a page for context."""

    content: str = Field(..., description="Formatted content string")
    tokens_est: int = Field(default=0, description="Estimated token count")
