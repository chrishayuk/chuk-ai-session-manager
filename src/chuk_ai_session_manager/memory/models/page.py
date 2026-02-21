# chuk_ai_session_manager/memory/models/page.py
"""Core page models: MemoryPage and PageTableEntry."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from chuk_ai_session_manager.memory.models.enums import (
    Affinity,
    CompressionLevel,
    Modality,
    PageType,
    StorageTier,
)

# =============================================================================
# Core Page Models
# =============================================================================


class MemoryPage(BaseModel):
    """
    Atomic unit of content in the virtual memory system.

    A page represents any piece of content (text, image, audio, video, structured)
    with identity, versioning, and multi-resolution representations.

    This is the RIGHT abstraction boundary because it enables:
    - Cross-modal coherence
    - Versioning and dirty tracking
    - Copy-on-write
    - Checkpoint consistency
    """

    # Identity
    page_id: str = Field(..., description="Unique identifier for this page")
    session_id: str | None = Field(default=None, description="Owning session")

    # Content type
    modality: Modality = Field(..., description="Content modality")

    # Page type (critical for eviction/compression decisions)
    page_type: PageType = Field(
        default=PageType.TRANSCRIPT,
        description="Page type determines eviction/compression behavior",
    )

    # Provenance: what pages justify this one (for claims, summaries)
    provenance: list[str] = Field(
        default_factory=list,
        description="page_ids that this page derives from (for claims/summaries)",
    )

    # Representation linking (for compression chain)
    represents: str | None = Field(
        default=None,
        description="page_id this is a compressed version of",
    )
    representation_level: CompressionLevel = Field(
        default=CompressionLevel.FULL,
        description="Compression level this representation is at",
    )

    # Location
    storage_tier: StorageTier = Field(default=StorageTier.L1, description="Current storage tier")
    artifact_id: str | None = Field(default=None, description="Reference to chuk-artifacts storage")

    # Content (when loaded into L0/L1)
    content: Any | None = Field(default=None, description="Actual content when in working set")
    compression_level: CompressionLevel = Field(default=CompressionLevel.FULL, description="Current compression level")

    # Multi-resolution representations
    # Maps compression level -> artifact_id for stored representations
    representations: dict[CompressionLevel, str] = Field(
        default_factory=dict, description="artifact_id for each compression level"
    )

    # Size tracking
    size_bytes: int = Field(default=0, description="Size in bytes")
    size_tokens: int | None = Field(default=None, description="Estimated token count (for text/transcript)")

    # Access tracking (for LRU/eviction)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = Field(default=0, description="Number of times accessed")

    # Importance (affects eviction priority)
    # Claims default to higher importance
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Importance score for eviction decisions",
    )

    # State tracking
    dirty: bool = Field(default=False, description="Has been modified since last flush")
    pinned: bool = Field(default=False, description="Pinned pages are never evicted")

    # Lineage (legacy, use provenance/represents instead)
    parent_page_id: str | None = Field(default=None, description="Parent page if derived (e.g., summary of original)")

    # Modality-specific metadata
    mime_type: str | None = Field(default=None)
    duration_seconds: float | None = Field(default=None, description="Duration for audio/video")
    dimensions: tuple[int, int] | None = Field(default=None, description="Width x height for image/video")
    transcript: str | None = Field(default=None, description="Transcript for audio/video (L1 representation)")
    caption: str | None = Field(default=None, description="Caption for image (L2 representation)")

    # Custom metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    def mark_accessed(self) -> None:
        """Update access tracking."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1

    def mark_dirty(self) -> None:
        """Mark page as modified."""
        self.dirty = True

    def mark_clean(self) -> None:
        """Mark page as flushed/clean."""
        self.dirty = False

    def estimate_tokens(self) -> int:
        """Estimate token count for current content."""
        if self.size_tokens is not None:
            return self.size_tokens

        if self.content is None:
            return 0

        if isinstance(self.content, str):
            # Rough estimate: 4 chars per token
            return len(self.content) // 4
        elif isinstance(self.content, dict):
            import json

            return len(json.dumps(self.content)) // 4
        else:
            return self.size_bytes // 4 if self.size_bytes > 0 else 100


class PageTableEntry(BaseModel):
    """
    Metadata entry for a page in the page table.

    Tracks location, state, and access patterns without holding content.
    This is what the PageTable stores for each page.
    """

    page_id: str

    # Location
    tier: StorageTier
    artifact_id: str | None = None
    compression_level: CompressionLevel = CompressionLevel.FULL

    # Page type (for eviction decisions)
    page_type: PageType = Field(
        default=PageType.TRANSCRIPT,
        description="Page type for eviction/compression decisions",
    )

    # Provenance (for tracing back to source)
    provenance: list[str] = Field(
        default_factory=list,
        description="page_ids this page derives from",
    )

    # State
    dirty: bool = Field(default=False, description="Modified since last flush")
    pinned: bool = Field(default=False, description="Pinned pages are never evicted")
    last_flushed: datetime | None = Field(default=None)

    # Access tracking
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = Field(default=0)

    # Size
    size_tokens: int | None = None

    # Modality (for filtering)
    modality: Modality = Modality.TEXT

    # Locality hints (for NUMA awareness)
    affinity: Affinity = Field(default=Affinity.LOCAL, description="Locality hint for distributed storage")

    def mark_accessed(self) -> None:
        """Update access tracking."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1

    @property
    def eviction_priority(self) -> float:
        """
        Calculate eviction priority based on page type.

        Lower value = less likely to evict.
        """
        type_weights = {
            PageType.CLAIM: 0.1,  # Very low - claims are precious
            PageType.INDEX: 0.2,  # Very low - indexes are needed for search
            PageType.PROCEDURE: 0.3,  # Low - procedures help tool usage
            PageType.SUMMARY: 0.4,  # Low - rebuildable but useful
            PageType.ARTIFACT: 0.6,  # Normal
            PageType.TRANSCRIPT: 0.7,  # Normal
        }
        base = type_weights.get(self.page_type, 0.5)
        # Pinned pages get 0 priority (never evict)
        if self.pinned:
            return 0.0
        return base
