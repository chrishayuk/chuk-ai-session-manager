# chuk_ai_session_manager/memory/manifest.py
"""
Manifest Builder for AI Virtual Memory.

The ManifestBuilder generates the VM:MANIFEST_JSON block - a machine-readable
index of all pages available to the model. This tells the model what exists
and where, enabling informed page_fault decisions.

Design principles:
- Machine-readable: Strict JSON for reliable parsing
- Discovery-focused: Hints help find relevant pages, not provide evidence
- Policy-aware: Includes fault limits and preferences
- Pydantic-native: All models are BaseModel subclasses
- No magic strings: Uses enums for all categorical values
"""

from typing import Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from .models import (
    ALL_COMPRESSION_LEVELS,
    CompressionLevel,
    MemoryPage,
    Modality,
    PageTableEntry,
    StorageTier,
)
from .page_table import PageTable


class WorkingSetEntry(BaseModel):
    """Entry for a page in the working set (already mapped)."""

    page_id: str
    modality: str
    level: int = Field(..., description="Compression level (0-3)")
    tokens_est: int
    importance: float = 0.5


class AvailablePageEntry(BaseModel):
    """Entry for a page available for loading (not yet mapped)."""

    page_id: str
    modality: str
    tier: str = Field(..., description="Current storage tier")
    levels: List[int] = Field(
        default_factory=lambda: list(ALL_COMPRESSION_LEVELS),
        description="Available compression levels",
    )
    hint: str = Field(default="", description="Discovery hint (NOT evidence)")


class ManifestPolicies(BaseModel):
    """Policies governing VM behavior."""

    faults_allowed: bool = True
    max_faults_per_turn: int = 2
    upgrade_budget_tokens: int = 4096
    prefer_levels: List[int] = Field(
        default_factory=lambda: [
            CompressionLevel.ABSTRACT.value,
            CompressionLevel.REDUCED.value,
            CompressionLevel.FULL.value,
        ],
        description="Preference order for compression levels",
    )


class VMManifest(BaseModel):
    """
    Complete VM manifest for inclusion in developer message.

    This is the machine-readable counterpart to VM:CONTEXT.
    """

    session_id: str
    working_set: List[WorkingSetEntry] = Field(default_factory=list)
    available_pages: List[AvailablePageEntry] = Field(default_factory=list)
    policies: ManifestPolicies = Field(default_factory=ManifestPolicies)

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=indent)

    def to_wrapped_json(self, indent: Optional[int] = None) -> str:
        """Serialize wrapped with VM:MANIFEST_JSON tags."""
        json_str = self.to_json(indent=indent)
        return f"<VM:MANIFEST_JSON>\n{json_str}\n</VM:MANIFEST_JSON>"


class ManifestBuilder(BaseModel):
    """
    Builds VM manifests from PageTable state.

    Usage:
        builder = ManifestBuilder()
        manifest = builder.build(
            session_id="sess_123",
            page_table=page_table,
            working_set_ids=["msg_1", "msg_2"],
            hint_generator=lambda e: f"message about {e.modality}"
        )
    """

    # Default policies
    default_policies: ManifestPolicies = Field(default_factory=ManifestPolicies)

    # Maximum available pages to include
    max_available_pages: int = Field(
        default=50, description="Limit available_pages to prevent manifest bloat"
    )

    def build(
        self,
        session_id: str,
        page_table: PageTable,
        working_set_ids: List[str],
        working_set_tokens: Optional[Dict[str, int]] = None,
        working_set_importance: Optional[Dict[str, float]] = None,
        hint_generator: Optional[Callable[[PageTableEntry], str]] = None,
        policies: Optional[ManifestPolicies] = None,
    ) -> VMManifest:
        """
        Build a complete VM manifest.

        Args:
            session_id: Session identifier
            page_table: PageTable with all page entries
            working_set_ids: Page IDs currently in working set (L0)
            working_set_tokens: Optional token counts per page
            working_set_importance: Optional importance scores per page
            hint_generator: Optional function(PageTableEntry) -> str for hints
            policies: Optional policy overrides

        Returns:
            VMManifest ready for serialization
        """
        working_set: List[WorkingSetEntry] = []
        available_pages: List[AvailablePageEntry] = []

        working_set_tokens = working_set_tokens or {}
        working_set_importance = working_set_importance or {}

        # Build working set entries
        for page_id in working_set_ids:
            entry = page_table.lookup(page_id)
            if entry:
                ws_entry = WorkingSetEntry(
                    page_id=page_id,
                    modality=entry.modality.value,
                    level=entry.compression_level,
                    tokens_est=working_set_tokens.get(
                        page_id, entry.size_tokens or 100
                    ),
                    importance=working_set_importance.get(page_id, 0.5),
                )
                working_set.append(ws_entry)

        # Build available pages (everything not in working set)
        available_count = 0
        for page_id, entry in page_table.entries.items():
            if page_id in working_set_ids:
                continue

            if available_count >= self.max_available_pages:
                break

            # Generate hint
            hint = ""
            if hint_generator:
                try:
                    hint = hint_generator(entry)
                except Exception:
                    hint = ""

            available_entry = AvailablePageEntry(
                page_id=page_id,
                modality=entry.modality.value,
                tier=entry.tier.value,
                levels=ALL_COMPRESSION_LEVELS,
                hint=hint,
            )
            available_pages.append(available_entry)
            available_count += 1

        return VMManifest(
            session_id=session_id,
            working_set=working_set,
            available_pages=available_pages,
            policies=policies or self.default_policies,
        )

    def build_from_pages(
        self,
        session_id: str,
        working_set_pages: List[MemoryPage],
        available_entries: List[PageTableEntry],
        hint_generator: Optional[Callable[[PageTableEntry], str]] = None,
        policies: Optional[ManifestPolicies] = None,
    ) -> VMManifest:
        """
        Build manifest directly from page objects (alternative API).

        Args:
            session_id: Session identifier
            working_set_pages: MemoryPage objects in working set
            available_entries: PageTableEntry objects for available pages
            hint_generator: Optional function(PageTableEntry) -> str
            policies: Optional policy overrides

        Returns:
            VMManifest
        """
        working_set: List[WorkingSetEntry] = []
        available_pages: List[AvailablePageEntry] = []

        # Build working set from pages
        for page in working_set_pages:
            ws_entry = WorkingSetEntry(
                page_id=page.page_id,
                modality=page.modality.value
                if hasattr(page.modality, "value")
                else str(page.modality),
                level=page.compression_level
                if isinstance(page.compression_level, int)
                else page.compression_level.value,
                tokens_est=page.size_tokens or page.estimate_tokens(),
                importance=page.importance,
            )
            working_set.append(ws_entry)

        # Build available pages
        working_set_ids = {p.page_id for p in working_set_pages}
        for entry in available_entries[: self.max_available_pages]:
            if entry.page_id in working_set_ids:
                continue

            hint = ""
            if hint_generator:
                try:
                    hint = hint_generator(entry)
                except Exception:
                    hint = ""

            available_entry = AvailablePageEntry(
                page_id=entry.page_id,
                modality=entry.modality.value
                if hasattr(entry.modality, "value")
                else str(entry.modality),
                tier=entry.tier.value
                if hasattr(entry.tier, "value")
                else str(entry.tier),
                levels=ALL_COMPRESSION_LEVELS,
                hint=hint,
            )
            available_pages.append(available_entry)

        return VMManifest(
            session_id=session_id,
            working_set=working_set,
            available_pages=available_pages,
            policies=policies or self.default_policies,
        )


# Hint type constants
class HintType:
    """Constants for hint types in manifest generation."""

    RECENT = "recent"
    STORED = "stored"
    ARCHIVED = "archived"
    SUMMARY = "summary"
    EXCERPT = "excerpt"
    CONTENT = "content"


def generate_simple_hint(entry: PageTableEntry) -> str:
    """
    Simple hint generator based on page metadata.

    This is a basic implementation - real systems would use
    summaries, embeddings, or other content-aware hints.
    """
    parts: List[str] = []

    # Modality
    if entry.modality != Modality.TEXT:
        parts.append(entry.modality.value)

    # Tier (indicates recency/importance)
    if entry.tier == StorageTier.L2:
        parts.append(HintType.RECENT)
    elif entry.tier == StorageTier.L3:
        parts.append(HintType.STORED)
    elif entry.tier == StorageTier.L4:
        parts.append(HintType.ARCHIVED)

    # Compression level
    if entry.compression_level == CompressionLevel.ABSTRACT:
        parts.append(HintType.SUMMARY)
    elif entry.compression_level == CompressionLevel.REDUCED:
        parts.append(HintType.EXCERPT)

    return " ".join(parts) if parts else HintType.CONTENT
