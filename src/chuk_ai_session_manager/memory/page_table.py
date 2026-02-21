# chuk_ai_session_manager/memory/page_table.py
"""
Page Table implementation for AI Virtual Memory.

The PageTable maps page IDs to their metadata (location, state, access patterns).
It's the core data structure for tracking all pages in the system.

Like an OS page table, it doesn't hold content - just metadata about where
content lives and its current state.

Design principles:
- Pydantic-native: BaseModel subclass with proper validation
- No magic strings: Uses StorageTier and Modality enums
- Type-safe: Full type annotations throughout
"""

from __future__ import annotations

import logging
from datetime import datetime

from pydantic import BaseModel, Field, PrivateAttr

from .models import (
    CompressionLevel,
    MemoryPage,
    Modality,
    PageTableEntry,
    PageTableStats,
    PageType,
    StorageTier,
)

logger = logging.getLogger(__name__)


class PageTable(BaseModel):
    """
    Maps page IDs to their current location and state.

    The page table is the source of truth for:
    - Where each page lives (which tier)
    - Whether it's dirty (needs flush)
    - Access patterns (for eviction decisions)
    """

    # Core mapping
    entries: dict[str, PageTableEntry] = Field(default_factory=dict)

    # Indexes for fast lookup - use PrivateAttr for internal state
    _by_tier: dict[StorageTier, set[str]] = PrivateAttr(default_factory=lambda: {t: set() for t in StorageTier})
    _by_modality: dict[Modality, set[str]] = PrivateAttr(default_factory=lambda: {m: set() for m in Modality})
    _dirty_pages: set[str] = PrivateAttr(default_factory=set)

    model_config = {"arbitrary_types_allowed": True}

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, page_id: str) -> bool:
        return page_id in self.entries

    def lookup(self, page_id: str) -> PageTableEntry | None:
        """
        Look up a page entry by ID.

        Returns None if page doesn't exist.
        Does NOT update access tracking (use mark_accessed for that).
        """
        return self.entries.get(page_id)

    def register(self, page: MemoryPage) -> PageTableEntry:
        """
        Register a new page in the table.

        Creates a PageTableEntry from the MemoryPage and adds it to indexes.
        """
        entry = PageTableEntry(
            page_id=page.page_id,
            tier=page.storage_tier,
            artifact_id=page.artifact_id,
            compression_level=page.compression_level,
            dirty=page.dirty,
            last_accessed=page.last_accessed,
            access_count=page.access_count,
            size_tokens=page.size_tokens or page.estimate_tokens(),
            modality=page.modality,
            # v0.8 fields
            page_type=page.page_type,
            provenance=page.provenance,
            pinned=page.pinned,
        )

        self._add_entry(entry)
        return entry

    def register_entry(self, entry: PageTableEntry) -> None:
        """Register an existing entry directly."""
        self._add_entry(entry)

    def _add_entry(self, entry: PageTableEntry) -> None:
        """Internal: add entry to table and indexes."""
        page_id = entry.page_id

        # Remove from old indexes if updating
        if page_id in self.entries:
            old_entry = self.entries[page_id]
            self._by_tier.get(old_entry.tier, set()).discard(page_id)
            self._by_modality.get(old_entry.modality, set()).discard(page_id)
            if old_entry.dirty:
                self._dirty_pages.discard(page_id)

        # Add to table
        self.entries[page_id] = entry

        # Update indexes
        if entry.tier not in self._by_tier:
            self._by_tier[entry.tier] = set()
        self._by_tier[entry.tier].add(page_id)

        if entry.modality not in self._by_modality:
            self._by_modality[entry.modality] = set()
        self._by_modality[entry.modality].add(page_id)

        if entry.dirty:
            self._dirty_pages.add(page_id)

    def remove(self, page_id: str) -> PageTableEntry | None:
        """
        Remove a page from the table.

        Returns the removed entry, or None if not found.
        """
        entry = self.entries.pop(page_id, None)
        if entry:
            self._by_tier.get(entry.tier, set()).discard(page_id)
            self._by_modality.get(entry.modality, set()).discard(page_id)
            self._dirty_pages.discard(page_id)
        return entry

    def update_location(
        self,
        page_id: str,
        tier: StorageTier,
        artifact_id: str | None = None,
        compression_level: CompressionLevel | None = None,
    ) -> bool:
        """
        Update a page's location (tier movement).

        Returns True if successful, False if page not found.
        """
        entry = self.entries.get(page_id)
        if not entry:
            return False

        # Update tier index
        self._by_tier.get(entry.tier, set()).discard(page_id)
        entry.tier = tier
        if tier not in self._by_tier:
            self._by_tier[tier] = set()
        self._by_tier[tier].add(page_id)

        # Update other fields if provided
        if artifact_id is not None:
            entry.artifact_id = artifact_id
        if compression_level is not None:
            entry.compression_level = compression_level

        return True

    def update_compression(
        self,
        page_id: str,
        compression_level: CompressionLevel,
    ) -> bool:
        """
        Update only the compression level for a page (no tier change).

        Returns True if successful, False if page not found.
        """
        entry = self.entries.get(page_id)
        if not entry:
            return False
        entry.compression_level = compression_level
        return True

    def mark_accessed(self, page_id: str) -> bool:
        """
        Mark a page as accessed (updates LRU tracking).

        Returns True if successful, False if page not found.
        """
        entry = self.entries.get(page_id)
        if not entry:
            return False

        entry.mark_accessed()
        return True

    def mark_dirty(self, page_id: str) -> bool:
        """
        Mark a page as dirty (modified).

        Returns True if successful, False if page not found.
        """
        entry = self.entries.get(page_id)
        if not entry:
            return False

        entry.dirty = True
        self._dirty_pages.add(page_id)
        return True

    def mark_clean(self, page_id: str) -> bool:
        """
        Mark a page as clean (flushed).

        Returns True if successful, False if page not found.
        """
        entry = self.entries.get(page_id)
        if not entry:
            return False

        entry.dirty = False
        entry.last_flushed = datetime.utcnow()
        self._dirty_pages.discard(page_id)
        return True

    def get_by_tier(self, tier: StorageTier) -> list[PageTableEntry]:
        """Get all entries in a specific tier."""
        page_ids = self._by_tier.get(tier, set())
        return [self.entries[pid] for pid in page_ids if pid in self.entries]

    def get_by_modality(self, modality: Modality) -> list[PageTableEntry]:
        """Get all entries of a specific modality."""
        page_ids = self._by_modality.get(modality, set())
        return [self.entries[pid] for pid in page_ids if pid in self.entries]

    def get_by_type(self, page_type: PageType) -> list[PageTableEntry]:
        """Get all entries of a specific page type."""
        return [e for e in self.entries.values() if e.page_type == page_type]

    def get_dirty_pages(self) -> list[PageTableEntry]:
        """Get all dirty (modified) pages."""
        return [self.entries[pid] for pid in self._dirty_pages if pid in self.entries]

    def get_working_set(self) -> list[PageTableEntry]:
        """Get all pages in L0 and L1 (the working set)."""
        l0 = self.get_by_tier(StorageTier.L0)
        l1 = self.get_by_tier(StorageTier.L1)
        return l0 + l1

    def get_eviction_candidates(
        self,
        tier: StorageTier = StorageTier.L1,
        limit: int = 10,
    ) -> list[PageTableEntry]:
        """
        Get pages that are candidates for eviction, sorted by LRU.

        Returns oldest-accessed pages first.
        """
        entries = self.get_by_tier(tier)
        # Sort by last_accessed (oldest first)
        entries.sort(key=lambda e: e.last_accessed)
        return entries[:limit]

    def get_stats(self) -> PageTableStats:
        """Get page table statistics."""
        return PageTableStats(
            total_pages=len(self.entries),
            dirty_pages=len(self._dirty_pages),
            pages_by_tier={t: len(self._by_tier.get(t, set())) for t in StorageTier},
            pages_by_modality={m: len(self._by_modality.get(m, set())) for m in Modality},
        )

    def get_total_tokens(self, tiers: list[StorageTier] | None = None) -> int:
        """
        Get total estimated tokens across specified tiers.

        If tiers is None, counts all pages.
        """
        total = 0
        entries_list = list(self.entries.values())

        if tiers:
            entries_list = [e for e in entries_list if e.tier in tiers]

        for entry in entries_list:
            if entry.size_tokens:
                total += entry.size_tokens

        return total
