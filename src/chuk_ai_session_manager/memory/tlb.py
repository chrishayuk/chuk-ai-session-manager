# chuk_ai_session_manager/memory/tlb.py
"""
Translation Lookaside Buffer (TLB) for AI Virtual Memory.

The TLB is a small, fast cache for recently accessed page table entries.
It avoids the overhead of full PageTable lookups for hot pages.

Without a TLB, every page access requires a PageTable lookup plus potentially
a storage lookup. This becomes a bottleneck at scale - you end up measuring
metadata latency, not content latency.

The TLB provides O(1) lookups for recently accessed pages with LRU eviction.

Design principles:
- Pydantic-native: BaseModel subclass with proper validation
- No magic strings: Uses StorageTier enum
- Type-safe: Full type annotations throughout
"""

from collections import OrderedDict

from pydantic import BaseModel, Field, PrivateAttr

from .models import CombinedPageTableStats, PageTableEntry, TLBStats


class PageTLB(BaseModel):
    """
    Translation Lookaside Buffer - fast cache for page table entries.

    Uses LRU eviction to keep the most recently accessed entries.

    Typical usage:
        1. Check TLB first (O(1))
        2. If miss, check PageTable
        3. Insert result into TLB
    """

    max_entries: int = Field(default=512, description="Maximum entries to cache")

    # LRU cache: OrderedDict maintains insertion order, move_to_end on access
    _entries: OrderedDict = PrivateAttr(default_factory=OrderedDict)

    # Stats
    hits: int = Field(default=0)
    misses: int = Field(default=0)

    model_config = {"arbitrary_types_allowed": True}

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, page_id: str) -> bool:
        return page_id in self._entries

    def lookup(self, page_id: str) -> PageTableEntry | None:
        """
        Look up a page entry in the TLB.

        Returns the entry if found (TLB hit), None otherwise (TLB miss).
        Updates LRU order on hit.
        """
        if page_id in self._entries:
            # Move to end (most recently used)
            self._entries.move_to_end(page_id)
            self.hits += 1
            return self._entries[page_id]
        else:
            self.misses += 1
            return None

    def insert(self, entry: PageTableEntry) -> None:
        """
        Insert or update an entry in the TLB.

        If the TLB is full, evicts the least recently used entry.
        """
        page_id = entry.page_id

        # If already present, update and move to end
        if page_id in self._entries:
            self._entries[page_id] = entry
            self._entries.move_to_end(page_id)
            return

        # Check if we need to evict
        if len(self._entries) >= self.max_entries:
            self._evict_lru()

        # Insert new entry
        self._entries[page_id] = entry

    def _evict_lru(self) -> str | None:
        """
        Evict the least recently used entry.

        Returns the evicted page_id, or None if TLB was empty.
        """
        if not self._entries:
            return None

        # OrderedDict.popitem(last=False) removes first item (oldest)
        page_id, _ = self._entries.popitem(last=False)
        return page_id

    def invalidate(self, page_id: str) -> bool:
        """
        Remove an entry from the TLB.

        Call this when a page is modified, evicted, or deleted.
        Returns True if the entry was present and removed.
        """
        if page_id in self._entries:
            del self._entries[page_id]
            return True
        return False

    def invalidate_tier(self, tier: str) -> int:
        """
        Invalidate all entries in a specific tier.

        Useful when flushing a tier or during checkpoints.
        Returns the number of entries invalidated.
        """
        to_remove = [page_id for page_id, entry in self._entries.items() if entry.tier == tier]
        for page_id in to_remove:
            del self._entries[page_id]
        return len(to_remove)

    def flush(self) -> int:
        """
        Clear the entire TLB.

        Call this on context switches, checkpoints, or when consistency
        with the PageTable is required.
        Returns the number of entries cleared.
        """
        count = len(self._entries)
        self._entries.clear()
        return count

    def get_all(self) -> list[PageTableEntry]:
        """Get all cached entries (for debugging/inspection)."""
        return list(self._entries.values())

    @property
    def hit_rate(self) -> float:
        """Calculate the TLB hit rate."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def reset_stats(self) -> None:
        """Reset hit/miss counters."""
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> TLBStats:
        """Get TLB statistics."""
        return TLBStats(
            size=len(self._entries),
            max_size=self.max_entries,
            utilization=len(self._entries) / self.max_entries if self.max_entries > 0 else 0,
            hits=self.hits,
            misses=self.misses,
            hit_rate=self.hit_rate,
        )


class TLBWithPageTable:
    """
    Convenience wrapper that combines TLB with PageTable for lookups.

    Provides a unified interface that automatically checks TLB first,
    then falls back to PageTable, and keeps TLB updated.
    """

    def __init__(self, page_table, tlb: PageTLB | None = None):
        """
        Initialize with a PageTable and optional TLB.

        If no TLB is provided, creates one with default settings.
        """
        from .page_table import PageTable

        self.page_table: PageTable = page_table
        # Use 'is None' check because empty PageTLB has __len__=0 which is falsy
        self.tlb = tlb if tlb is not None else PageTLB()

    def lookup(self, page_id: str) -> PageTableEntry | None:
        """
        Look up a page, checking TLB first.

        Returns the entry if found, None otherwise.
        Automatically populates TLB on cache miss.
        """
        # Check TLB first
        entry = self.tlb.lookup(page_id)
        if entry is not None:
            return entry

        # TLB miss - check page table
        entry = self.page_table.lookup(page_id)
        if entry is not None:
            # Populate TLB for next time
            self.tlb.insert(entry)

        return entry

    def register(self, page) -> PageTableEntry:
        """Register a page in both PageTable and TLB."""
        entry = self.page_table.register(page)
        self.tlb.insert(entry)
        return entry

    def update_location(self, page_id: str, **kwargs) -> bool:
        """Update location in PageTable and invalidate TLB entry."""
        success = self.page_table.update_location(page_id, **kwargs)
        if success:
            # Invalidate stale TLB entry - will be refreshed on next lookup
            self.tlb.invalidate(page_id)
        return success

    def mark_dirty(self, page_id: str) -> bool:
        """Mark page dirty and invalidate TLB."""
        success = self.page_table.mark_dirty(page_id)
        if success:
            self.tlb.invalidate(page_id)
        return success

    def remove(self, page_id: str) -> PageTableEntry | None:
        """Remove from both PageTable and TLB."""
        self.tlb.invalidate(page_id)
        return self.page_table.remove(page_id)

    def get_stats(self) -> CombinedPageTableStats:
        """Get combined stats."""
        return CombinedPageTableStats(
            page_table=self.page_table.get_stats(),
            tlb=self.tlb.get_stats(),
        )
