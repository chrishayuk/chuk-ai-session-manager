# chuk_ai_session_manager/memory/pack_cache.py
"""
Context Pack Cache - caches packed context to avoid re-packing on small incremental turns.

The TLB caches address translations. But in LLM terms, the expensive part is often:
- Selecting pages
- Choosing compression levels
- Rendering into VM:CONTEXT

This cache stores the packed context output to avoid recomputing it when:
- User says something short
- Assistant responds
- No pages changed

Cache is keyed by: (session_id, model_id, token_budget, working_set_hash)
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from datetime import datetime

from pydantic import BaseModel, Field

from .models import PackCacheStats

logger = logging.getLogger(__name__)


class PackedContext(BaseModel):
    """A cached packed context result."""

    # The packed content
    vm_context: str = Field(..., description="The VM:CONTEXT formatted content")
    vm_manifest_json: str = Field(..., description="The VM:MANIFEST_JSON content")

    # Metadata
    page_ids: list[str] = Field(default_factory=list, description="Pages included")
    tokens_used: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Cache key components (for debugging)
    session_id: str = Field(default="")
    model_id: str = Field(default="")
    token_budget: int = Field(default=0)
    working_set_hash: str = Field(default="")


class CacheEntry(BaseModel):
    """Internal cache entry with LRU tracking."""

    packed: PackedContext
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = Field(default=0)


class ContextPackCache:
    """
    Cache packed context to avoid re-packing on small incremental turns.
    Invalidate on working set changes.

    This drastically reduces overhead for "small incremental turns" -
    user says something short, assistant responds, no pages changed, reuse the pack.
    """

    def __init__(self, max_entries: int = 32):
        self.max_entries = max_entries
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "invalidations": 0,
        }

    def _make_key(
        self,
        session_id: str,
        model_id: str,
        token_budget: int,
        working_set_hash: str,
    ) -> str:
        """Create cache key from components."""
        key_str = f"{session_id}:{model_id}:{token_budget}:{working_set_hash}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    @staticmethod
    def compute_working_set_hash(page_ids: list[str], versions: dict[str, int] | None = None) -> str:
        """
        Compute a hash of the working set for cache key.

        Args:
            page_ids: List of page IDs in working set
            versions: Optional dict of page_id -> version number

        Returns:
            Hash string representing the working set state
        """
        # Sort for consistency
        sorted_ids = sorted(page_ids)

        parts = [f"{pid}:{versions.get(pid, 0)}" for pid in sorted_ids] if versions else sorted_ids

        content = "|".join(parts)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(
        self,
        session_id: str,
        model_id: str,
        token_budget: int,
        working_set_hash: str,
    ) -> PackedContext | None:
        """
        O(1) lookup for cached pack.

        Returns None if not found or expired.
        """
        key = self._make_key(session_id, model_id, token_budget, working_set_hash)

        if key not in self._cache:
            self._stats["misses"] += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)

        entry = self._cache[key]
        entry.last_accessed = datetime.utcnow()
        entry.access_count += 1

        self._stats["hits"] += 1
        return entry.packed

    def put(
        self,
        session_id: str,
        model_id: str,
        token_budget: int,
        working_set_hash: str,
        packed: PackedContext,
    ) -> None:
        """Store packed context, evict LRU if full."""
        key = self._make_key(session_id, model_id, token_budget, working_set_hash)

        # Update packed metadata
        packed.session_id = session_id
        packed.model_id = model_id
        packed.token_budget = token_budget
        packed.working_set_hash = working_set_hash

        # Evict if at capacity
        while len(self._cache) >= self.max_entries:
            # Remove oldest (first item)
            self._cache.popitem(last=False)
            self._stats["evictions"] += 1

        self._cache[key] = CacheEntry(packed=packed)

    def invalidate_session(self, session_id: str) -> int:
        """
        Invalidate all cached packs for session.

        Called when working set changes.
        Returns number of entries invalidated.
        """
        keys_to_remove = [key for key, entry in self._cache.items() if entry.packed.session_id == session_id]

        for key in keys_to_remove:
            del self._cache[key]
            self._stats["invalidations"] += 1

        return len(keys_to_remove)

    def invalidate_all(self) -> None:
        """Clear entire cache."""
        count = len(self._cache)
        self._cache.clear()
        self._stats["invalidations"] += count

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self._stats["hits"] + self._stats["misses"]
        if total == 0:
            return 0.0
        return self._stats["hits"] / total

    def get_stats(self) -> PackCacheStats:
        """Get cache statistics."""
        return PackCacheStats(
            hits=self._stats["hits"],
            misses=self._stats["misses"],
            evictions=self._stats["evictions"],
            invalidations=self._stats["invalidations"],
            size=len(self._cache),
            max_size=self.max_entries,
        )
