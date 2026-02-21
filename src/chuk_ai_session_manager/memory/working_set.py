# chuk_ai_session_manager/memory/working_set.py
"""
Working Set Manager for AI Virtual Memory.

The WorkingSetManager tracks which pages are currently "hot" (in L0/L1)
and manages capacity constraints. It's the gatekeeper for what's in context.

Design principles:
- Pydantic-native: BaseModel subclass with proper validation
- Token-aware: Tracks token budget across modalities
- Eviction-ready: Provides candidates when under pressure
- Pinning support: Critical pages are never evicted
- Anti-thrash: Prevent evicting recently faulted pages
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, PrivateAttr

if TYPE_CHECKING:
    from .page_table import PageTable

from .models import (
    MemoryPage,
    Modality,
    StorageTier,
    TokenBudget,
    WorkingSetStats,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Pinned Set
# =============================================================================


class PinnedSet(BaseModel):
    """
    Pages that are never evicted from working set.
    Pinning prevents thrash on critical context.

    Auto-pinned by default:
    - System prompt page
    - Active goal/plan page
    - User preferences page
    - Current tool schemas
    - Last N turns (configurable, typically 2-4)
    """

    # Explicitly pinned pages
    pinned: set[str] = Field(default_factory=set)

    # Auto-pin configuration
    auto_pin_last_n_turns: int = Field(default=3, description="Auto-pin last N user+assistant turn pairs")
    auto_pin_system_prompt: bool = Field(default=True)
    auto_pin_claims: bool = Field(default=True, description="Auto-pin claim pages (high-value)")

    # Pages auto-pinned (tracked separately for debugging)
    auto_pinned: set[str] = Field(default_factory=set)

    def pin(self, page_id: str) -> None:
        """Explicitly pin a page."""
        self.pinned.add(page_id)

    def unpin(self, page_id: str) -> None:
        """Unpin a page (only affects explicit pins)."""
        self.pinned.discard(page_id)
        self.auto_pinned.discard(page_id)

    def is_pinned(self, page_id: str) -> bool:
        """Check if a page is pinned (explicitly or auto)."""
        return page_id in self.pinned or page_id in self.auto_pinned

    def auto_pin(self, page_id: str) -> None:
        """Auto-pin a page (e.g., recent turn, claim)."""
        self.auto_pinned.add(page_id)

    def clear_auto_pins(self) -> None:
        """Clear all auto-pins (before recalculating)."""
        self.auto_pinned.clear()

    def get_all_pinned(self) -> set[str]:
        """Get all pinned pages (explicit + auto)."""
        return self.pinned | self.auto_pinned

    def count(self) -> int:
        """Total number of pinned pages."""
        return len(self.get_all_pinned())


# =============================================================================
# Anti-Thrash Policy
# =============================================================================


class AntiThrashPolicy(BaseModel):
    """
    Prevent evicting pages that were just faulted in.

    OS working sets fail when you thrash. LLM working sets thrash
    when user toggles between topics.
    """

    # Recently-evicted pages get a "do not evict again" window
    eviction_cooldown_turns: int = Field(default=3)

    # Recently-faulted pages get temporary protection
    fault_protection_turns: int = Field(default=2)

    # Track eviction/fault history: page_id -> turn number
    _eviction_history: dict[str, int] = PrivateAttr(default_factory=dict)
    _fault_history: dict[str, int] = PrivateAttr(default_factory=dict)

    def record_eviction(self, page_id: str, turn: int) -> None:
        """Record that a page was evicted at this turn."""
        self._eviction_history[page_id] = turn

    def record_fault(self, page_id: str, turn: int) -> None:
        """Record that a page was faulted in at this turn."""
        self._fault_history[page_id] = turn

    def can_evict(self, page_id: str, current_turn: int) -> bool:
        """Check if page is in cooldown period."""
        # Check fault protection
        if page_id in self._fault_history:
            fault_turn = self._fault_history[page_id]
            if current_turn - fault_turn < self.fault_protection_turns:
                return False

        # Check eviction cooldown (avoid re-evicting)
        if page_id in self._eviction_history:
            evict_turn = self._eviction_history[page_id]
            if current_turn - evict_turn < self.eviction_cooldown_turns:
                return False

        return True

    def get_eviction_penalty(self, page_id: str, current_turn: int) -> float:
        """
        Higher penalty = less likely to evict.

        Recently faulted = high penalty (we just loaded it!)
        Recently evicted = high penalty (avoid re-evicting)
        """
        penalty = 0.0

        if page_id in self._fault_history:
            fault_turn = self._fault_history[page_id]
            turns_since = current_turn - fault_turn
            if turns_since < self.fault_protection_turns:
                # High penalty if recently faulted
                penalty += 1.0 - (turns_since / self.fault_protection_turns)

        if page_id in self._eviction_history:
            evict_turn = self._eviction_history[page_id]
            turns_since = current_turn - evict_turn
            if turns_since < self.eviction_cooldown_turns:
                # Moderate penalty if recently evicted
                penalty += 0.5 * (1.0 - (turns_since / self.eviction_cooldown_turns))

        return min(1.0, penalty)

    def cleanup_old_history(self, current_turn: int, max_age: int = 20) -> None:
        """Remove old history entries to prevent memory growth."""
        self._fault_history = {pid: turn for pid, turn in self._fault_history.items() if current_turn - turn <= max_age}
        self._eviction_history = {
            pid: turn for pid, turn in self._eviction_history.items() if current_turn - turn <= max_age
        }


class WorkingSetConfig(BaseModel):
    """Configuration for working set management."""

    # Token limits
    max_l0_tokens: int = Field(default=128_000, description="Maximum tokens in L0 (context window)")
    max_l1_pages: int = Field(default=100, description="Maximum pages in L1 cache")

    # Eviction thresholds
    eviction_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Trigger eviction when utilization exceeds this",
    )
    target_utilization: float = Field(default=0.70, ge=0.0, le=1.0, description="Target utilization after eviction")

    # Reserved tokens
    reserved_tokens: int = Field(default=4000, description="Reserved for system prompt, tools, etc.")


class WorkingSetManager(BaseModel):
    """
    Manages the working set (L0 + L1 pages).

    The working set is what's currently "hot" - either in the context window
    (L0) or in fast cache (L1). This manager tracks capacity, handles
    promotion/demotion, and identifies eviction candidates.

    Includes pinning (never evict critical pages) and anti-thrash
    (don't evict recently-faulted pages).
    """

    config: WorkingSetConfig = Field(default_factory=WorkingSetConfig)

    # Token budget tracking
    budget: TokenBudget = Field(default_factory=TokenBudget)

    # L0 pages (in context) - ordered by position
    l0_pages: list[str] = Field(default_factory=list, description="Page IDs in L0, ordered")

    # L1 pages (hot cache) - maps page_id -> MemoryPage
    l1_cache: dict[str, MemoryPage] = Field(default_factory=dict, description="Pages in L1 cache")

    # Page importance overrides
    importance_overrides: dict[str, float] = Field(default_factory=dict, description="Manual importance adjustments")

    # Pinned set - pages that are never evicted
    pinned_set: PinnedSet = Field(default_factory=PinnedSet)

    # Anti-thrash policy
    anti_thrash: AntiThrashPolicy = Field(default_factory=AntiThrashPolicy)

    # Current turn number (for anti-thrash)
    current_turn: int = Field(default=0)

    model_config = {"arbitrary_types_allowed": True}

    # Pluggable eviction policy (Protocol, not serializable)
    _eviction_policy: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Sync budget limits from config on construction."""
        if self.budget.total_limit != self.config.max_l0_tokens:
            self.budget = TokenBudget(
                total_limit=self.config.max_l0_tokens,
                reserved=self.config.reserved_tokens,
            )

    def set_eviction_policy(self, policy: Any) -> None:
        """Set a custom eviction policy (EvictionPolicy protocol)."""
        self._eviction_policy = policy

    def __len__(self) -> int:
        """Total pages in working set."""
        return len(self.l0_pages) + len(self.l1_cache)

    @property
    def l0_count(self) -> int:
        """Number of pages in L0."""
        return len(self.l0_pages)

    @property
    def l1_count(self) -> int:
        """Number of pages in L1."""
        return len(self.l1_cache)

    @property
    def utilization(self) -> float:
        """Current L0 token utilization (0-1)."""
        return self.budget.utilization

    @property
    def tokens_used(self) -> int:
        """Total tokens in L0."""
        return self.budget.used

    @property
    def tokens_available(self) -> int:
        """Available tokens for new content."""
        return self.budget.available

    def needs_eviction(self) -> bool:
        """Check if eviction is needed."""
        return self.utilization > self.config.eviction_threshold

    def can_fit(self, tokens: int) -> bool:
        """Check if additional tokens can fit in L0."""
        return self.budget.can_fit(tokens)

    def add_to_l0(self, page: MemoryPage) -> bool:
        """
        Add a page to L0 (context window).

        Returns True if successful, False if insufficient space.
        Does NOT automatically evict - call get_eviction_candidates first.
        """
        tokens = page.size_tokens or page.estimate_tokens()

        if not self.budget.can_fit(tokens):
            return False

        # Add to budget
        self.budget.add(tokens, page.modality)

        # Add to L0 list (at end = most recent)
        if page.page_id not in self.l0_pages:
            self.l0_pages.append(page.page_id)

        # Remove from L1 if present (promoted to L0)
        self.l1_cache.pop(page.page_id, None)

        # Update page state
        page.storage_tier = StorageTier.L0
        page.mark_accessed()

        return True

    def add_to_l1(self, page: MemoryPage) -> bool:
        """
        Add a page to L1 (hot cache).

        Returns True if successful, False if L1 is full.
        """
        if len(self.l1_cache) >= self.config.max_l1_pages:
            return False

        # Remove from L0 if present (demoted to L1)
        if page.page_id in self.l0_pages:
            self.l0_pages.remove(page.page_id)
            tokens = page.size_tokens or page.estimate_tokens()
            self.budget.remove(tokens, page.modality)

        # Add to L1
        self.l1_cache[page.page_id] = page
        page.storage_tier = StorageTier.L1
        page.mark_accessed()

        return True

    def remove(self, page_id: str) -> MemoryPage | None:
        """
        Remove a page from the working set entirely.

        Returns the removed page, or None if not found.
        """
        # Check L1 first
        page = self.l1_cache.pop(page_id, None)

        # Check L0
        if page_id in self.l0_pages:
            self.l0_pages.remove(page_id)
            # We need the page to update budget - caller should handle this
            # For now, assume average tokens per modality

        return page

    def remove_from_l0(self, page_id: str, page: MemoryPage) -> bool:
        """
        Remove a specific page from L0, updating budget.

        Returns True if removed, False if not in L0.
        """
        if page_id not in self.l0_pages:
            return False

        self.l0_pages.remove(page_id)
        tokens = page.size_tokens or page.estimate_tokens()
        self.budget.remove(tokens, page.modality)
        return True

    def promote_to_l0(self, page: MemoryPage) -> bool:
        """
        Promote a page from L1 to L0.

        Returns True if successful.
        """
        if page.page_id not in self.l1_cache:
            return False

        return self.add_to_l0(page)

    def demote_to_l1(self, page: MemoryPage) -> bool:
        """
        Demote a page from L0 to L1.

        Returns True if successful.
        """
        if page.page_id not in self.l0_pages:
            return False

        return self.add_to_l1(page)

    def get_page(self, page_id: str) -> MemoryPage | None:
        """Get a page from L1 cache (L0 pages are tracked by ID only)."""
        return self.l1_cache.get(page_id)

    def update_page_tokens(
        self,
        page_id: str,
        old_tokens: int,
        new_tokens: int,
        modality: Modality = Modality.TEXT,
    ) -> None:
        """
        Adjust the token budget after an in-place page change (e.g. compression).

        Only affects L0 budget accounting. Removes old_tokens and adds new_tokens
        for the given modality.
        """
        if page_id not in self.l0_pages:
            return
        self.budget.remove(old_tokens, modality)
        self.budget.add(new_tokens, modality)

    def is_in_l0(self, page_id: str) -> bool:
        """Check if a page is in L0."""
        return page_id in self.l0_pages

    def is_in_l1(self, page_id: str) -> bool:
        """Check if a page is in L1."""
        return page_id in self.l1_cache

    def is_in_working_set(self, page_id: str) -> bool:
        """Check if a page is in the working set (L0 or L1)."""
        return self.is_in_l0(page_id) or self.is_in_l1(page_id)

    def get_eviction_candidates(
        self,
        tokens_needed: int = 0,
        from_tier: StorageTier = StorageTier.L0,
        page_table: PageTable | None = None,
    ) -> list[tuple[str, float]]:
        """
        Get pages that are candidates for eviction, scored by priority.

        Returns list of (page_id, eviction_score) tuples.
        Lower score = evict first.

        Scoring considers:
        - Pinning (pinned pages are excluded)
        - Anti-thrash (recently faulted pages get penalty)
        - Recency (LRU component)
        - Access frequency (LFU component)
        - Importance (user/system-assigned)
        - Position (older messages first)

        If a custom eviction policy is set via set_eviction_policy(),
        delegates scoring to that policy instead.
        """
        # Delegate to custom policy if set
        if self._eviction_policy is not None:
            from .eviction_policy import EvictionContext

            context = EvictionContext(
                current_turn=self.current_turn,
                l0_page_ids=list(self.l0_pages),
                l1_pages=dict(self.l1_cache),
                pinned_page_ids=self.pinned_set.get_all_pinned(),
                importance_overrides=dict(self.importance_overrides),
                anti_thrash=self.anti_thrash,
                page_table=page_table,
            )
            candidates = self._eviction_policy.score_candidates(context, from_tier, tokens_needed)
            return [(c.page_id, c.score) for c in candidates]

        candidates = []

        if from_tier == StorageTier.L0:
            # For L0, we only have page IDs - score by position
            for i, page_id in enumerate(self.l0_pages):
                # Skip pinned pages
                if self.pinned_set.is_pinned(page_id):
                    continue

                # Check anti-thrash policy
                if not self.anti_thrash.can_evict(page_id, self.current_turn):
                    continue

                # Earlier position = lower score = evict first
                position_score = i / max(len(self.l0_pages), 1)
                importance = self.importance_overrides.get(page_id, 0.5)

                # Add anti-thrash penalty (higher penalty = higher score = less likely to evict)
                thrash_penalty = self.anti_thrash.get_eviction_penalty(page_id, self.current_turn)

                # Combine position, importance, and anti-thrash
                score = position_score * 0.4 + importance * 0.4 + thrash_penalty * 0.2
                candidates.append((page_id, score))
        else:
            # For L1, we have full pages with access tracking
            now = datetime.utcnow()
            for page_id, page in self.l1_cache.items():
                # Skip pinned pages
                if self.pinned_set.is_pinned(page_id) or page.pinned:
                    continue

                # Check anti-thrash policy
                if not self.anti_thrash.can_evict(page_id, self.current_turn):
                    continue

                # Recency score (seconds since access, normalized)
                age_seconds = (now - page.last_accessed).total_seconds()
                recency_score = 1.0 / (1.0 + age_seconds / 3600)  # Decay over hours

                # Frequency score (log scale)
                frequency_score = math.log1p(page.access_count) / 10.0
                frequency_score = min(1.0, frequency_score)

                # Importance (page type affects this)
                importance = self.importance_overrides.get(page_id, page.importance)

                # Add anti-thrash penalty
                thrash_penalty = self.anti_thrash.get_eviction_penalty(page_id, self.current_turn)

                # Combined score (higher = keep longer)
                score = recency_score * 0.3 + frequency_score * 0.2 + importance * 0.3 + thrash_penalty * 0.2
                candidates.append((page_id, score))

        # Sort by score (lowest first = evict first)
        candidates.sort(key=lambda x: x[1])

        return candidates

    def new_turn(self) -> None:
        """Advance to a new turn, updating anti-thrash tracking."""
        self.current_turn += 1
        # Cleanup old history periodically
        if self.current_turn % 10 == 0:
            self.anti_thrash.cleanup_old_history(self.current_turn)

    def pin_page(self, page_id: str) -> None:
        """Pin a page (will not be evicted)."""
        self.pinned_set.pin(page_id)

    def unpin_page(self, page_id: str) -> None:
        """Unpin a page."""
        self.pinned_set.unpin(page_id)

    def is_pinned(self, page_id: str) -> bool:
        """Check if a page is pinned."""
        return self.pinned_set.is_pinned(page_id)

    def record_fault(self, page_id: str) -> None:
        """Record a page fault for anti-thrash tracking."""
        self.anti_thrash.record_fault(page_id, self.current_turn)

    def record_eviction(self, page_id: str) -> None:
        """Record an eviction for anti-thrash tracking."""
        self.anti_thrash.record_eviction(page_id, self.current_turn)

    def calculate_eviction_target(self, tokens_needed: int = 0) -> int:
        """
        Calculate how many tokens to free to reach target utilization.

        Args:
            tokens_needed: Additional tokens we need to fit

        Returns:
            Number of tokens to evict
        """
        current = self.budget.used
        max_tokens = self.config.max_l0_tokens - self.config.reserved_tokens
        target = int(max_tokens * self.config.target_utilization)

        # Need to get below target AND fit new tokens
        required_free = current - target + tokens_needed

        return max(0, required_free)

    def set_importance(self, page_id: str, importance: float) -> None:
        """Set importance override for a page."""
        self.importance_overrides[page_id] = max(0.0, min(1.0, importance))

    def clear_importance(self, page_id: str) -> None:
        """Remove importance override for a page."""
        self.importance_overrides.pop(page_id, None)

    def get_l0_page_ids(self) -> list[str]:
        """Get all page IDs in L0, in order."""
        return list(self.l0_pages)

    def get_l1_pages(self) -> list[MemoryPage]:
        """Get all pages in L1."""
        return list(self.l1_cache.values())

    def get_stats(self) -> WorkingSetStats:
        """Get working set statistics."""
        return WorkingSetStats(
            l0_pages=len(self.l0_pages),
            l1_pages=len(self.l1_cache),
            total_pages=len(self),
            tokens_used=self.tokens_used,
            tokens_available=self.tokens_available,
            utilization=self.utilization,
            needs_eviction=self.needs_eviction(),
            tokens_by_modality=dict(self.budget.tokens_by_modality),
        )

    def clear(self) -> None:
        """Clear the entire working set."""
        self.l0_pages.clear()
        self.l1_cache.clear()
        self.budget = TokenBudget(
            total_limit=self.config.max_l0_tokens,
            reserved=self.config.reserved_tokens,
        )
        self.importance_overrides.clear()
