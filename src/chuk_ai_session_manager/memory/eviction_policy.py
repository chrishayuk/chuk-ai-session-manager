# chuk_ai_session_manager/memory/eviction_policy.py
"""
Eviction policy protocol and implementations for AI Virtual Memory.

Provides swappable eviction strategies. The default (ImportanceWeightedLRU)
exactly reproduces the scoring logic previously hardcoded in
WorkingSetManager.get_eviction_candidates().

Usage::

    from chuk_ai_session_manager.memory.eviction_policy import (
        ImportanceWeightedLRU,
        LRUEvictionPolicy,
        ModalityAwareLRU,
    )

    # Use default (same behavior as before)
    vm = MemoryManager(eviction_policy=ImportanceWeightedLRU())

    # Simple LRU
    vm = MemoryManager(eviction_policy=LRUEvictionPolicy())

    # Media evicts before text
    vm = MemoryManager(eviction_policy=ModalityAwareLRU())
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from .models import MemoryPage, Modality, StorageTier

logger = logging.getLogger(__name__)

# =============================================================================
# Models
# =============================================================================


class EvictionCandidate(BaseModel):
    """A scored eviction candidate. Lower score = evict first."""

    page_id: str
    score: float


class EvictionContext(BaseModel):
    """
    Context passed to eviction policies for scoring.

    Encapsulates everything a policy needs without coupling
    to WorkingSetManager internals.
    """

    model_config = {"arbitrary_types_allowed": True}

    current_turn: int = 0
    l0_page_ids: list[str] = Field(default_factory=list)
    l1_pages: dict[str, MemoryPage] = Field(default_factory=dict)
    pinned_page_ids: set[str] = Field(default_factory=set)
    importance_overrides: dict[str, float] = Field(default_factory=dict)
    anti_thrash: Any = None  # AntiThrashPolicy (avoid circular import)
    page_table: Any = None  # Optional PageTable for metadata lookups


# =============================================================================
# Protocol
# =============================================================================


@runtime_checkable
class EvictionPolicy(Protocol):
    """
    Protocol for swappable eviction strategies.

    Implementations receive an EvictionContext and return
    scored candidates. Lower score = evict first.
    Pinned and anti-thrash-protected pages MUST be excluded.
    """

    def score_candidates(
        self,
        context: EvictionContext,
        from_tier: StorageTier,
        tokens_needed: int = 0,
    ) -> list[EvictionCandidate]: ...


# =============================================================================
# Implementations
# =============================================================================


class ImportanceWeightedLRUConfig(BaseModel):
    """Configuration for ImportanceWeightedLRU policy."""

    # L0 weights (must sum to 1.0)
    l0_position_weight: float = 0.4
    l0_importance_weight: float = 0.4
    l0_thrash_weight: float = 0.2

    # L1 weights (must sum to 1.0)
    l1_recency_weight: float = 0.3
    l1_frequency_weight: float = 0.2
    l1_importance_weight: float = 0.3
    l1_thrash_weight: float = 0.2


class ImportanceWeightedLRU:
    """
    Default eviction policy.

    Exact behavioral match to the hardcoded logic previously in
    WorkingSetManager.get_eviction_candidates(). Configurable weights.

    L0 scoring: position * w1 + importance * w2 + thrash_penalty * w3
    L1 scoring: recency * w1 + frequency * w2 + importance * w3 + thrash * w4
    """

    def __init__(self, config: ImportanceWeightedLRUConfig | None = None) -> None:
        self.config = config or ImportanceWeightedLRUConfig()

    def score_candidates(
        self,
        context: EvictionContext,
        from_tier: StorageTier,
        tokens_needed: int = 0,  # noqa: ARG002 — part of EvictionPolicy protocol
    ) -> list[EvictionCandidate]:
        candidates = self._score_l0(context) if from_tier == StorageTier.L0 else self._score_l1(context)
        candidates.sort(key=lambda c: c.score)
        return candidates

    def _score_l0(self, context: EvictionContext) -> list[EvictionCandidate]:
        candidates: list[EvictionCandidate] = []
        cfg = self.config

        for i, page_id in enumerate(context.l0_page_ids):
            if page_id in context.pinned_page_ids:
                continue
            if context.anti_thrash and not context.anti_thrash.can_evict(page_id, context.current_turn):
                continue

            position_score = i / max(len(context.l0_page_ids), 1)
            importance = context.importance_overrides.get(page_id, 0.5)

            thrash_penalty = 0.0
            if context.anti_thrash:
                thrash_penalty = context.anti_thrash.get_eviction_penalty(page_id, context.current_turn)

            score = (
                position_score * cfg.l0_position_weight
                + importance * cfg.l0_importance_weight
                + thrash_penalty * cfg.l0_thrash_weight
            )
            candidates.append(EvictionCandidate(page_id=page_id, score=score))

        return candidates

    def _score_l1(self, context: EvictionContext) -> list[EvictionCandidate]:
        candidates: list[EvictionCandidate] = []
        cfg = self.config
        now = datetime.utcnow()

        for page_id, page in context.l1_pages.items():
            if page_id in context.pinned_page_ids or page.pinned:
                continue
            if context.anti_thrash and not context.anti_thrash.can_evict(page_id, context.current_turn):
                continue

            age_seconds = (now - page.last_accessed).total_seconds()
            recency_score = 1.0 / (1.0 + age_seconds / 3600)

            frequency_score = math.log1p(page.access_count) / 10.0
            frequency_score = min(1.0, frequency_score)

            importance = context.importance_overrides.get(page_id, page.importance)

            thrash_penalty = 0.0
            if context.anti_thrash:
                thrash_penalty = context.anti_thrash.get_eviction_penalty(page_id, context.current_turn)

            score = (
                recency_score * cfg.l1_recency_weight
                + frequency_score * cfg.l1_frequency_weight
                + importance * cfg.l1_importance_weight
                + thrash_penalty * cfg.l1_thrash_weight
            )
            candidates.append(EvictionCandidate(page_id=page_id, score=score))

        return candidates


class LRUEvictionPolicy:
    """
    Simple LRU eviction policy.

    L0: Position-based (older = lower score = evict first).
    L1: Recency-based (least recently accessed = evict first).
    No importance weighting.
    """

    def score_candidates(
        self,
        context: EvictionContext,
        from_tier: StorageTier,
        tokens_needed: int = 0,  # noqa: ARG002 — part of EvictionPolicy protocol
    ) -> list[EvictionCandidate]:
        candidates: list[EvictionCandidate] = []

        if from_tier == StorageTier.L0:
            for i, page_id in enumerate(context.l0_page_ids):
                if page_id in context.pinned_page_ids:
                    continue
                if context.anti_thrash and not context.anti_thrash.can_evict(page_id, context.current_turn):
                    continue

                score = i / max(len(context.l0_page_ids), 1)
                candidates.append(EvictionCandidate(page_id=page_id, score=score))
        else:
            now = datetime.utcnow()
            for page_id, page in context.l1_pages.items():
                if page_id in context.pinned_page_ids or page.pinned:
                    continue
                if context.anti_thrash and not context.anti_thrash.can_evict(page_id, context.current_turn):
                    continue

                age_seconds = (now - page.last_accessed).total_seconds()
                score = 1.0 / (1.0 + age_seconds / 3600)
                candidates.append(EvictionCandidate(page_id=page_id, score=score))

        candidates.sort(key=lambda c: c.score)
        return candidates


class ModalityAwareLRUConfig(BaseModel):
    """Configuration for modality-aware eviction."""

    modality_weights: dict[Modality, float] = Field(
        default_factory=lambda: {
            Modality.TEXT: 1.0,
            Modality.STRUCTURED: 0.9,
            Modality.IMAGE: 0.5,
            Modality.AUDIO: 0.4,
            Modality.VIDEO: 0.3,
        }
    )


class ModalityAwareLRU:
    """
    Modality-aware eviction: wraps a base policy and applies
    modality-based score multipliers.

    Higher modality weight = higher final score = keep longer.
    Media pages (low weight) evict before text pages (high weight).

    Requires page_table in EvictionContext for modality lookups.
    """

    def __init__(
        self,
        config: ModalityAwareLRUConfig | None = None,
        base_policy: EvictionPolicy | None = None,
    ) -> None:
        self.config = config or ModalityAwareLRUConfig()
        self._base = base_policy or ImportanceWeightedLRU()

    def score_candidates(
        self,
        context: EvictionContext,
        from_tier: StorageTier,
        tokens_needed: int = 0,
    ) -> list[EvictionCandidate]:
        base_candidates = self._base.score_candidates(context, from_tier, tokens_needed)

        adjusted: list[EvictionCandidate] = []
        for candidate in base_candidates:
            modality_weight = self._get_modality_weight(candidate.page_id, context)
            adjusted.append(
                EvictionCandidate(
                    page_id=candidate.page_id,
                    score=candidate.score * modality_weight,
                )
            )

        adjusted.sort(key=lambda c: c.score)
        return adjusted

    def _get_modality_weight(self, page_id: str, context: EvictionContext) -> float:
        """Look up modality weight for a page via page_table or L1 cache."""
        default_weight = 1.0

        # Try L1 cache first (has full MemoryPage)
        if page_id in context.l1_pages:
            modality = context.l1_pages[page_id].modality
            return self.config.modality_weights.get(modality, default_weight)

        # Try page table
        if context.page_table is not None:
            entry = context.page_table.lookup(page_id)
            if entry is not None:
                return self.config.modality_weights.get(entry.modality, default_weight)

        return default_weight
