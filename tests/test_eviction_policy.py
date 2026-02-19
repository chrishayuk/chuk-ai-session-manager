# tests/test_eviction_policy.py
"""
Tests for the eviction policy module.

Covers:
- EvictionCandidate model creation and ordering
- EvictionContext construction from working set state
- ImportanceWeightedLRU (default policy matching original hardcoded logic)
- LRUEvictionPolicy (simple position/recency-based)
- ModalityAwareLRU (modality-based score multipliers)
- Backward compatibility (WorkingSetManager with and without custom policy)
"""

import math
from datetime import datetime, timedelta
from typing import Set

import pytest

from chuk_ai_session_manager.memory.eviction_policy import (
    EvictionCandidate,
    EvictionContext,
    ImportanceWeightedLRU,
    ImportanceWeightedLRUConfig,
    LRUEvictionPolicy,
    ModalityAwareLRU,
    ModalityAwareLRUConfig,
)
from chuk_ai_session_manager.memory.models import (
    MemoryPage,
    Modality,
    PageTableEntry,
    PageType,
    StorageTier,
)
from chuk_ai_session_manager.memory.working_set import (
    AntiThrashPolicy,
    WorkingSetManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockAntiThrash:
    """Mock anti-thrash policy that blocks specific page IDs."""

    def __init__(self, blocked_ids: Set[str] | None = None):
        self._blocked = blocked_ids or set()

    def can_evict(self, page_id: str, current_turn: int) -> bool:
        return page_id not in self._blocked

    def get_eviction_penalty(self, page_id: str, current_turn: int) -> float:
        return 1.0 if page_id in self._blocked else 0.0


class MockPageTable:
    """Minimal mock page table with a lookup method."""

    def __init__(self, entries: dict[str, PageTableEntry] | None = None):
        self._entries = entries or {}

    def lookup(self, page_id: str):
        return self._entries.get(page_id)


def _make_l1_page(
    page_id: str,
    modality: Modality = Modality.TEXT,
    importance: float = 0.5,
    access_count: int = 1,
    age_hours: float = 1.0,
    pinned: bool = False,
) -> MemoryPage:
    """Create a MemoryPage suitable for L1 scoring tests."""
    accessed = datetime.utcnow() - timedelta(hours=age_hours)
    return MemoryPage(
        page_id=page_id,
        modality=modality,
        importance=importance,
        access_count=access_count,
        last_accessed=accessed,
        storage_tier=StorageTier.L1,
        pinned=pinned,
    )


# =============================================================================
# TestEvictionCandidate
# =============================================================================


class TestEvictionCandidate:
    """Tests for the EvictionCandidate model."""

    def test_creation(self):
        """EvictionCandidate stores page_id and score."""
        c = EvictionCandidate(page_id="page-1", score=0.42)
        assert c.page_id == "page-1"
        assert c.score == pytest.approx(0.42)

    def test_metadata_fields(self):
        """EvictionCandidate is a Pydantic model with expected fields."""
        c = EvictionCandidate(page_id="abc", score=0.0)
        data = c.model_dump()
        assert "page_id" in data
        assert "score" in data
        assert len(data) == 2

    def test_score_ordering(self):
        """Candidates can be sorted by score; lower score = evict first."""
        candidates = [
            EvictionCandidate(page_id="high", score=0.9),
            EvictionCandidate(page_id="low", score=0.1),
            EvictionCandidate(page_id="mid", score=0.5),
        ]
        candidates.sort(key=lambda c: c.score)
        assert [c.page_id for c in candidates] == ["low", "mid", "high"]


# =============================================================================
# TestEvictionContext
# =============================================================================


class TestEvictionContext:
    """Tests for the EvictionContext model."""

    def test_from_working_set_state(self):
        """EvictionContext can be built from typical working set state."""
        page = _make_l1_page("l1-page")
        ctx = EvictionContext(
            current_turn=5,
            l0_page_ids=["p1", "p2"],
            l1_pages={"l1-page": page},
            pinned_page_ids={"p1"},
            importance_overrides={"p2": 0.8},
        )
        assert ctx.current_turn == 5
        assert ctx.l0_page_ids == ["p1", "p2"]
        assert "l1-page" in ctx.l1_pages
        assert "p1" in ctx.pinned_page_ids
        assert ctx.importance_overrides["p2"] == pytest.approx(0.8)
        assert ctx.anti_thrash is None
        assert ctx.page_table is None

    def test_empty_context(self):
        """EvictionContext defaults produce an empty context."""
        ctx = EvictionContext()
        assert ctx.current_turn == 0
        assert ctx.l0_page_ids == []
        assert ctx.l1_pages == {}
        assert ctx.pinned_page_ids == set()
        assert ctx.importance_overrides == {}

    def test_with_page_table(self):
        """EvictionContext accepts a page_table reference."""
        mock_pt = MockPageTable()
        ctx = EvictionContext(page_table=mock_pt)
        assert ctx.page_table is mock_pt


# =============================================================================
# TestImportanceWeightedLRU
# =============================================================================


class TestImportanceWeightedLRU:
    """Tests for the ImportanceWeightedLRU eviction policy."""

    def test_default_config(self):
        """Default config has expected weight values."""
        policy = ImportanceWeightedLRU()
        cfg = policy.config
        assert cfg.l0_position_weight == pytest.approx(0.4)
        assert cfg.l0_importance_weight == pytest.approx(0.4)
        assert cfg.l0_thrash_weight == pytest.approx(0.2)
        assert cfg.l1_recency_weight == pytest.approx(0.3)
        assert cfg.l1_frequency_weight == pytest.approx(0.2)
        assert cfg.l1_importance_weight == pytest.approx(0.3)
        assert cfg.l1_thrash_weight == pytest.approx(0.2)

    def test_custom_config(self):
        """Custom config is stored and used."""
        cfg = ImportanceWeightedLRUConfig(
            l0_position_weight=0.6,
            l0_importance_weight=0.3,
            l0_thrash_weight=0.1,
        )
        policy = ImportanceWeightedLRU(config=cfg)
        assert policy.config.l0_position_weight == pytest.approx(0.6)

    def test_l0_scoring_matches_original(self):
        """L0 scoring reproduces the original hardcoded formula.

        Original formula:
            score = position_score * 0.4 + importance * 0.4 + thrash_penalty * 0.2
        where position_score = i / len(l0_page_ids), default importance = 0.5,
        and thrash_penalty = 0 (no anti-thrash).
        """
        policy = ImportanceWeightedLRU()
        l0_ids = ["p0", "p1", "p2", "p3"]
        ctx = EvictionContext(l0_page_ids=l0_ids)

        candidates = policy.score_candidates(ctx, StorageTier.L0)

        assert len(candidates) == 4
        # Verify the formula for the first page (position 0)
        # score = (0/4) * 0.4 + 0.5 * 0.4 + 0 * 0.2 = 0 + 0.2 + 0 = 0.2
        assert candidates[0].score == pytest.approx(0.2)
        # Third page (position 2): (2/4) * 0.4 + 0.5 * 0.4 = 0.2 + 0.2 = 0.4
        p2_cand = [c for c in candidates if c.page_id == "p2"][0]
        assert p2_cand.score == pytest.approx(0.4)

    def test_l1_scoring_matches_original(self):
        """L1 scoring reproduces the original hardcoded formula.

        Original formula:
            score = recency * 0.3 + frequency * 0.2 + importance * 0.3 + thrash * 0.2
        """
        page = _make_l1_page("l1-a", access_count=5, age_hours=2.0, importance=0.7)
        policy = ImportanceWeightedLRU()
        ctx = EvictionContext(l1_pages={"l1-a": page})

        candidates = policy.score_candidates(ctx, StorageTier.L1)

        assert len(candidates) == 1
        c = candidates[0]
        assert c.page_id == "l1-a"

        # Verify the components manually
        recency = 1.0 / (1.0 + 2.0)  # age_hours / 1 hour normalization
        frequency = min(1.0, math.log1p(5) / 10.0)
        expected = recency * 0.3 + frequency * 0.2 + 0.7 * 0.3 + 0.0 * 0.2
        assert c.score == pytest.approx(expected, abs=0.01)

    def test_pinned_excluded(self):
        """Pinned pages are excluded from L0 candidates."""
        policy = ImportanceWeightedLRU()
        ctx = EvictionContext(
            l0_page_ids=["p0", "p1", "p2"],
            pinned_page_ids={"p1"},
        )
        candidates = policy.score_candidates(ctx, StorageTier.L0)
        ids = {c.page_id for c in candidates}
        assert "p1" not in ids
        assert len(candidates) == 2

    def test_anti_thrash_excluded(self):
        """Pages blocked by anti-thrash are excluded."""
        policy = ImportanceWeightedLRU()
        mock_at = MockAntiThrash(blocked_ids={"p2"})
        ctx = EvictionContext(
            l0_page_ids=["p0", "p1", "p2"],
            anti_thrash=mock_at,
        )
        candidates = policy.score_candidates(ctx, StorageTier.L0)
        ids = {c.page_id for c in candidates}
        assert "p2" not in ids
        assert len(candidates) == 2

    def test_empty_l0(self):
        """Empty L0 produces no candidates."""
        policy = ImportanceWeightedLRU()
        ctx = EvictionContext(l0_page_ids=[])
        candidates = policy.score_candidates(ctx, StorageTier.L0)
        assert candidates == []

    def test_empty_l1(self):
        """Empty L1 produces no candidates."""
        policy = ImportanceWeightedLRU()
        ctx = EvictionContext(l1_pages={})
        candidates = policy.score_candidates(ctx, StorageTier.L1)
        assert candidates == []

    def test_sorted_order(self):
        """Candidates are returned sorted by score (ascending)."""
        policy = ImportanceWeightedLRU()
        # p0 at position 0 should score lowest, p4 at position 4 highest
        l0_ids = [f"p{i}" for i in range(5)]
        ctx = EvictionContext(l0_page_ids=l0_ids)
        candidates = policy.score_candidates(ctx, StorageTier.L0)
        scores = [c.score for c in candidates]
        assert scores == sorted(scores)

    def test_importance_overrides(self):
        """Importance overrides affect scoring for specific pages."""
        policy = ImportanceWeightedLRU()
        ctx = EvictionContext(
            l0_page_ids=["p0", "p1"],
            importance_overrides={"p0": 1.0},  # High importance
        )
        candidates = policy.score_candidates(ctx, StorageTier.L0)
        scores = {c.page_id: c.score for c in candidates}
        # p0 has importance=1.0 vs p1 has default=0.5
        # p0: (0/2)*0.4 + 1.0*0.4 = 0.4
        # p1: (1/2)*0.4 + 0.5*0.4 = 0.2 + 0.2 = 0.4
        # With importance=1.0, p0 score is higher
        assert scores["p0"] == pytest.approx(0.4)
        assert scores["p1"] == pytest.approx(0.4)

    def test_l1_pinned_on_page_excluded(self):
        """Pages with pinned=True on the MemoryPage itself are excluded from L1."""
        page_pinned = _make_l1_page("pinned-page", pinned=True)
        page_normal = _make_l1_page("normal-page", pinned=False)
        policy = ImportanceWeightedLRU()
        ctx = EvictionContext(
            l1_pages={"pinned-page": page_pinned, "normal-page": page_normal},
        )
        candidates = policy.score_candidates(ctx, StorageTier.L1)
        ids = {c.page_id for c in candidates}
        assert "pinned-page" not in ids
        assert "normal-page" in ids

    def test_anti_thrash_penalty_affects_l0_score(self):
        """Anti-thrash penalty increases the score (less likely to evict)."""
        policy = ImportanceWeightedLRU()
        # Without anti-thrash
        ctx_no_at = EvictionContext(l0_page_ids=["p0"])
        c_no_at = policy.score_candidates(ctx_no_at, StorageTier.L0)

        # With anti-thrash that allows eviction but adds penalty
        at = AntiThrashPolicy()
        at.record_fault("p0", 0)  # Fault at turn 0
        ctx_with_at = EvictionContext(
            l0_page_ids=["p0"],
            anti_thrash=at,
            current_turn=1,  # Within fault_protection_turns (default=2)
        )
        c_with_at = policy.score_candidates(ctx_with_at, StorageTier.L0)

        # The page may be excluded entirely if can_evict returns False
        # or it may get a higher score if can_evict is True with penalty
        if len(c_with_at) > 0:
            assert c_with_at[0].score >= c_no_at[0].score


# =============================================================================
# TestLRUEvictionPolicy
# =============================================================================


class TestLRUEvictionPolicy:
    """Tests for the LRUEvictionPolicy (simple LRU)."""

    def test_l0_position_scoring(self):
        """L0 pages scored purely by position."""
        policy = LRUEvictionPolicy()
        l0_ids = ["p0", "p1", "p2"]
        ctx = EvictionContext(l0_page_ids=l0_ids)
        candidates = policy.score_candidates(ctx, StorageTier.L0)

        assert len(candidates) == 3
        # p0 at position 0: score = 0/3 = 0.0
        assert candidates[0].page_id == "p0"
        assert candidates[0].score == pytest.approx(0.0)
        # p1 at position 1: score = 1/3
        assert candidates[1].page_id == "p1"
        assert candidates[1].score == pytest.approx(1 / 3)

    def test_l1_recency_scoring(self):
        """L1 pages scored by recency; older page gets lower score."""
        old_page = _make_l1_page("old", age_hours=10.0)
        new_page = _make_l1_page("new", age_hours=0.1)
        policy = LRUEvictionPolicy()
        ctx = EvictionContext(l1_pages={"old": old_page, "new": new_page})

        candidates = policy.score_candidates(ctx, StorageTier.L1)
        scores = {c.page_id: c.score for c in candidates}
        # Older page should have a lower recency score (evict first)
        assert scores["old"] < scores["new"]

    def test_pinned_excluded(self):
        """Pinned pages are excluded from LRU candidates."""
        policy = LRUEvictionPolicy()
        ctx = EvictionContext(
            l0_page_ids=["p0", "p1"],
            pinned_page_ids={"p0"},
        )
        candidates = policy.score_candidates(ctx, StorageTier.L0)
        ids = {c.page_id for c in candidates}
        assert "p0" not in ids
        assert len(candidates) == 1

    def test_anti_thrash_excluded(self):
        """Anti-thrash blocked pages are excluded."""
        policy = LRUEvictionPolicy()
        mock_at = MockAntiThrash(blocked_ids={"p1"})
        ctx = EvictionContext(
            l0_page_ids=["p0", "p1", "p2"],
            anti_thrash=mock_at,
        )
        candidates = policy.score_candidates(ctx, StorageTier.L0)
        ids = {c.page_id for c in candidates}
        assert "p1" not in ids

    def test_empty(self):
        """Empty tiers produce no candidates."""
        policy = LRUEvictionPolicy()
        ctx = EvictionContext()
        assert policy.score_candidates(ctx, StorageTier.L0) == []
        assert policy.score_candidates(ctx, StorageTier.L1) == []

    def test_sorted_by_score(self):
        """Results are sorted ascending by score."""
        policy = LRUEvictionPolicy()
        l0_ids = [f"p{i}" for i in range(6)]
        ctx = EvictionContext(l0_page_ids=l0_ids)
        candidates = policy.score_candidates(ctx, StorageTier.L0)
        scores = [c.score for c in candidates]
        assert scores == sorted(scores)


# =============================================================================
# TestModalityAwareLRU
# =============================================================================


class TestModalityAwareLRU:
    """Tests for the ModalityAwareLRU eviction policy."""

    def test_default_weights(self):
        """Default modality weights: TEXT > STRUCTURED > IMAGE > AUDIO > VIDEO."""
        policy = ModalityAwareLRU()
        w = policy.config.modality_weights
        assert w[Modality.TEXT] > w[Modality.STRUCTURED]
        assert w[Modality.STRUCTURED] > w[Modality.IMAGE]
        assert w[Modality.IMAGE] > w[Modality.AUDIO]
        assert w[Modality.AUDIO] > w[Modality.VIDEO]

    def test_custom_weights(self):
        """Custom modality weights override defaults."""
        cfg = ModalityAwareLRUConfig(
            modality_weights={
                Modality.TEXT: 0.1,
                Modality.IMAGE: 0.9,
            }
        )
        policy = ModalityAwareLRU(config=cfg)
        assert policy.config.modality_weights[Modality.TEXT] == pytest.approx(0.1)
        assert policy.config.modality_weights[Modality.IMAGE] == pytest.approx(0.9)

    def test_image_evicted_before_text(self):
        """Image pages get lower final scores than text pages (evict images first).

        Image weight (0.5) < Text weight (1.0), so image score is halved,
        making it sort earlier (lower score = evict first).
        """
        text_page = _make_l1_page("text-page", modality=Modality.TEXT, age_hours=1.0)
        image_page = _make_l1_page("image-page", modality=Modality.IMAGE, age_hours=1.0)

        policy = ModalityAwareLRU()
        ctx = EvictionContext(
            l1_pages={"text-page": text_page, "image-page": image_page},
        )
        candidates = policy.score_candidates(ctx, StorageTier.L1)
        scores = {c.page_id: c.score for c in candidates}
        # Image should have lower score -> evict first
        assert scores["image-page"] < scores["text-page"]

    def test_video_evicted_before_audio(self):
        """Video pages (weight 0.3) evict before audio pages (weight 0.4)."""
        audio_page = _make_l1_page("audio-page", modality=Modality.AUDIO, age_hours=1.0)
        video_page = _make_l1_page("video-page", modality=Modality.VIDEO, age_hours=1.0)

        policy = ModalityAwareLRU()
        ctx = EvictionContext(
            l1_pages={"audio-page": audio_page, "video-page": video_page},
        )
        candidates = policy.score_candidates(ctx, StorageTier.L1)
        scores = {c.page_id: c.score for c in candidates}
        assert scores["video-page"] < scores["audio-page"]

    def test_wraps_base_policy(self):
        """ModalityAwareLRU delegates to a base policy for initial scoring."""
        base = LRUEvictionPolicy()
        policy = ModalityAwareLRU(base_policy=base)
        # Without modality info (L0, no page_table), weight defaults to 1.0
        ctx = EvictionContext(l0_page_ids=["p0", "p1"])
        candidates = policy.score_candidates(ctx, StorageTier.L0)

        # Should still get 2 candidates from the base policy
        assert len(candidates) == 2

    def test_pinned_excluded(self):
        """Pinned pages are excluded (inherited from base policy)."""
        policy = ModalityAwareLRU()
        page_a = _make_l1_page("a", modality=Modality.TEXT)
        page_b = _make_l1_page("b", modality=Modality.IMAGE)
        ctx = EvictionContext(
            l1_pages={"a": page_a, "b": page_b},
            pinned_page_ids={"a"},
        )
        candidates = policy.score_candidates(ctx, StorageTier.L1)
        ids = {c.page_id for c in candidates}
        assert "a" not in ids
        assert "b" in ids

    def test_mixed_modalities_sorted(self):
        """Mixed-modality candidates are sorted by adjusted score."""
        text_page = _make_l1_page("txt", modality=Modality.TEXT, age_hours=1.0)
        img_page = _make_l1_page("img", modality=Modality.IMAGE, age_hours=1.0)
        vid_page = _make_l1_page("vid", modality=Modality.VIDEO, age_hours=1.0)

        policy = ModalityAwareLRU()
        ctx = EvictionContext(
            l1_pages={"txt": text_page, "img": img_page, "vid": vid_page},
        )
        candidates = policy.score_candidates(ctx, StorageTier.L1)
        scores = [c.score for c in candidates]
        # Must be sorted ascending
        assert scores == sorted(scores)
        # Video should be evicted first (lowest score)
        assert candidates[0].page_id == "vid"


# =============================================================================
# TestBackwardCompat
# =============================================================================


class TestBackwardCompat:
    """Tests for backward compatibility of WorkingSetManager eviction."""

    def test_no_policy_uses_old_logic(self):
        """Without set_eviction_policy, get_eviction_candidates uses hardcoded logic."""
        ws = WorkingSetManager()
        ws.l0_pages = ["p0", "p1", "p2"]
        ws.current_turn = 10

        candidates = ws.get_eviction_candidates(from_tier=StorageTier.L0)

        # Should return tuples of (page_id, score) using the original formula
        assert len(candidates) == 3
        assert all(isinstance(c, tuple) and len(c) == 2 for c in candidates)
        # First candidate (position 0) should have lowest score
        # score = (0/3)*0.4 + 0.5*0.4 + 0*0.2 = 0.2
        assert candidates[0][0] == "p0"
        assert candidates[0][1] == pytest.approx(0.2)

    def test_with_policy_delegates(self):
        """After set_eviction_policy, get_eviction_candidates delegates to it."""
        ws = WorkingSetManager()
        ws.l0_pages = ["p0", "p1", "p2"]
        ws.current_turn = 10

        policy = LRUEvictionPolicy()
        ws.set_eviction_policy(policy)

        candidates = ws.get_eviction_candidates(from_tier=StorageTier.L0)

        # LRU policy uses pure position scoring: score = i / len(l0)
        assert len(candidates) == 3
        # p0: score = 0/3 = 0.0
        assert candidates[0][0] == "p0"
        assert candidates[0][1] == pytest.approx(0.0)
        # p1: score = 1/3
        assert candidates[1][0] == "p1"
        assert candidates[1][1] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


class TestImportanceWeightedLRUL1AntiThrash:
    """Cover the L1 anti-thrash continue branch (line 185)."""

    def test_l1_anti_thrash_excluded(self):
        page_a = _make_l1_page("a", age_hours=2.0)
        page_b = _make_l1_page("b", age_hours=1.0)
        ctx = EvictionContext(
            current_turn=5,
            l1_pages={"a": page_a, "b": page_b},
            anti_thrash=MockAntiThrash(blocked_ids={"a"}),
        )
        policy = ImportanceWeightedLRU()
        candidates = policy.score_candidates(ctx, StorageTier.L1)
        ids = [c.page_id for c in candidates]
        assert "a" not in ids
        assert "b" in ids

    def test_l1_anti_thrash_penalty_applied(self):
        """Verify anti-thrash penalty affects L1 score (line 199)."""
        page = _make_l1_page("p1", age_hours=1.0, importance=0.5)
        ctx_no_thrash = EvictionContext(
            current_turn=5,
            l1_pages={"p1": page},
        )

        # With a mock that returns non-zero penalty
        class PenaltyThrash:
            def can_evict(self, pid, turn):
                return True

            def get_eviction_penalty(self, pid, turn):
                return 0.8

        ctx_with_thrash = EvictionContext(
            current_turn=5,
            l1_pages={"p1": page},
            anti_thrash=PenaltyThrash(),
        )
        policy = ImportanceWeightedLRU()
        score_clean = policy.score_candidates(ctx_no_thrash, StorageTier.L1)[0].score
        score_thrash = policy.score_candidates(ctx_with_thrash, StorageTier.L1)[0].score
        assert score_thrash > score_clean  # Penalty increases score (harder to evict)


class TestLRUEvictionPolicyL1Coverage:
    """Cover LRU L1 anti-thrash branches (lines 248, 252)."""

    def test_l1_pinned_excluded(self):
        page = _make_l1_page("p1", pinned=True)
        ctx = EvictionContext(
            current_turn=5,
            l1_pages={"p1": page},
            pinned_page_ids=set(),
        )
        candidates = LRUEvictionPolicy().score_candidates(ctx, StorageTier.L1)
        assert len(candidates) == 0  # Pinned via page.pinned flag

    def test_l1_anti_thrash_excluded(self):
        page = _make_l1_page("p1")
        ctx = EvictionContext(
            current_turn=5,
            l1_pages={"p1": page},
            anti_thrash=MockAntiThrash(blocked_ids={"p1"}),
        )
        candidates = LRUEvictionPolicy().score_candidates(ctx, StorageTier.L1)
        assert len(candidates) == 0


class TestModalityAwareLRUPageTableLookup:
    """Cover page_table modality lookup branch (lines 335-337)."""

    def test_modality_from_page_table(self):
        """When page is in L0 (not L1), look up modality via page_table."""
        entry = PageTableEntry(
            page_id="p0",
            page_type=PageType.TRANSCRIPT,
            modality=Modality.IMAGE,
            tier=StorageTier.L0,
        )
        pt = MockPageTable(entries={"p0": entry})
        ctx = EvictionContext(
            current_turn=5,
            l0_page_ids=["p0"],
            page_table=pt,
        )
        policy = ModalityAwareLRU()
        candidates = policy.score_candidates(ctx, StorageTier.L0)
        assert len(candidates) == 1
        # IMAGE weight = 0.5, so score should be multiplied by 0.5
        base_policy = ImportanceWeightedLRU()
        base_candidates = base_policy.score_candidates(ctx, StorageTier.L0)
        assert candidates[0].score == pytest.approx(base_candidates[0].score * 0.5)

    def test_modality_unknown_falls_to_default(self):
        """When page_table entry has no matching weight, use default 1.0."""
        ctx = EvictionContext(
            current_turn=5,
            l0_page_ids=["p0"],
            page_table=MockPageTable(entries={}),  # no entry
        )
        policy = ModalityAwareLRU()
        candidates = policy.score_candidates(ctx, StorageTier.L0)
        base = ImportanceWeightedLRU().score_candidates(ctx, StorageTier.L0)
        # Default weight = 1.0, so score should match base
        assert candidates[0].score == pytest.approx(base[0].score)
