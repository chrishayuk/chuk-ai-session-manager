"""
Comprehensive tests for the memory subsystem to achieve >90% coverage.

Covers: page_table, tlb, working_set, context_packer, vm_prompts,
        manifest, artifacts_bridge, prefetcher.
"""

import json
from datetime import datetime, timedelta

import pytest

from chuk_ai_session_manager.exceptions import StorageError
from chuk_ai_session_manager.memory.artifacts_bridge import (
    ArtifactsBridge,
    InMemoryBackend,
)
from chuk_ai_session_manager.memory.context_packer import (
    ContextPacker,
    ContextPackerConfig,
)
from chuk_ai_session_manager.memory.manifest import (
    AvailablePageEntry,
    HintType,
    ManifestBuilder,
    ManifestPolicies,
    VMManifest,
    WorkingSetEntry,
    generate_simple_hint,
)
from chuk_ai_session_manager.memory.models import (
    CompressionLevel,
    MemoryPage,
    Modality,
    PageTableEntry,
    PageTableStats,
    PageType,
    StorageTier,
)
from chuk_ai_session_manager.memory.page_table import PageTable
from chuk_ai_session_manager.memory.prefetcher import (
    SimplePrefetcher,
)
from chuk_ai_session_manager.memory.tlb import PageTLB, TLBWithPageTable
from chuk_ai_session_manager.memory.vm_prompts import (
    PAGE_FAULT_TOOL,
    SEARCH_PAGES_TOOL,
    VM_PASSIVE_PROMPT,
    VM_PROMPTS,
    VM_RELAXED_PROMPT,
    VM_STRICT_PROMPT,
    VM_TOOL_DEFINITIONS,
    VM_TOOLS,
    VMMode,
    build_vm_developer_message,
    get_prompt_for_mode,
    get_vm_tools,
    get_vm_tools_as_dicts,
)
from chuk_ai_session_manager.memory.working_set import (
    AntiThrashPolicy,
    PinnedSet,
    WorkingSetConfig,
    WorkingSetManager,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_page(
    page_id="p1",
    content="test content",
    tier=StorageTier.L0,
    modality=Modality.TEXT,
    **kwargs,
):
    return MemoryPage(
        page_id=page_id,
        session_id="test-session",
        content=content,
        storage_tier=tier,
        modality=modality,
        **kwargs,
    )


def make_entry(
    page_id="p1",
    tier=StorageTier.L0,
    modality=Modality.TEXT,
    size_tokens=100,
    **kwargs,
):
    return PageTableEntry(
        page_id=page_id,
        tier=tier,
        modality=modality,
        size_tokens=size_tokens,
        **kwargs,
    )


# ===================================================================
# 1. PageTable tests
# ===================================================================


class TestPageTable:
    """Tests for PageTable."""

    def test_len_empty(self):
        pt = PageTable()
        assert len(pt) == 0

    def test_len_with_entries(self):
        pt = PageTable()
        pt.register(make_page("p1"))
        pt.register(make_page("p2"))
        assert len(pt) == 2

    def test_contains_present(self):
        pt = PageTable()
        pt.register(make_page("p1"))
        assert "p1" in pt

    def test_contains_absent(self):
        pt = PageTable()
        assert "p1" not in pt

    def test_lookup_found(self):
        pt = PageTable()
        pt.register(make_page("p1"))
        entry = pt.lookup("p1")
        assert entry is not None
        assert entry.page_id == "p1"

    def test_lookup_not_found(self):
        pt = PageTable()
        assert pt.lookup("missing") is None

    def test_register_creates_entry_from_page(self):
        pt = PageTable()
        page = make_page("p1", tier=StorageTier.L1, modality=Modality.IMAGE)
        entry = pt.register(page)
        assert entry.page_id == "p1"
        assert entry.tier == StorageTier.L1
        assert entry.modality == Modality.IMAGE
        assert "p1" in pt

    def test_register_page_with_tokens(self):
        page = make_page("p1", content="hello world", size_tokens=42)
        pt = PageTable()
        entry = pt.register(page)
        assert entry.size_tokens == 42

    def test_register_page_estimates_tokens_when_none(self):
        page = make_page("p1", content="A" * 100)
        pt = PageTable()
        entry = pt.register(page)
        # size_tokens should be estimate_tokens() = 100 // 4 = 25
        assert entry.size_tokens == 25

    def test_register_entry_direct(self):
        pt = PageTable()
        entry = make_entry("p1", tier=StorageTier.L2)
        pt.register_entry(entry)
        assert "p1" in pt
        assert pt.lookup("p1").tier == StorageTier.L2

    def test_add_entry_updates_existing(self):
        pt = PageTable()
        entry1 = make_entry("p1", tier=StorageTier.L0, modality=Modality.TEXT)
        pt.register_entry(entry1)
        assert len(pt.get_by_tier(StorageTier.L0)) == 1

        # Re-register with different tier and modality
        entry2 = make_entry("p1", tier=StorageTier.L2, modality=Modality.IMAGE)
        pt.register_entry(entry2)

        assert len(pt) == 1
        assert pt.lookup("p1").tier == StorageTier.L2
        assert pt.lookup("p1").modality == Modality.IMAGE
        assert len(pt.get_by_tier(StorageTier.L0)) == 0
        assert len(pt.get_by_tier(StorageTier.L2)) == 1

    def test_add_entry_moves_from_old_dirty_index(self):
        pt = PageTable()
        entry = make_entry("p1", tier=StorageTier.L0, dirty=True)
        pt.register_entry(entry)
        assert len(pt.get_dirty_pages()) == 1

        # Replace with clean entry
        entry2 = make_entry("p1", tier=StorageTier.L1, dirty=False)
        pt.register_entry(entry2)
        assert len(pt.get_dirty_pages()) == 0

    def test_remove_found(self):
        pt = PageTable()
        pt.register(make_page("p1"))
        removed = pt.remove("p1")
        assert removed is not None
        assert removed.page_id == "p1"
        assert "p1" not in pt

    def test_remove_not_found(self):
        pt = PageTable()
        removed = pt.remove("missing")
        assert removed is None

    def test_remove_clears_indexes(self):
        pt = PageTable()
        pt.register(make_page("p1", tier=StorageTier.L0, modality=Modality.TEXT))
        pt.mark_dirty("p1")
        pt.remove("p1")
        assert len(pt.get_by_tier(StorageTier.L0)) == 0
        assert len(pt.get_by_modality(Modality.TEXT)) == 0
        assert len(pt.get_dirty_pages()) == 0

    def test_update_location_success(self):
        pt = PageTable()
        pt.register(make_page("p1", tier=StorageTier.L0))
        result = pt.update_location("p1", tier=StorageTier.L3)
        assert result is True
        assert pt.lookup("p1").tier == StorageTier.L3

    def test_update_location_with_artifact_id(self):
        pt = PageTable()
        pt.register(make_page("p1"))
        pt.update_location("p1", tier=StorageTier.L3, artifact_id="art_123")
        assert pt.lookup("p1").artifact_id == "art_123"

    def test_update_location_with_compression(self):
        pt = PageTable()
        pt.register(make_page("p1"))
        pt.update_location("p1", tier=StorageTier.L3, compression_level=CompressionLevel.ABSTRACT)
        assert pt.lookup("p1").compression_level == CompressionLevel.ABSTRACT

    def test_update_location_not_found(self):
        pt = PageTable()
        result = pt.update_location("missing", tier=StorageTier.L3)
        assert result is False

    def test_update_location_tier_index_updated(self):
        pt = PageTable()
        pt.register(make_page("p1", tier=StorageTier.L0))
        assert len(pt.get_by_tier(StorageTier.L0)) == 1
        assert len(pt.get_by_tier(StorageTier.L2)) == 0

        pt.update_location("p1", tier=StorageTier.L2)
        assert len(pt.get_by_tier(StorageTier.L0)) == 0
        assert len(pt.get_by_tier(StorageTier.L2)) == 1

    def test_mark_accessed_success(self):
        pt = PageTable()
        pt.register(make_page("p1"))
        old_count = pt.lookup("p1").access_count
        result = pt.mark_accessed("p1")
        assert result is True
        assert pt.lookup("p1").access_count == old_count + 1

    def test_mark_accessed_not_found(self):
        pt = PageTable()
        assert pt.mark_accessed("missing") is False

    def test_mark_dirty_success(self):
        pt = PageTable()
        pt.register(make_page("p1"))
        result = pt.mark_dirty("p1")
        assert result is True
        assert pt.lookup("p1").dirty is True
        assert "p1" in [e.page_id for e in pt.get_dirty_pages()]

    def test_mark_dirty_not_found(self):
        pt = PageTable()
        assert pt.mark_dirty("missing") is False

    def test_mark_clean_success(self):
        pt = PageTable()
        pt.register(make_page("p1"))
        pt.mark_dirty("p1")
        assert pt.lookup("p1").dirty is True

        result = pt.mark_clean("p1")
        assert result is True
        assert pt.lookup("p1").dirty is False
        assert pt.lookup("p1").last_flushed is not None
        assert len(pt.get_dirty_pages()) == 0

    def test_mark_clean_not_found(self):
        pt = PageTable()
        assert pt.mark_clean("missing") is False

    def test_get_by_tier_with_entries(self):
        pt = PageTable()
        pt.register(make_page("p1", tier=StorageTier.L0))
        pt.register(make_page("p2", tier=StorageTier.L0))
        pt.register(make_page("p3", tier=StorageTier.L2))
        entries = pt.get_by_tier(StorageTier.L0)
        assert len(entries) == 2
        ids = {e.page_id for e in entries}
        assert ids == {"p1", "p2"}

    def test_get_by_tier_empty(self):
        pt = PageTable()
        entries = pt.get_by_tier(StorageTier.L4)
        assert entries == []

    def test_get_by_modality_with_entries(self):
        pt = PageTable()
        pt.register(make_page("p1", modality=Modality.TEXT))
        pt.register(make_page("p2", modality=Modality.IMAGE))
        pt.register(make_page("p3", modality=Modality.TEXT))
        entries = pt.get_by_modality(Modality.TEXT)
        assert len(entries) == 2

    def test_get_by_modality_empty(self):
        pt = PageTable()
        entries = pt.get_by_modality(Modality.VIDEO)
        assert entries == []

    def test_get_dirty_pages(self):
        pt = PageTable()
        pt.register(make_page("p1"))
        pt.register(make_page("p2"))
        pt.mark_dirty("p1")
        dirty = pt.get_dirty_pages()
        assert len(dirty) == 1
        assert dirty[0].page_id == "p1"

    def test_get_working_set(self):
        pt = PageTable()
        pt.register(make_page("p1", tier=StorageTier.L0))
        pt.register(make_page("p2", tier=StorageTier.L1))
        pt.register(make_page("p3", tier=StorageTier.L2))
        ws = pt.get_working_set()
        ids = {e.page_id for e in ws}
        assert ids == {"p1", "p2"}

    def test_get_eviction_candidates_sorted_by_lru(self):
        pt = PageTable()
        # Create pages with specific access times (oldest first)
        now = datetime.utcnow()
        for i, pid in enumerate(["old", "mid", "new"]):
            page = make_page(pid, tier=StorageTier.L1)
            entry = pt.register(page)
            entry.last_accessed = now - timedelta(seconds=100 - i * 50)

        candidates = pt.get_eviction_candidates(tier=StorageTier.L1, limit=10)
        assert candidates[0].page_id == "old"
        assert candidates[-1].page_id == "new"

    def test_get_eviction_candidates_respects_limit(self):
        pt = PageTable()
        for i in range(10):
            pt.register(make_page(f"p{i}", tier=StorageTier.L1))
        candidates = pt.get_eviction_candidates(tier=StorageTier.L1, limit=3)
        assert len(candidates) == 3

    def test_get_stats(self):
        pt = PageTable()
        pt.register(make_page("p1", tier=StorageTier.L0, modality=Modality.TEXT))
        pt.register(make_page("p2", tier=StorageTier.L1, modality=Modality.IMAGE))
        pt.mark_dirty("p1")

        stats = pt.get_stats()
        assert isinstance(stats, PageTableStats)
        assert stats.total_pages == 2
        assert stats.dirty_pages == 1
        assert stats.pages_by_tier[StorageTier.L0] == 1
        assert stats.pages_by_tier[StorageTier.L1] == 1
        assert stats.pages_by_modality[Modality.TEXT] == 1
        assert stats.pages_by_modality[Modality.IMAGE] == 1

    def test_get_stats_working_set_size(self):
        pt = PageTable()
        pt.register(make_page("p1", tier=StorageTier.L0))
        pt.register(make_page("p2", tier=StorageTier.L1))
        pt.register(make_page("p3", tier=StorageTier.L2))
        stats = pt.get_stats()
        assert stats.working_set_size == 2

    def test_get_total_tokens_all_tiers(self):
        pt = PageTable()
        pt.register(make_page("p1", size_tokens=100, tier=StorageTier.L0))
        pt.register(make_page("p2", size_tokens=200, tier=StorageTier.L1))
        total = pt.get_total_tokens()
        assert total == 300

    def test_get_total_tokens_specific_tiers(self):
        pt = PageTable()
        pt.register(make_page("p1", size_tokens=100, tier=StorageTier.L0))
        pt.register(make_page("p2", size_tokens=200, tier=StorageTier.L1))
        pt.register(make_page("p3", size_tokens=300, tier=StorageTier.L2))
        total = pt.get_total_tokens(tiers=[StorageTier.L0, StorageTier.L1])
        assert total == 300

    def test_get_total_tokens_with_none_size(self):
        pt = PageTable()
        page = make_page("p1", tier=StorageTier.L0)
        entry = pt.register(page)
        # If size_tokens is 0 or falsy, it shouldn't add
        entry.size_tokens = 0
        total = pt.get_total_tokens()
        assert total == 0


# ===================================================================
# 2. TLB tests
# ===================================================================


class TestPageTLB:
    """Tests for PageTLB."""

    def test_len_empty(self):
        tlb = PageTLB()
        assert len(tlb) == 0

    def test_len_with_entries(self):
        tlb = PageTLB()
        tlb.insert(make_entry("p1"))
        tlb.insert(make_entry("p2"))
        assert len(tlb) == 2

    def test_contains_present(self):
        tlb = PageTLB()
        tlb.insert(make_entry("p1"))
        assert "p1" in tlb

    def test_contains_absent(self):
        tlb = PageTLB()
        assert "p1" not in tlb

    def test_lookup_hit(self):
        tlb = PageTLB()
        tlb.insert(make_entry("p1"))
        entry = tlb.lookup("p1")
        assert entry is not None
        assert entry.page_id == "p1"
        assert tlb.hits == 1

    def test_lookup_miss(self):
        tlb = PageTLB()
        entry = tlb.lookup("missing")
        assert entry is None
        assert tlb.misses == 1

    def test_insert_new(self):
        tlb = PageTLB()
        tlb.insert(make_entry("p1"))
        assert "p1" in tlb

    def test_insert_update_existing(self):
        tlb = PageTLB()
        entry1 = make_entry("p1", tier=StorageTier.L0)
        entry2 = make_entry("p1", tier=StorageTier.L2)
        tlb.insert(entry1)
        tlb.insert(entry2)
        assert len(tlb) == 1
        result = tlb.lookup("p1")
        assert result.tier == StorageTier.L2

    def test_eviction_on_overflow(self):
        tlb = PageTLB(max_entries=2)
        tlb.insert(make_entry("p1"))
        tlb.insert(make_entry("p2"))
        tlb.insert(make_entry("p3"))
        assert len(tlb) == 2
        # p1 should have been evicted (LRU)
        assert "p1" not in tlb
        assert "p2" in tlb
        assert "p3" in tlb

    def test_eviction_respects_lru_order(self):
        tlb = PageTLB(max_entries=2)
        tlb.insert(make_entry("p1"))
        tlb.insert(make_entry("p2"))
        # Access p1 to make it most recently used
        tlb.lookup("p1")
        # Now insert p3; p2 should be evicted (LRU)
        tlb.insert(make_entry("p3"))
        assert "p2" not in tlb
        assert "p1" in tlb
        assert "p3" in tlb

    def test_invalidate_present(self):
        tlb = PageTLB()
        tlb.insert(make_entry("p1"))
        result = tlb.invalidate("p1")
        assert result is True
        assert "p1" not in tlb

    def test_invalidate_absent(self):
        tlb = PageTLB()
        result = tlb.invalidate("missing")
        assert result is False

    def test_invalidate_tier(self):
        tlb = PageTLB()
        tlb.insert(make_entry("p1", tier=StorageTier.L0))
        tlb.insert(make_entry("p2", tier=StorageTier.L0))
        tlb.insert(make_entry("p3", tier=StorageTier.L1))
        count = tlb.invalidate_tier(StorageTier.L0)
        assert count == 2
        assert len(tlb) == 1
        assert "p3" in tlb

    def test_flush(self):
        tlb = PageTLB()
        tlb.insert(make_entry("p1"))
        tlb.insert(make_entry("p2"))
        count = tlb.flush()
        assert count == 2
        assert len(tlb) == 0

    def test_get_all(self):
        tlb = PageTLB()
        tlb.insert(make_entry("p1"))
        tlb.insert(make_entry("p2"))
        all_entries = tlb.get_all()
        assert len(all_entries) == 2

    def test_hit_rate_no_lookups(self):
        tlb = PageTLB()
        assert tlb.hit_rate == 0.0

    def test_hit_rate_with_lookups(self):
        tlb = PageTLB()
        tlb.insert(make_entry("p1"))
        tlb.lookup("p1")  # hit
        tlb.lookup("p1")  # hit
        tlb.lookup("missing")  # miss
        assert tlb.hit_rate == pytest.approx(2.0 / 3.0)

    def test_reset_stats(self):
        tlb = PageTLB()
        tlb.insert(make_entry("p1"))
        tlb.lookup("p1")
        tlb.lookup("missing")
        tlb.reset_stats()
        assert tlb.hits == 0
        assert tlb.misses == 0

    def test_get_stats(self):
        tlb = PageTLB(max_entries=100)
        tlb.insert(make_entry("p1"))
        tlb.lookup("p1")
        tlb.lookup("missing")

        stats = tlb.get_stats()
        assert stats.size == 1
        assert stats.max_size == 100
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_rate == pytest.approx(0.5)
        assert stats.utilization == pytest.approx(1.0 / 100.0)

    def test_evict_lru_empty(self):
        tlb = PageTLB()
        result = tlb._evict_lru()
        assert result is None


class TestTLBWithPageTable:
    """Tests for TLBWithPageTable."""

    def test_lookup_tlb_hit(self):
        pt = PageTable()
        pt.register(make_page("p1"))
        tlb = PageTLB()
        tlb.insert(pt.lookup("p1"))

        combo = TLBWithPageTable(pt, tlb)
        entry = combo.lookup("p1")
        assert entry is not None
        assert tlb.hits == 1

    def test_lookup_tlb_miss_pt_hit(self):
        pt = PageTable()
        pt.register(make_page("p1"))
        tlb = PageTLB()

        combo = TLBWithPageTable(pt, tlb)
        entry = combo.lookup("p1")
        assert entry is not None
        assert entry.page_id == "p1"
        assert tlb.misses == 1
        # Entry should now be in TLB
        assert "p1" in tlb

    def test_lookup_both_miss(self):
        pt = PageTable()
        tlb = PageTLB()
        combo = TLBWithPageTable(pt, tlb)
        entry = combo.lookup("missing")
        assert entry is None
        assert tlb.misses == 1

    def test_default_tlb_created(self):
        pt = PageTable()
        combo = TLBWithPageTable(pt)
        assert combo.tlb is not None
        assert isinstance(combo.tlb, PageTLB)

    def test_register(self):
        pt = PageTable()
        combo = TLBWithPageTable(pt)
        page = make_page("p1")
        entry = combo.register(page)
        assert entry.page_id == "p1"
        assert "p1" in pt
        assert "p1" in combo.tlb

    def test_update_location(self):
        pt = PageTable()
        combo = TLBWithPageTable(pt)
        combo.register(make_page("p1", tier=StorageTier.L0))
        # Put in TLB
        combo.lookup("p1")
        assert "p1" in combo.tlb

        result = combo.update_location("p1", tier=StorageTier.L2)
        assert result is True
        # TLB entry should be invalidated
        assert "p1" not in combo.tlb

    def test_update_location_not_found(self):
        pt = PageTable()
        combo = TLBWithPageTable(pt)
        result = combo.update_location("missing", tier=StorageTier.L2)
        assert result is False

    def test_mark_dirty(self):
        pt = PageTable()
        combo = TLBWithPageTable(pt)
        combo.register(make_page("p1"))
        combo.lookup("p1")
        assert "p1" in combo.tlb

        result = combo.mark_dirty("p1")
        assert result is True
        assert "p1" not in combo.tlb  # invalidated

    def test_mark_dirty_not_found(self):
        pt = PageTable()
        combo = TLBWithPageTable(pt)
        result = combo.mark_dirty("missing")
        assert result is False

    def test_remove(self):
        pt = PageTable()
        combo = TLBWithPageTable(pt)
        combo.register(make_page("p1"))
        combo.lookup("p1")

        removed = combo.remove("p1")
        assert removed is not None
        assert "p1" not in pt
        assert "p1" not in combo.tlb

    def test_remove_not_found(self):
        pt = PageTable()
        combo = TLBWithPageTable(pt)
        removed = combo.remove("missing")
        assert removed is None

    def test_get_stats(self):
        pt = PageTable()
        combo = TLBWithPageTable(pt)
        combo.register(make_page("p1"))
        stats = combo.get_stats()
        assert stats.page_table.total_pages == 1
        assert stats.tlb.size == 1


# ===================================================================
# 3. WorkingSet tests
# ===================================================================


class TestPinnedSet:
    """Tests for PinnedSet."""

    def test_pin_and_is_pinned(self):
        ps = PinnedSet()
        ps.pin("p1")
        assert ps.is_pinned("p1") is True

    def test_unpin(self):
        ps = PinnedSet()
        ps.pin("p1")
        ps.unpin("p1")
        assert ps.is_pinned("p1") is False

    def test_auto_pin(self):
        ps = PinnedSet()
        ps.auto_pin("p1")
        assert ps.is_pinned("p1") is True

    def test_unpin_removes_auto_pin_too(self):
        ps = PinnedSet()
        ps.auto_pin("p1")
        ps.unpin("p1")
        assert ps.is_pinned("p1") is False

    def test_clear_auto_pins(self):
        ps = PinnedSet()
        ps.auto_pin("p1")
        ps.auto_pin("p2")
        ps.pin("p3")
        ps.clear_auto_pins()
        assert ps.is_pinned("p1") is False
        assert ps.is_pinned("p2") is False
        assert ps.is_pinned("p3") is True  # explicit pin survives

    def test_get_all_pinned(self):
        ps = PinnedSet()
        ps.pin("p1")
        ps.auto_pin("p2")
        all_pinned = ps.get_all_pinned()
        assert all_pinned == {"p1", "p2"}

    def test_count(self):
        ps = PinnedSet()
        ps.pin("p1")
        ps.auto_pin("p2")
        assert ps.count() == 2

    def test_count_no_duplicates(self):
        ps = PinnedSet()
        ps.pin("p1")
        ps.auto_pin("p1")
        assert ps.count() == 1


class TestAntiThrashPolicy:
    """Tests for AntiThrashPolicy."""

    def test_record_fault_and_can_evict(self):
        policy = AntiThrashPolicy(fault_protection_turns=2)
        policy.record_fault("p1", turn=5)
        # Within protection window
        assert policy.can_evict("p1", current_turn=5) is False
        assert policy.can_evict("p1", current_turn=6) is False
        # Outside protection window
        assert policy.can_evict("p1", current_turn=7) is True

    def test_record_eviction_and_cooldown(self):
        policy = AntiThrashPolicy(eviction_cooldown_turns=3)
        policy.record_eviction("p1", turn=5)
        assert policy.can_evict("p1", current_turn=5) is False
        assert policy.can_evict("p1", current_turn=7) is False
        assert policy.can_evict("p1", current_turn=8) is True

    def test_can_evict_no_history(self):
        policy = AntiThrashPolicy()
        assert policy.can_evict("p1", current_turn=100) is True

    def test_get_eviction_penalty_recently_faulted(self):
        policy = AntiThrashPolicy(fault_protection_turns=4)
        policy.record_fault("p1", turn=10)
        penalty = policy.get_eviction_penalty("p1", current_turn=10)
        assert penalty > 0.0

    def test_get_eviction_penalty_recently_evicted(self):
        policy = AntiThrashPolicy(eviction_cooldown_turns=4)
        policy.record_eviction("p1", turn=10)
        penalty = policy.get_eviction_penalty("p1", current_turn=11)
        assert penalty > 0.0

    def test_get_eviction_penalty_no_history(self):
        policy = AntiThrashPolicy()
        penalty = policy.get_eviction_penalty("p1", current_turn=100)
        assert penalty == 0.0

    def test_get_eviction_penalty_capped_at_one(self):
        policy = AntiThrashPolicy(fault_protection_turns=4, eviction_cooldown_turns=4)
        policy.record_fault("p1", turn=10)
        policy.record_eviction("p1", turn=10)
        penalty = policy.get_eviction_penalty("p1", current_turn=10)
        assert penalty <= 1.0

    def test_cleanup_old_history(self):
        policy = AntiThrashPolicy()
        policy.record_fault("p1", turn=1)
        policy.record_eviction("p2", turn=2)
        policy.cleanup_old_history(current_turn=100, max_age=20)
        # Both should be cleaned up as they are very old
        assert policy.can_evict("p1", current_turn=100) is True
        assert policy.can_evict("p2", current_turn=100) is True


class TestWorkingSetManager:
    """Tests for WorkingSetManager."""

    def test_add_to_l0(self):
        wsm = WorkingSetManager()
        page = make_page("p1", content="hello")
        result = wsm.add_to_l0(page)
        assert result is True
        assert wsm.is_in_l0("p1") is True

    def test_add_to_l0_updates_tier(self):
        wsm = WorkingSetManager()
        page = make_page("p1", tier=StorageTier.L2)
        wsm.add_to_l0(page)
        assert page.storage_tier == StorageTier.L0

    def test_add_to_l0_removes_from_l1(self):
        wsm = WorkingSetManager()
        page = make_page("p1")
        wsm.add_to_l1(page)
        assert wsm.is_in_l1("p1")
        wsm.add_to_l0(page)
        assert wsm.is_in_l0("p1")
        assert not wsm.is_in_l1("p1")

    def test_add_to_l0_rejects_when_over_budget(self):
        # With max_l0_tokens=100 and reserved=90, only 10 tokens available
        # add_to_l0 does NOT auto-evict — MemoryManager handles eviction
        config = WorkingSetConfig(max_l0_tokens=100, reserved_tokens=90)
        wsm = WorkingSetManager(config=config)
        page = make_page("p1", size_tokens=50)
        result = wsm.add_to_l0(page)
        assert result is False  # 50 > 10 available tokens

    def test_add_to_l0_duplicate(self):
        wsm = WorkingSetManager()
        page = make_page("p1", content="hello")
        wsm.add_to_l0(page)
        # Adding same page again should not duplicate
        wsm.add_to_l0(page)
        assert wsm.l0_pages.count("p1") == 1

    def test_add_to_l1(self):
        wsm = WorkingSetManager()
        page = make_page("p1")
        result = wsm.add_to_l1(page)
        assert result is True
        assert wsm.is_in_l1("p1")

    def test_add_to_l1_full(self):
        config = WorkingSetConfig(max_l1_pages=1)
        wsm = WorkingSetManager(config=config)
        wsm.add_to_l1(make_page("p1"))
        result = wsm.add_to_l1(make_page("p2"))
        assert result is False

    def test_add_to_l1_demotes_from_l0(self):
        wsm = WorkingSetManager()
        page = make_page("p1", content="hello", size_tokens=50)
        wsm.add_to_l0(page)
        assert wsm.is_in_l0("p1")
        tokens_before = wsm.tokens_used

        wsm.add_to_l1(page)
        assert wsm.is_in_l1("p1")
        assert not wsm.is_in_l0("p1")
        # Budget should have been reduced
        assert wsm.tokens_used < tokens_before

    def test_remove(self):
        wsm = WorkingSetManager()
        page = make_page("p1")
        wsm.add_to_l1(page)
        removed = wsm.remove("p1")
        assert removed is not None
        assert not wsm.is_in_l1("p1")

    def test_remove_from_l0(self):
        wsm = WorkingSetManager()
        page = make_page("p1")
        wsm.add_to_l0(page)
        wsm.remove("p1")
        assert not wsm.is_in_l0("p1")

    def test_remove_not_found(self):
        wsm = WorkingSetManager()
        removed = wsm.remove("missing")
        assert removed is None

    def test_remove_from_l0_with_page(self):
        wsm = WorkingSetManager()
        page = make_page("p1", size_tokens=50)
        wsm.add_to_l0(page)
        result = wsm.remove_from_l0("p1", page)
        assert result is True
        assert not wsm.is_in_l0("p1")

    def test_remove_from_l0_not_in_l0(self):
        wsm = WorkingSetManager()
        page = make_page("p1")
        result = wsm.remove_from_l0("p1", page)
        assert result is False

    def test_promote_to_l0(self):
        wsm = WorkingSetManager()
        page = make_page("p1")
        wsm.add_to_l1(page)
        result = wsm.promote_to_l0(page)
        assert result is True
        assert wsm.is_in_l0("p1")

    def test_promote_to_l0_not_in_l1(self):
        wsm = WorkingSetManager()
        page = make_page("p1")
        result = wsm.promote_to_l0(page)
        assert result is False

    def test_demote_to_l1(self):
        wsm = WorkingSetManager()
        page = make_page("p1", content="hello")
        wsm.add_to_l0(page)
        result = wsm.demote_to_l1(page)
        assert result is True
        assert wsm.is_in_l1("p1")
        assert not wsm.is_in_l0("p1")

    def test_demote_to_l1_not_in_l0(self):
        wsm = WorkingSetManager()
        page = make_page("p1")
        result = wsm.demote_to_l1(page)
        assert result is False

    def test_get_page_from_l1(self):
        wsm = WorkingSetManager()
        page = make_page("p1", content="hello")
        wsm.add_to_l1(page)
        retrieved = wsm.get_page("p1")
        assert retrieved is not None
        assert retrieved.content == "hello"

    def test_get_page_not_found(self):
        wsm = WorkingSetManager()
        assert wsm.get_page("missing") is None

    def test_is_in_working_set(self):
        wsm = WorkingSetManager()
        page1 = make_page("p1")
        page2 = make_page("p2")
        wsm.add_to_l0(page1)
        wsm.add_to_l1(page2)
        assert wsm.is_in_working_set("p1") is True
        assert wsm.is_in_working_set("p2") is True
        assert wsm.is_in_working_set("missing") is False

    def test_len(self):
        wsm = WorkingSetManager()
        wsm.add_to_l0(make_page("p1"))
        wsm.add_to_l1(make_page("p2"))
        assert len(wsm) == 2

    def test_l0_count_and_l1_count(self):
        wsm = WorkingSetManager()
        wsm.add_to_l0(make_page("p1"))
        wsm.add_to_l1(make_page("p2"))
        wsm.add_to_l1(make_page("p3"))
        assert wsm.l0_count == 1
        assert wsm.l1_count == 2

    def test_utilization(self):
        wsm = WorkingSetManager()
        assert wsm.utilization == 0.0

    def test_tokens_used_and_available(self):
        wsm = WorkingSetManager()
        assert wsm.tokens_used == 0
        assert wsm.tokens_available > 0

    def test_needs_eviction(self):
        wsm = WorkingSetManager()
        assert wsm.needs_eviction() is False

    def test_can_fit(self):
        wsm = WorkingSetManager()
        assert wsm.can_fit(100) is True

    def test_get_eviction_candidates_l0(self):
        wsm = WorkingSetManager()
        wsm.add_to_l0(make_page("p1"))
        wsm.add_to_l0(make_page("p2"))
        wsm.add_to_l0(make_page("p3"))
        candidates = wsm.get_eviction_candidates(from_tier=StorageTier.L0)
        assert len(candidates) == 3
        # Each candidate is (page_id, score) tuple
        assert all(isinstance(c, tuple) and len(c) == 2 for c in candidates)

    def test_get_eviction_candidates_l0_excludes_pinned(self):
        wsm = WorkingSetManager()
        wsm.add_to_l0(make_page("p1"))
        wsm.add_to_l0(make_page("p2"))
        wsm.pin_page("p1")
        candidates = wsm.get_eviction_candidates(from_tier=StorageTier.L0)
        ids = [c[0] for c in candidates]
        assert "p1" not in ids
        assert "p2" in ids

    def test_get_eviction_candidates_l1(self):
        wsm = WorkingSetManager()
        wsm.add_to_l1(make_page("p1"))
        wsm.add_to_l1(make_page("p2"))
        candidates = wsm.get_eviction_candidates(from_tier=StorageTier.L1)
        assert len(candidates) == 2

    def test_get_eviction_candidates_l1_excludes_pinned_page_flag(self):
        wsm = WorkingSetManager()
        page = make_page("p1", pinned=True)
        wsm.add_to_l1(page)
        wsm.add_to_l1(make_page("p2"))
        candidates = wsm.get_eviction_candidates(from_tier=StorageTier.L1)
        ids = [c[0] for c in candidates]
        assert "p1" not in ids

    def test_get_eviction_candidates_l1_excludes_anti_thrash(self):
        wsm = WorkingSetManager()
        wsm.add_to_l1(make_page("p1"))
        wsm.add_to_l1(make_page("p2"))
        wsm.anti_thrash.record_fault("p1", turn=wsm.current_turn)
        candidates = wsm.get_eviction_candidates(from_tier=StorageTier.L1)
        ids = [c[0] for c in candidates]
        assert "p1" not in ids

    def test_new_turn(self):
        wsm = WorkingSetManager()
        wsm.new_turn()
        assert wsm.current_turn == 1

    def test_new_turn_cleanup_on_interval(self):
        wsm = WorkingSetManager()
        wsm.anti_thrash.record_fault("p1", turn=0)
        # Advance to turn 10 to trigger cleanup
        for _ in range(10):
            wsm.new_turn()
        assert wsm.current_turn == 10

    def test_pin_unpin_is_pinned(self):
        wsm = WorkingSetManager()
        wsm.pin_page("p1")
        assert wsm.is_pinned("p1") is True
        wsm.unpin_page("p1")
        assert wsm.is_pinned("p1") is False

    def test_record_fault_and_eviction(self):
        wsm = WorkingSetManager()
        wsm.record_fault("p1")
        wsm.record_eviction("p2")
        # Just verify no errors; anti-thrash policy has been updated

    def test_calculate_eviction_target(self):
        wsm = WorkingSetManager()
        target = wsm.calculate_eviction_target(tokens_needed=1000)
        # Should return max(0, used - target_util * budget + tokens_needed)
        assert isinstance(target, int)

    def test_set_importance(self):
        wsm = WorkingSetManager()
        wsm.set_importance("p1", 0.9)
        assert wsm.importance_overrides["p1"] == 0.9

    def test_set_importance_clamped(self):
        wsm = WorkingSetManager()
        wsm.set_importance("p1", 5.0)
        assert wsm.importance_overrides["p1"] == 1.0
        wsm.set_importance("p2", -1.0)
        assert wsm.importance_overrides["p2"] == 0.0

    def test_clear_importance(self):
        wsm = WorkingSetManager()
        wsm.set_importance("p1", 0.9)
        wsm.clear_importance("p1")
        assert "p1" not in wsm.importance_overrides

    def test_clear_importance_missing(self):
        wsm = WorkingSetManager()
        wsm.clear_importance("missing")  # Should not raise

    def test_get_l0_page_ids(self):
        wsm = WorkingSetManager()
        wsm.add_to_l0(make_page("p1"))
        wsm.add_to_l0(make_page("p2"))
        ids = wsm.get_l0_page_ids()
        assert ids == ["p1", "p2"]

    def test_get_l1_pages(self):
        wsm = WorkingSetManager()
        wsm.add_to_l1(make_page("p1"))
        wsm.add_to_l1(make_page("p2"))
        pages = wsm.get_l1_pages()
        assert len(pages) == 2

    def test_get_stats(self):
        wsm = WorkingSetManager()
        wsm.add_to_l0(make_page("p1", size_tokens=50))
        wsm.add_to_l1(make_page("p2"))
        stats = wsm.get_stats()
        assert stats.l0_pages == 1
        assert stats.l1_pages == 1
        assert stats.total_pages == 2
        assert isinstance(stats.utilization, float)
        assert isinstance(stats.needs_eviction, bool)

    def test_clear(self):
        wsm = WorkingSetManager()
        wsm.add_to_l0(make_page("p1", size_tokens=50))
        wsm.add_to_l1(make_page("p2"))
        wsm.set_importance("p1", 0.9)
        wsm.clear()
        assert len(wsm) == 0
        assert wsm.tokens_used == 0
        assert len(wsm.importance_overrides) == 0


# ===================================================================
# 4. ContextPacker tests
# ===================================================================


class TestContextPacker:
    """Tests for ContextPacker."""

    def test_pack_empty(self):
        packer = ContextPacker()
        result = packer.pack([])
        assert result.content == ""
        assert result.tokens_est == 0
        assert result.pages_included == []

    def test_pack_text_pages(self):
        packer = ContextPacker()
        pages = [
            make_page("p1", content="Hello world"),
            make_page("p2", content="Goodbye world"),
        ]
        result = packer.pack(pages)
        assert "p1" in result.content
        assert "p2" in result.content
        assert "Hello world" in result.content
        assert result.pages_included == ["p1", "p2"]
        assert result.tokens_est > 0

    def test_pack_with_budget_truncation(self):
        packer = ContextPacker()
        pages = [
            make_page("p1", content="Short"),
            make_page("p2", content="A" * 10000),
        ]
        # Very tight budget: first page should fit, second should be truncated
        result = packer.pack(pages, token_budget=200)
        assert "p1" in result.pages_included

    def test_pack_with_budget_omit(self):
        packer = ContextPacker(config=ContextPackerConfig(max_text_length=100))
        pages = [
            make_page("p1", content="A" * 50, size_tokens=100),
            make_page("p2", content="B" * 50, size_tokens=100),
        ]
        # Set budget so only first page fits (formatted line ~55 chars / 4 = ~13 tokens)
        result = packer.pack(pages, token_budget=15)
        assert "p2" in result.pages_omitted

    def test_pack_budget_omit_non_text(self):
        packer = ContextPacker()
        pages = [
            make_page("p1", content="A" * 50, size_tokens=100),
            make_page(
                "p2",
                content="some image",
                modality=Modality.IMAGE,
                size_tokens=200,
            ),
        ]
        # Budget only fits p1 (formatted line ~55 chars / 4 = ~13 tokens)
        result = packer.pack(pages, token_budget=15)
        assert "p2" in result.pages_omitted

    def test_pack_budget_truncate_text_remaining_small(self):
        """When remaining budget is <50 tokens for text, omit instead of truncate."""
        packer = ContextPacker()
        pages = [
            make_page("p1", content="A" * 400, size_tokens=100),
            make_page("p2", content="B" * 400, size_tokens=100),
        ]
        # After p1, only ~10 tokens left
        result = packer.pack(pages, token_budget=110)
        # p2 should be omitted (remaining < 50)
        assert "p2" in result.pages_omitted or "p2" in result.pages_truncated

    def test_format_text_page(self):
        packer = ContextPacker()
        page = make_page("msg_1", content="Hello there")
        result = packer._format_page(page)
        assert "msg_1" in result.content
        assert "Hello there" in result.content

    def test_format_text_with_role_user(self):
        packer = ContextPacker()
        page = make_page("msg_1", content="Hello", metadata={"role": "user"})
        result = packer._format_page(page)
        assert result.content.startswith("U ")

    def test_format_text_with_role_assistant(self):
        packer = ContextPacker()
        page = make_page("msg_1", content="Hi", metadata={"role": "assistant"})
        result = packer._format_page(page)
        assert result.content.startswith("A ")

    def test_format_text_with_role_tool(self):
        packer = ContextPacker()
        page = make_page("msg_1", content="result", metadata={"role": "tool"})
        result = packer._format_page(page)
        assert result.content.startswith("T ")

    def test_format_text_with_summary_page_type(self):
        packer = ContextPacker()
        page = make_page("summary_seg_01", content="Key points")
        result = packer._format_page(page)
        assert result.content.startswith("S ")

    def test_format_text_with_summary_in_metadata(self):
        packer = ContextPacker()
        page = make_page("msg_1", content="Key points", metadata={"type": "summary"})
        result = packer._format_page(page)
        assert result.content.startswith("S ")

    def test_format_text_without_page_ids(self):
        packer = ContextPacker(config=ContextPackerConfig(include_page_ids=False))
        page = make_page("p1", content="Hello")
        result = packer._format_page(page)
        assert "(p1)" not in result.content

    def test_format_text_with_max_text_length(self):
        packer = ContextPacker(config=ContextPackerConfig(max_text_length=10))
        page = make_page("p1", content="A" * 100)
        result = packer._format_page(page)
        assert "..." in result.content

    def test_format_text_with_max_tokens_truncation(self):
        packer = ContextPacker()
        page = make_page("p1", content="A" * 1000)
        result = packer._format_page(page, max_tokens=10)
        # max_tokens=10 -> max_chars=40
        assert "..." in result.content

    def test_format_image_page(self):
        packer = ContextPacker()
        page = make_page(
            "img_1",
            content="architecture diagram",
            modality=Modality.IMAGE,
            caption="Architecture diagram",
            dimensions=(1200, 800),
        )
        result = packer._format_page(page)
        assert "IMAGE" in result.content
        assert "Architecture diagram" in result.content
        assert "1200x800" in result.content
        assert "img_1" in result.content

    def test_format_image_no_caption_uses_content(self):
        packer = ContextPacker()
        page = make_page(
            "img_1",
            content="fallback desc",
            modality=Modality.IMAGE,
        )
        result = packer._format_page(page)
        assert "fallback desc" in result.content

    def test_format_image_no_dimensions(self):
        packer = ContextPacker()
        page = make_page(
            "img_1",
            content=None,
            modality=Modality.IMAGE,
        )
        result = packer._format_page(page)
        assert "IMAGE" in result.content

    def test_format_image_without_page_ids(self):
        packer = ContextPacker(config=ContextPackerConfig(include_page_ids=False))
        page = make_page("img_1", content=None, modality=Modality.IMAGE)
        result = packer._format_page(page)
        assert "(img_1)" not in result.content

    def test_format_audio_page(self):
        packer = ContextPacker()
        page = make_page(
            "audio_1",
            content=None,
            modality=Modality.AUDIO,
            duration_seconds=342.0,
            transcript="So the key insight is...",
        )
        result = packer._format_page(page)
        assert "AUDIO" in result.content
        assert "5:42" in result.content
        assert "So the key insight is..." in result.content

    def test_format_audio_with_content_fallback(self):
        packer = ContextPacker()
        page = make_page(
            "audio_1",
            content="audio transcript content",
            modality=Modality.AUDIO,
        )
        result = packer._format_page(page)
        assert "audio transcript content" in result.content

    def test_format_audio_long_transcript_truncated(self):
        packer = ContextPacker()
        page = make_page(
            "audio_1",
            content=None,
            modality=Modality.AUDIO,
            transcript="A" * 300,
        )
        result = packer._format_page(page)
        assert "..." in result.content

    def test_format_audio_long_content_truncated(self):
        packer = ContextPacker()
        page = make_page(
            "audio_1",
            content="B" * 300,
            modality=Modality.AUDIO,
        )
        result = packer._format_page(page)
        assert "..." in result.content

    def test_format_audio_without_page_ids(self):
        packer = ContextPacker(config=ContextPackerConfig(include_page_ids=False))
        page = make_page("audio_1", content=None, modality=Modality.AUDIO)
        result = packer._format_page(page)
        assert "(audio_1)" not in result.content

    def test_format_video_page(self):
        packer = ContextPacker()
        page = make_page(
            "video_1",
            content=None,
            modality=Modality.VIDEO,
            duration_seconds=750.0,
            caption="system walkthrough",
            metadata={"scene_count": 8, "topic": "demo"},
        )
        result = packer._format_page(page)
        assert "VIDEO" in result.content
        assert "12:30" in result.content
        assert "8 scenes" in result.content
        assert "demo" in result.content

    def test_format_video_with_caption_no_topic(self):
        packer = ContextPacker()
        page = make_page(
            "video_1",
            content=None,
            modality=Modality.VIDEO,
            caption="my video",
            metadata={},
        )
        result = packer._format_page(page)
        assert "my video" in result.content

    def test_format_video_no_metadata(self):
        packer = ContextPacker()
        page = make_page(
            "video_1",
            content=None,
            modality=Modality.VIDEO,
        )
        result = packer._format_page(page)
        assert "VIDEO" in result.content

    def test_format_video_without_page_ids(self):
        packer = ContextPacker(config=ContextPackerConfig(include_page_ids=False))
        page = make_page("video_1", content=None, modality=Modality.VIDEO)
        result = packer._format_page(page)
        assert "(video_1)" not in result.content

    def test_format_structured_page(self):
        packer = ContextPacker()
        page = make_page(
            "data_1",
            content={"key": "value", "count": 42},
            modality=Modality.STRUCTURED,
        )
        result = packer._format_page(page)
        assert "J " in result.content
        assert "data_1" in result.content
        assert "key" in result.content

    def test_format_structured_page_none_content(self):
        packer = ContextPacker()
        page = make_page(
            "data_1",
            content=None,
            modality=Modality.STRUCTURED,
        )
        result = packer._format_page(page)
        assert "{}" in result.content

    def test_format_structured_with_max_tokens(self):
        packer = ContextPacker()
        page = make_page(
            "data_1",
            content={"data": "A" * 1000},
            modality=Modality.STRUCTURED,
        )
        result = packer._format_page(page, max_tokens=10)
        assert "..." in result.content

    def test_format_structured_string_content(self):
        packer = ContextPacker()
        page = make_page(
            "data_1",
            content="plain string",
            modality=Modality.STRUCTURED,
        )
        result = packer._format_page(page)
        assert "plain string" in result.content

    def test_format_structured_without_page_ids(self):
        packer = ContextPacker(config=ContextPackerConfig(include_page_ids=False))
        page = make_page(
            "data_1",
            content={"k": "v"},
            modality=Modality.STRUCTURED,
        )
        result = packer._format_page(page)
        assert "(data_1)" not in result.content

    def test_format_generic_fallback(self):
        """Pages with an unexpected modality should get generic formatting."""
        packer = ContextPacker()
        page = make_page("p1", content="some data")
        # Simulate an unknown modality by calling _format_generic directly
        result = packer._format_generic(page)
        assert "?" in result.content
        assert "some data" in result.content

    def test_format_generic_no_content(self):
        packer = ContextPacker()
        page = make_page("p1", content=None)
        result = packer._format_generic(page)
        assert "[no content]" in result.content

    def test_format_generic_without_page_ids(self):
        packer = ContextPacker(config=ContextPackerConfig(include_page_ids=False))
        page = make_page("p1", content="data")
        result = packer._format_generic(page)
        assert "(p1)" not in result.content

    def test_pack_with_wrapper(self):
        packer = ContextPacker()
        pages = [make_page("p1", content="Hello")]
        result = packer.pack_with_wrapper(pages)
        assert "<VM:CONTEXT>" in result.content
        assert "</VM:CONTEXT>" in result.content
        assert result.tokens_est > 0


# ===================================================================
# 5. vm_prompts tests
# ===================================================================


class TestVmPrompts:
    """Tests for vm_prompts module."""

    def test_vm_strict_prompt_exists(self):
        assert isinstance(VM_STRICT_PROMPT, str)
        assert len(VM_STRICT_PROMPT) > 0
        assert "STRICT" in VM_STRICT_PROMPT

    def test_vm_relaxed_prompt_exists(self):
        assert isinstance(VM_RELAXED_PROMPT, str)
        assert "virtual memory" in VM_RELAXED_PROMPT.lower()

    def test_vm_passive_prompt_exists(self):
        assert isinstance(VM_PASSIVE_PROMPT, str)
        assert "helpful assistant" in VM_PASSIVE_PROMPT.lower()

    def test_vm_prompts_map(self):
        assert VMMode.STRICT in VM_PROMPTS
        assert VMMode.RELAXED in VM_PROMPTS
        assert VMMode.PASSIVE in VM_PROMPTS

    def test_get_prompt_for_mode_strict(self):
        prompt = get_prompt_for_mode(VMMode.STRICT)
        assert prompt == VM_STRICT_PROMPT

    def test_get_prompt_for_mode_relaxed(self):
        prompt = get_prompt_for_mode(VMMode.RELAXED)
        assert prompt == VM_RELAXED_PROMPT

    def test_get_prompt_for_mode_passive(self):
        prompt = get_prompt_for_mode(VMMode.PASSIVE)
        assert prompt == VM_PASSIVE_PROMPT

    def test_vm_tool_definitions(self):
        assert isinstance(VM_TOOL_DEFINITIONS, list)
        assert len(VM_TOOL_DEFINITIONS) == 2
        names = [t.function.name for t in VM_TOOL_DEFINITIONS]
        assert "page_fault" in names
        assert "search_pages" in names

    def test_page_fault_tool(self):
        assert PAGE_FAULT_TOOL.function.name == "page_fault"
        assert "page_id" in PAGE_FAULT_TOOL.function.parameters.properties
        assert "page_id" in PAGE_FAULT_TOOL.function.parameters.required

    def test_search_pages_tool(self):
        assert SEARCH_PAGES_TOOL.function.name == "search_pages"
        assert "query" in SEARCH_PAGES_TOOL.function.parameters.properties

    def test_get_vm_tools_with_search(self):
        tools = get_vm_tools(include_search=True)
        assert len(tools) == 2

    def test_get_vm_tools_without_search(self):
        tools = get_vm_tools(include_search=False)
        assert len(tools) == 1
        assert tools[0].function.name == "page_fault"

    def test_get_vm_tools_as_dicts(self):
        dicts = get_vm_tools_as_dicts(include_search=True)
        assert isinstance(dicts, list)
        assert len(dicts) == 2
        assert all(isinstance(d, dict) for d in dicts)

    def test_get_vm_tools_as_dicts_without_search(self):
        dicts = get_vm_tools_as_dicts(include_search=False)
        assert len(dicts) == 1

    def test_vm_tools_legacy(self):
        assert isinstance(VM_TOOLS, list)
        assert len(VM_TOOLS) == 2

    def test_build_vm_developer_message_strict(self):
        msg = build_vm_developer_message(
            mode=VMMode.STRICT,
            manifest_json='{"pages": []}',
            context="U (msg_1): Hello",
            system_prompt="Be helpful",
            max_faults_per_turn=3,
        )
        assert "Be helpful" in msg
        assert "VM:RULES" in msg
        assert "VM:MANIFEST_JSON" in msg
        assert "VM:CONTEXT" in msg
        assert "(3)" in msg  # max_faults_per_turn injected

    def test_build_vm_developer_message_strict_without_system_prompt(self):
        msg = build_vm_developer_message(
            mode=VMMode.STRICT,
            manifest_json="{}",
            context="Hello",
        )
        assert "VM:RULES" in msg
        assert "VM:CONTEXT" in msg

    def test_build_vm_developer_message_relaxed(self):
        msg = build_vm_developer_message(
            mode=VMMode.RELAXED,
            manifest_json='{"pages": []}',
            context="Hello",
        )
        assert "VM:RULES" in msg
        assert "VM:MANIFEST_JSON" in msg
        assert "VM:CONTEXT" in msg

    def test_build_vm_developer_message_passive(self):
        msg = build_vm_developer_message(
            mode=VMMode.PASSIVE,
            manifest_json="{}",
            context="Hello",
        )
        assert "VM:CONTEXT" in msg
        # Passive mode should NOT include VM:RULES or VM:MANIFEST_JSON
        assert "VM:RULES" not in msg
        assert "VM:MANIFEST_JSON" not in msg

    def test_build_vm_developer_message_passive_with_system_prompt(self):
        msg = build_vm_developer_message(
            mode=VMMode.PASSIVE,
            manifest_json="{}",
            context="Hello",
            system_prompt="Custom system prompt",
        )
        assert "Custom system prompt" in msg


# ===================================================================
# 6. Manifest tests
# ===================================================================


class TestVMManifest:
    """Tests for VMManifest."""

    def test_to_json(self):
        manifest = VMManifest(session_id="sess_1")
        json_str = manifest.to_json()
        data = json.loads(json_str)
        assert data["session_id"] == "sess_1"

    def test_to_json_with_indent(self):
        manifest = VMManifest(session_id="sess_1")
        json_str = manifest.to_json(indent=2)
        assert "\n" in json_str

    def test_to_wrapped_json(self):
        manifest = VMManifest(session_id="sess_1")
        wrapped = manifest.to_wrapped_json()
        assert "<VM:MANIFEST_JSON>" in wrapped
        assert "</VM:MANIFEST_JSON>" in wrapped

    def test_manifest_with_working_set(self):
        ws = WorkingSetEntry(page_id="p1", modality="text", level=0, tokens_est=100)
        manifest = VMManifest(session_id="sess_1", working_set=[ws])
        assert len(manifest.working_set) == 1

    def test_manifest_with_available_pages(self):
        ap = AvailablePageEntry(page_id="p2", modality="image", tier="L3", hint="a picture")
        manifest = VMManifest(session_id="sess_1", available_pages=[ap])
        assert len(manifest.available_pages) == 1


class TestManifestBuilder:
    """Tests for ManifestBuilder."""

    def test_build_empty(self):
        builder = ManifestBuilder()
        pt = PageTable()
        manifest = builder.build(
            session_id="sess_1",
            page_table=pt,
            working_set_ids=[],
        )
        assert manifest.session_id == "sess_1"
        assert len(manifest.working_set) == 0
        assert len(manifest.available_pages) == 0

    def test_build_with_working_set(self):
        builder = ManifestBuilder()
        pt = PageTable()
        pt.register(make_page("p1", size_tokens=100))
        manifest = builder.build(
            session_id="sess_1",
            page_table=pt,
            working_set_ids=["p1"],
        )
        assert len(manifest.working_set) == 1
        assert manifest.working_set[0].page_id == "p1"
        assert manifest.working_set[0].tokens_est == 100

    def test_build_with_working_set_tokens_override(self):
        builder = ManifestBuilder()
        pt = PageTable()
        pt.register(make_page("p1", size_tokens=100))
        manifest = builder.build(
            session_id="sess_1",
            page_table=pt,
            working_set_ids=["p1"],
            working_set_tokens={"p1": 999},
        )
        assert manifest.working_set[0].tokens_est == 999

    def test_build_with_working_set_importance_override(self):
        builder = ManifestBuilder()
        pt = PageTable()
        pt.register(make_page("p1"))
        manifest = builder.build(
            session_id="sess_1",
            page_table=pt,
            working_set_ids=["p1"],
            working_set_importance={"p1": 0.9},
        )
        assert manifest.working_set[0].importance == 0.9

    def test_build_with_available_pages(self):
        builder = ManifestBuilder()
        pt = PageTable()
        pt.register(make_page("p1", tier=StorageTier.L0))
        pt.register(make_page("p2", tier=StorageTier.L2))
        manifest = builder.build(
            session_id="sess_1",
            page_table=pt,
            working_set_ids=["p1"],
        )
        assert len(manifest.available_pages) == 1
        assert manifest.available_pages[0].page_id == "p2"

    def test_build_with_hint_generator(self):
        builder = ManifestBuilder()
        pt = PageTable()
        pt.register(make_page("p1", tier=StorageTier.L0))
        pt.register(make_page("p2", tier=StorageTier.L2))

        def hint_gen(entry):
            return f"hint for {entry.page_id}"

        manifest = builder.build(
            session_id="sess_1",
            page_table=pt,
            working_set_ids=["p1"],
            hint_generator=hint_gen,
        )
        assert manifest.available_pages[0].hint == "hint for p2"

    def test_build_hint_generator_exception(self):
        builder = ManifestBuilder()
        pt = PageTable()
        pt.register(make_page("p1", tier=StorageTier.L0))
        pt.register(make_page("p2", tier=StorageTier.L2))

        def bad_hint_gen(entry):
            raise ValueError("oops")

        manifest = builder.build(
            session_id="sess_1",
            page_table=pt,
            working_set_ids=["p1"],
            hint_generator=bad_hint_gen,
        )
        assert manifest.available_pages[0].hint == ""

    def test_build_max_available_pages(self):
        builder = ManifestBuilder(max_available_pages=2)
        pt = PageTable()
        for i in range(10):
            pt.register(make_page(f"p{i}", tier=StorageTier.L2))
        manifest = builder.build(
            session_id="sess_1",
            page_table=pt,
            working_set_ids=[],
        )
        assert len(manifest.available_pages) == 2

    def test_build_custom_policies(self):
        builder = ManifestBuilder()
        pt = PageTable()
        policies = ManifestPolicies(max_faults_per_turn=5)
        manifest = builder.build(
            session_id="sess_1",
            page_table=pt,
            working_set_ids=[],
            policies=policies,
        )
        assert manifest.policies.max_faults_per_turn == 5

    def test_build_missing_working_set_id(self):
        """Working set ID not in page table should be skipped."""
        builder = ManifestBuilder()
        pt = PageTable()
        manifest = builder.build(
            session_id="sess_1",
            page_table=pt,
            working_set_ids=["nonexistent"],
        )
        assert len(manifest.working_set) == 0

    def test_build_from_pages(self):
        builder = ManifestBuilder()
        ws_page = make_page("p1", content="hello", size_tokens=100)
        avail_entry = make_entry("p2", tier=StorageTier.L3)

        manifest = builder.build_from_pages(
            session_id="sess_1",
            working_set_pages=[ws_page],
            available_entries=[avail_entry],
        )
        assert len(manifest.working_set) == 1
        assert len(manifest.available_pages) == 1

    def test_build_from_pages_with_hint_generator(self):
        builder = ManifestBuilder()
        avail_entry = make_entry("p2", tier=StorageTier.L3)

        manifest = builder.build_from_pages(
            session_id="sess_1",
            working_set_pages=[],
            available_entries=[avail_entry],
            hint_generator=lambda e: "a hint",
        )
        assert manifest.available_pages[0].hint == "a hint"

    def test_build_from_pages_hint_generator_exception(self):
        builder = ManifestBuilder()
        avail_entry = make_entry("p2", tier=StorageTier.L3)

        manifest = builder.build_from_pages(
            session_id="sess_1",
            working_set_pages=[],
            available_entries=[avail_entry],
            hint_generator=lambda e: 1 / 0,
        )
        assert manifest.available_pages[0].hint == ""

    def test_build_from_pages_skips_ws_ids_in_available(self):
        builder = ManifestBuilder()
        ws_page = make_page("p1")
        # Available entry with same id as working set should be skipped
        avail_entry = make_entry("p1", tier=StorageTier.L3)

        manifest = builder.build_from_pages(
            session_id="sess_1",
            working_set_pages=[ws_page],
            available_entries=[avail_entry],
        )
        assert len(manifest.available_pages) == 0


class TestGenerateSimpleHint:
    """Tests for generate_simple_hint."""

    def test_text_l2_full(self):
        entry = make_entry("p1", tier=StorageTier.L2, modality=Modality.TEXT)
        hint = generate_simple_hint(entry)
        assert HintType.RECENT in hint

    def test_image_l3(self):
        entry = make_entry("p1", tier=StorageTier.L3, modality=Modality.IMAGE)
        hint = generate_simple_hint(entry)
        assert "image" in hint
        assert HintType.STORED in hint

    def test_audio_l4(self):
        entry = make_entry("p1", tier=StorageTier.L4, modality=Modality.AUDIO)
        hint = generate_simple_hint(entry)
        assert "audio" in hint
        assert HintType.ARCHIVED in hint

    def test_abstract_compression(self):
        entry = make_entry(
            "p1",
            tier=StorageTier.L2,
            compression_level=CompressionLevel.ABSTRACT,
        )
        hint = generate_simple_hint(entry)
        assert HintType.SUMMARY in hint

    def test_reduced_compression(self):
        entry = make_entry(
            "p1",
            tier=StorageTier.L2,
            compression_level=CompressionLevel.REDUCED,
        )
        hint = generate_simple_hint(entry)
        assert HintType.EXCERPT in hint

    def test_text_l0_full_returns_content(self):
        entry = make_entry("p1", tier=StorageTier.L0, modality=Modality.TEXT)
        hint = generate_simple_hint(entry)
        assert hint == HintType.CONTENT

    def test_text_l1_full_returns_content(self):
        entry = make_entry("p1", tier=StorageTier.L1, modality=Modality.TEXT)
        hint = generate_simple_hint(entry)
        assert hint == HintType.CONTENT

    def test_video_l3_reduced(self):
        entry = make_entry(
            "p1",
            tier=StorageTier.L3,
            modality=Modality.VIDEO,
            compression_level=CompressionLevel.REDUCED,
        )
        hint = generate_simple_hint(entry)
        assert "video" in hint
        assert HintType.STORED in hint
        assert HintType.EXCERPT in hint


# ===================================================================
# 7. ArtifactsBridge tests
# ===================================================================


class TestInMemoryBackend:
    """Tests for InMemoryBackend."""

    async def test_store_and_load(self):
        backend = InMemoryBackend()
        page = make_page("p1", content="hello")
        artifact_id = await backend.store(page, StorageTier.L3)
        assert artifact_id.startswith("mem_L3_")

        loaded = await backend.load(artifact_id)
        assert loaded is not None
        assert loaded.page_id == "p1"
        assert loaded.content == "hello"

    async def test_load_not_found(self):
        backend = InMemoryBackend()
        result = await backend.load("nonexistent")
        assert result is None

    async def test_delete_existing(self):
        backend = InMemoryBackend()
        page = make_page("p1")
        artifact_id = await backend.store(page, StorageTier.L3)
        result = await backend.delete(artifact_id)
        assert result is True
        assert await backend.load(artifact_id) is None

    async def test_delete_not_found(self):
        backend = InMemoryBackend()
        result = await backend.delete("nonexistent")
        assert result is False

    async def test_clear(self):
        backend = InMemoryBackend()
        await backend.store(make_page("p1"), StorageTier.L3)
        await backend.store(make_page("p2"), StorageTier.L3)
        backend.clear()
        assert len(backend.pages) == 0
        assert backend.counter == 0

    async def test_counter_increments(self):
        backend = InMemoryBackend()
        id1 = await backend.store(make_page("p1"), StorageTier.L3)
        id2 = await backend.store(make_page("p2"), StorageTier.L3)
        assert id1 != id2
        assert backend.counter == 2


class TestArtifactsBridge:
    """Tests for ArtifactsBridge."""

    async def test_configure_in_memory(self):
        bridge = ArtifactsBridge()
        await bridge.configure(session_id="sess_1")
        assert bridge.is_persistent is False
        assert bridge.session_id == "sess_1"

    async def test_store_page_in_memory(self):
        bridge = ArtifactsBridge()
        await bridge.configure()
        page = make_page("p1", content="hello")
        artifact_id = await bridge.store_page(page, StorageTier.L3)
        assert artifact_id is not None

    async def test_load_page_in_memory(self):
        bridge = ArtifactsBridge()
        await bridge.configure()
        page = make_page("p1", content="hello")
        artifact_id = await bridge.store_page(page)
        loaded = await bridge.load_page(artifact_id)
        assert loaded is not None
        assert loaded.page_id == "p1"

    async def test_load_page_not_found(self):
        bridge = ArtifactsBridge()
        await bridge.configure()
        loaded = await bridge.load_page("nonexistent")
        assert loaded is None

    async def test_delete_page_in_memory(self):
        bridge = ArtifactsBridge()
        await bridge.configure()
        page = make_page("p1")
        artifact_id = await bridge.store_page(page)
        result = await bridge.delete_page(artifact_id)
        assert result is True

    async def test_delete_page_not_found(self):
        bridge = ArtifactsBridge()
        await bridge.configure()
        result = await bridge.delete_page("nonexistent")
        assert result is False

    async def test_store_page_not_configured(self):
        bridge = ArtifactsBridge()
        page = make_page("p1")
        with pytest.raises(StorageError, match="Storage error in"):
            await bridge.store_page(page)

    async def test_load_page_not_configured(self):
        bridge = ArtifactsBridge()
        with pytest.raises(StorageError, match="Storage error in"):
            await bridge.load_page("some_id")

    async def test_delete_page_not_configured(self):
        bridge = ArtifactsBridge()
        with pytest.raises(StorageError, match="Storage error in"):
            await bridge.delete_page("some_id")

    async def test_store_checkpoint(self):
        bridge = ArtifactsBridge()
        await bridge.configure()
        pages = [
            make_page("p1", content="page one"),
            make_page("p2", content="page two"),
        ]
        checkpoint_id = await bridge.store_checkpoint(pages, "cp_1")
        assert checkpoint_id is not None
        assert checkpoint_id.startswith("checkpoint_")

    async def test_load_checkpoint(self):
        bridge = ArtifactsBridge()
        await bridge.configure()
        pages = [
            make_page("p1", content="page one"),
            make_page("p2", content="page two"),
        ]
        checkpoint_id = await bridge.store_checkpoint(pages, "cp_1")
        restored = await bridge.load_checkpoint(checkpoint_id)
        assert len(restored) == 2
        ids = {p.page_id for p in restored}
        assert ids == {"p1", "p2"}

    async def test_load_checkpoint_not_found(self):
        bridge = ArtifactsBridge()
        await bridge.configure()
        restored = await bridge.load_checkpoint("nonexistent")
        assert restored == []

    async def test_load_checkpoint_not_configured(self):
        bridge = ArtifactsBridge()
        with pytest.raises(StorageError, match="Storage error in"):
            await bridge.load_checkpoint("some_id")

    async def test_store_checkpoint_not_configured(self):
        bridge = ArtifactsBridge()
        with pytest.raises(StorageError, match="Storage error in"):
            await bridge.store_checkpoint([], "cp")

    def test_get_stats_unconfigured(self):
        bridge = ArtifactsBridge()
        stats = bridge.get_stats()
        assert stats.backend == "unconfigured"
        assert stats.persistent is False

    async def test_get_stats_in_memory(self):
        bridge = ArtifactsBridge()
        await bridge.configure(session_id="sess_1")
        page = make_page("p1")
        await bridge.store_page(page)
        stats = bridge.get_stats()
        assert stats.backend == "in-memory"
        assert stats.persistent is False
        assert stats.pages_stored == 1
        assert stats.session_id == "sess_1"


# ===================================================================
# 8. Prefetcher tests
# ===================================================================


class TestSimplePrefetcher:
    """Tests for SimplePrefetcher."""

    def test_record_page_access(self):
        pf = SimplePrefetcher()
        pf.record_page_access("p1")
        pf.record_page_access("p1")
        pf.record_page_access("p2")
        stats = pf.get_stats()
        assert stats["pages_tracked"] == 2
        assert stats["total_accesses"] == 3

    def test_record_tool_call(self):
        pf = SimplePrefetcher()
        pf.record_tool_call("weather", turn=1)
        pf.record_tool_call("weather", turn=2, pages_accessed_before=["p1"])
        stats = pf.get_stats()
        assert stats["tools_tracked"] == 1

    def test_get_likely_tools(self):
        pf = SimplePrefetcher()
        pf.record_tool_call("weather", turn=1)
        pf.record_tool_call("calculator", turn=2)
        pf.record_tool_call("weather", turn=3)
        tools = pf.get_likely_tools()
        assert "weather" in tools
        assert "calculator" in tools

    def test_get_tool_prereq_pages(self):
        pf = SimplePrefetcher()
        pf.record_tool_call("weather", turn=1, pages_accessed_before=["loc_1"])
        pf.record_tool_call("weather", turn=2, pages_accessed_before=["loc_1", "pref_1"])
        prereqs = pf.get_tool_prereq_pages("weather")
        assert "loc_1" in prereqs

    def test_get_tool_prereq_pages_no_prereqs(self):
        pf = SimplePrefetcher()
        pf.record_tool_call("weather", turn=1)
        prereqs = pf.get_tool_prereq_pages("weather")
        assert prereqs == []

    def test_get_tool_prereq_pages_unknown_tool(self):
        pf = SimplePrefetcher()
        prereqs = pf.get_tool_prereq_pages("unknown_tool")
        assert prereqs == []

    def test_set_last_segment_summary(self):
        pf = SimplePrefetcher()
        pf.set_last_segment_summary("summary_001")
        assert pf._last_segment_summary_id == "summary_001"

    def test_get_top_claims_no_page_table(self):
        pf = SimplePrefetcher()
        pf.record_page_access("p1")
        pf.record_page_access("p1")
        pf.record_page_access("p2")
        claims = pf.get_top_claims()
        assert "p1" in claims

    def test_get_top_claims_with_page_table(self):
        pf = SimplePrefetcher()
        pt = PageTable()
        # Register a claim page and a non-claim page
        claim_page = make_page("claim_1", page_type=PageType.CLAIM)
        non_claim_page = make_page("p1", page_type=PageType.TRANSCRIPT)
        pt.register(claim_page)
        pt.register(non_claim_page)

        pf.record_page_access("claim_1")
        pf.record_page_access("claim_1")
        pf.record_page_access("p1")
        pf.record_page_access("p1")
        pf.record_page_access("p1")

        claims = pf.get_top_claims(page_table=pt)
        assert "claim_1" in claims
        assert "p1" not in claims  # Not a claim type

    def test_get_top_claims_with_limit(self):
        pf = SimplePrefetcher()
        for i in range(20):
            pf.record_page_access(f"p{i}")
        claims = pf.get_top_claims(limit=5)
        assert len(claims) <= 5

    def test_get_top_claims_page_table_filters_missing(self):
        pf = SimplePrefetcher()
        pt = PageTable()
        # Access page not in page table
        pf.record_page_access("ghost")
        claims = pf.get_top_claims(page_table=pt)
        assert claims == []

    async def test_prefetch_on_turn_start_empty(self):
        pf = SimplePrefetcher()
        pages = await pf.prefetch_on_turn_start("sess_1")
        assert pages == []

    async def test_prefetch_on_turn_start_with_summary(self):
        pf = SimplePrefetcher()
        pf.set_last_segment_summary("summary_001")
        pages = await pf.prefetch_on_turn_start("sess_1")
        assert "summary_001" in pages

    async def test_prefetch_on_turn_start_with_claims(self):
        pf = SimplePrefetcher()
        pf.record_page_access("claim_1")
        pages = await pf.prefetch_on_turn_start("sess_1")
        assert "claim_1" in pages

    async def test_prefetch_on_turn_start_with_tool_prereqs(self):
        pf = SimplePrefetcher()
        pf.record_tool_call("weather", turn=1, pages_accessed_before=["loc_1"])
        pages = await pf.prefetch_on_turn_start("sess_1")
        assert "loc_1" in pages

    async def test_prefetch_on_turn_start_no_duplicates(self):
        pf = SimplePrefetcher()
        pf.set_last_segment_summary("p1")
        pf.record_page_access("p1")
        pf.record_tool_call("tool", turn=1, pages_accessed_before=["p1"])
        pages = await pf.prefetch_on_turn_start("sess_1")
        assert pages.count("p1") == 1

    async def test_prefetch_on_turn_start_with_page_table(self):
        pf = SimplePrefetcher()
        pt = PageTable()
        claim_page = make_page("claim_1", page_type=PageType.CLAIM)
        pt.register(claim_page)
        pf.record_page_access("claim_1")
        pages = await pf.prefetch_on_turn_start("sess_1", page_table=pt)
        assert "claim_1" in pages

    def test_clear(self):
        pf = SimplePrefetcher()
        pf.record_page_access("p1")
        pf.record_tool_call("tool", turn=1, pages_accessed_before=["p1"])
        pf.set_last_segment_summary("s1")
        pf.clear()
        stats = pf.get_stats()
        assert stats["tools_tracked"] == 0
        assert stats["pages_tracked"] == 0
        assert stats["total_accesses"] == 0
        assert pf._last_segment_summary_id is None

    def test_get_stats(self):
        pf = SimplePrefetcher()
        pf.record_page_access("p1")
        pf.record_tool_call("tool", turn=1)
        stats = pf.get_stats()
        assert stats["tools_tracked"] == 1
        assert stats["pages_tracked"] == 1
        assert stats["total_accesses"] == 1


# ===================================================================
# Additional edge-case coverage
# ===================================================================


class TestPageTableEntryEvictionPriority:
    """Tests for PageTableEntry.eviction_priority property."""

    def test_claim_has_low_priority(self):
        entry = make_entry("p1", page_type=PageType.CLAIM)
        assert entry.eviction_priority == pytest.approx(0.1)

    def test_transcript_has_normal_priority(self):
        entry = make_entry("p1", page_type=PageType.TRANSCRIPT)
        assert entry.eviction_priority == pytest.approx(0.7)

    def test_pinned_has_zero_priority(self):
        entry = make_entry("p1", page_type=PageType.TRANSCRIPT, pinned=True)
        assert entry.eviction_priority == 0.0

    def test_summary_priority(self):
        entry = make_entry("p1", page_type=PageType.SUMMARY)
        assert entry.eviction_priority == pytest.approx(0.4)

    def test_procedure_priority(self):
        entry = make_entry("p1", page_type=PageType.PROCEDURE)
        assert entry.eviction_priority == pytest.approx(0.3)

    def test_index_priority(self):
        entry = make_entry("p1", page_type=PageType.INDEX)
        assert entry.eviction_priority == pytest.approx(0.2)

    def test_artifact_priority(self):
        entry = make_entry("p1", page_type=PageType.ARTIFACT)
        assert entry.eviction_priority == pytest.approx(0.6)


class TestMemoryPageEstimateTokens:
    """Tests for MemoryPage.estimate_tokens edge cases."""

    def test_with_size_tokens_set(self):
        page = make_page("p1", content="hello", size_tokens=42)
        assert page.estimate_tokens() == 42

    def test_with_string_content(self):
        page = make_page("p1", content="A" * 100)
        assert page.estimate_tokens() == 25

    def test_with_dict_content(self):
        page = make_page("p1", content={"key": "value"})
        assert page.estimate_tokens() > 0

    def test_with_none_content(self):
        page = make_page("p1", content=None)
        assert page.estimate_tokens() == 0

    def test_with_bytes_content(self):
        page = make_page("p1", content=b"binary data", size_bytes=100)
        assert page.estimate_tokens() == 25  # 100 // 4

    def test_with_other_content_no_size(self):
        page = make_page("p1", content=12345, size_bytes=0)
        assert page.estimate_tokens() == 100  # fallback


class TestContextPackerEdgeCases:
    """Additional edge cases for context packer."""

    def test_format_text_with_tool_result_type(self):
        packer = ContextPacker()
        page = make_page(
            "msg_1",
            content="tool output",
            metadata={"type": "transcript"},
        )
        result = packer._format_page(page)
        # "transcript" matches PageType.TOOL_RESULT.value, so prefix is T (TOOL)
        assert result.content[0] == "T"

    def test_pack_multiple_modalities(self):
        packer = ContextPacker()
        pages = [
            make_page("t1", content="text data", modality=Modality.TEXT),
            make_page(
                "i1",
                content=None,
                modality=Modality.IMAGE,
                caption="diagram",
            ),
            make_page(
                "a1",
                content=None,
                modality=Modality.AUDIO,
                duration_seconds=60,
            ),
            make_page(
                "v1",
                content=None,
                modality=Modality.VIDEO,
                duration_seconds=120,
            ),
            make_page(
                "s1",
                content={"result": 42},
                modality=Modality.STRUCTURED,
            ),
        ]
        result = packer.pack(pages)
        assert len(result.pages_included) == 5
        assert "t1" in result.content
        assert "IMAGE" in result.content
        assert "AUDIO" in result.content
        assert "VIDEO" in result.content


class TestManifestBuilderFromPages:
    """Additional tests for ManifestBuilder.build_from_pages."""

    def test_build_from_pages_with_policies(self):
        builder = ManifestBuilder()
        policies = ManifestPolicies(max_faults_per_turn=10)
        manifest = builder.build_from_pages(
            session_id="sess_1",
            working_set_pages=[],
            available_entries=[],
            policies=policies,
        )
        assert manifest.policies.max_faults_per_turn == 10

    def test_build_from_pages_max_available(self):
        builder = ManifestBuilder(max_available_pages=2)
        entries = [make_entry(f"p{i}", tier=StorageTier.L3) for i in range(10)]
        manifest = builder.build_from_pages(
            session_id="sess_1",
            working_set_pages=[],
            available_entries=entries,
        )
        assert len(manifest.available_pages) == 2


class TestWorkingSetManagerEvictionCandidatesL1Details:
    """Detailed tests for L1 eviction candidate scoring."""

    def test_l1_eviction_with_importance_override(self):
        wsm = WorkingSetManager()
        page = make_page("p1", importance=0.5)
        wsm.add_to_l1(page)
        wsm.set_importance("p1", 0.1)  # Low importance
        candidates = wsm.get_eviction_candidates(from_tier=StorageTier.L1)
        assert len(candidates) == 1

    def test_l1_eviction_with_anti_thrash_penalty(self):
        wsm = WorkingSetManager()
        wsm.add_to_l1(make_page("p1"))
        wsm.add_to_l1(make_page("p2"))
        # Record recent fault for p1
        wsm.anti_thrash.record_fault("p1", turn=wsm.current_turn)
        candidates = wsm.get_eviction_candidates(from_tier=StorageTier.L1)
        # p1 should be excluded (fault protection)
        ids = [c[0] for c in candidates]
        assert "p1" not in ids
        assert "p2" in ids

    def test_l0_eviction_with_anti_thrash(self):
        wsm = WorkingSetManager()
        wsm.add_to_l0(make_page("p1"))
        wsm.add_to_l0(make_page("p2"))
        wsm.anti_thrash.record_fault("p1", turn=wsm.current_turn)
        candidates = wsm.get_eviction_candidates(from_tier=StorageTier.L0)
        ids = [c[0] for c in candidates]
        assert "p1" not in ids


class TestPageTableRegisterPageTypes:
    """Tests for registering pages with various types and provenance."""

    def test_register_claim_page(self):
        pt = PageTable()
        page = make_page("claim_1", page_type=PageType.CLAIM, pinned=True)
        entry = pt.register(page)
        assert entry.page_type == PageType.CLAIM
        assert entry.pinned is True

    def test_register_page_with_provenance(self):
        pt = PageTable()
        page = make_page("s1")
        page.provenance = ["p1", "p2"]
        entry = pt.register(page)
        assert entry.provenance == ["p1", "p2"]

    def test_dirty_page_registration(self):
        pt = PageTable()
        page = make_page("p1", dirty=True)
        pt.register(page)
        assert len(pt.get_dirty_pages()) == 1


# ---------------------------------------------------------------------------
# ArtifactsBridge with mocked chuk-artifacts
# ---------------------------------------------------------------------------


class MockArtifactStore:
    """Mock for chuk_artifacts.ArtifactStore."""

    def __init__(self):
        self._store = {}
        self._counter = 0

    async def store(self, data, mime, scope, summary, metadata=None):
        self._counter += 1
        artifact_id = f"art_{self._counter}"
        self._store[artifact_id] = data
        return artifact_id

    async def retrieve(self, artifact_id):
        return self._store.get(artifact_id)

    async def delete(self, artifact_id):
        if artifact_id in self._store:
            del self._store[artifact_id]
            return True
        raise KeyError(f"Not found: {artifact_id}")


class MockStorageScope:
    """Mock for chuk_artifacts.models.StorageScope."""

    SESSION = "session"
    SANDBOX = "sandbox"


class TestArtifactsBridgeWithArtifacts:
    """Tests for ArtifactsBridge when chuk-artifacts is available (mocked)."""

    async def _make_bridge_with_artifacts(self):
        """Create a bridge configured with a mock artifact store."""
        import chuk_ai_session_manager.memory.artifacts_bridge as ab_module

        # Save originals
        orig_available = ab_module.ARTIFACTS_AVAILABLE
        orig_scope = ab_module.StorageScope

        # Patch
        ab_module.ARTIFACTS_AVAILABLE = True
        ab_module.StorageScope = MockStorageScope

        bridge = ArtifactsBridge()
        mock_store = MockArtifactStore()
        await bridge.configure(artifact_store=mock_store, session_id="test-session")

        return bridge, mock_store, ab_module, orig_available, orig_scope

    def _restore(self, ab_module, orig_available, orig_scope):
        ab_module.ARTIFACTS_AVAILABLE = orig_available
        ab_module.StorageScope = orig_scope

    async def test_configure_with_artifacts(self):
        (
            bridge,
            mock_store,
            ab_module,
            orig_available,
            orig_scope,
        ) = await self._make_bridge_with_artifacts()
        try:
            assert bridge._using_artifacts is True
            assert bridge._artifact_store is mock_store
            assert bridge.is_persistent is True
        finally:
            self._restore(ab_module, orig_available, orig_scope)

    async def test_store_page_with_artifacts(self):
        (
            bridge,
            mock_store,
            ab_module,
            orig_available,
            orig_scope,
        ) = await self._make_bridge_with_artifacts()
        try:
            page = make_page("p1", content="hello world")
            artifact_id = await bridge.store_page(page, StorageTier.L3)
            assert artifact_id.startswith("art_")
            # Verify data was stored
            assert artifact_id in mock_store._store
        finally:
            self._restore(ab_module, orig_available, orig_scope)

    async def test_store_page_l4_scope(self):
        (
            bridge,
            mock_store,
            ab_module,
            orig_available,
            orig_scope,
        ) = await self._make_bridge_with_artifacts()
        try:
            page = make_page("p1", content="archive data")
            artifact_id = await bridge.store_page(page, StorageTier.L4)
            assert artifact_id.startswith("art_")
        finally:
            self._restore(ab_module, orig_available, orig_scope)

    async def test_load_page_with_artifacts_bytes(self):
        (
            bridge,
            mock_store,
            ab_module,
            orig_available,
            orig_scope,
        ) = await self._make_bridge_with_artifacts()
        try:
            page = make_page("p1", content="load me")
            artifact_id = await bridge.store_page(page, StorageTier.L3)
            loaded = await bridge.load_page(artifact_id)
            assert loaded is not None
            assert loaded.page_id == "p1"
            assert loaded.content == "load me"
        finally:
            self._restore(ab_module, orig_available, orig_scope)

    async def test_load_page_with_artifacts_string(self):
        """Test _load_with_artifacts when retrieve returns a string."""
        (
            bridge,
            mock_store,
            ab_module,
            orig_available,
            orig_scope,
        ) = await self._make_bridge_with_artifacts()
        try:
            page = make_page("p1", content="string data")
            # Store normally first to get the serialized form
            artifact_id = await bridge.store_page(page, StorageTier.L3)
            # Replace stored data with a string instead of bytes
            stored_bytes = mock_store._store[artifact_id]
            mock_store._store[artifact_id] = stored_bytes.decode("utf-8")
            loaded = await bridge.load_page(artifact_id)
            assert loaded is not None
            assert loaded.page_id == "p1"
        finally:
            self._restore(ab_module, orig_available, orig_scope)

    async def test_load_page_with_artifacts_not_found(self):
        (
            bridge,
            mock_store,
            ab_module,
            orig_available,
            orig_scope,
        ) = await self._make_bridge_with_artifacts()
        try:
            loaded = await bridge.load_page("nonexistent")
            assert loaded is None
        finally:
            self._restore(ab_module, orig_available, orig_scope)

    async def test_load_page_with_artifacts_bad_data(self):
        """Test _load_with_artifacts when data is not deserializable."""
        (
            bridge,
            mock_store,
            ab_module,
            orig_available,
            orig_scope,
        ) = await self._make_bridge_with_artifacts()
        try:
            # Store non-page data
            mock_store._store["bad_id"] = 12345  # Not bytes or str
            loaded = await bridge.load_page("bad_id")
            assert loaded is None
        finally:
            self._restore(ab_module, orig_available, orig_scope)

    async def test_load_page_with_artifacts_exception(self):
        """Test _load_with_artifacts when exception occurs during deserialize."""
        (
            bridge,
            mock_store,
            ab_module,
            orig_available,
            orig_scope,
        ) = await self._make_bridge_with_artifacts()
        try:
            mock_store._store["corrupt"] = b"not valid json"
            loaded = await bridge.load_page("corrupt")
            assert loaded is None
        finally:
            self._restore(ab_module, orig_available, orig_scope)

    async def test_delete_page_with_artifacts(self):
        (
            bridge,
            mock_store,
            ab_module,
            orig_available,
            orig_scope,
        ) = await self._make_bridge_with_artifacts()
        try:
            page = make_page("p1", content="delete me")
            artifact_id = await bridge.store_page(page, StorageTier.L3)
            result = await bridge.delete_page(artifact_id)
            assert result is True
        finally:
            self._restore(ab_module, orig_available, orig_scope)

    async def test_delete_page_with_artifacts_not_found(self):
        (
            bridge,
            mock_store,
            ab_module,
            orig_available,
            orig_scope,
        ) = await self._make_bridge_with_artifacts()
        try:
            result = await bridge.delete_page("nonexistent")
            # delete raises KeyError in mock, caught by except -> False
            assert result is False
        finally:
            self._restore(ab_module, orig_available, orig_scope)

    async def test_store_checkpoint_with_artifacts(self):
        (
            bridge,
            mock_store,
            ab_module,
            orig_available,
            orig_scope,
        ) = await self._make_bridge_with_artifacts()
        try:
            pages = [
                make_page("p1", content="page one"),
                make_page("p2", content="page two"),
            ]
            checkpoint_id = await bridge.store_checkpoint(pages, "test_checkpoint")
            assert checkpoint_id.startswith("art_")
            # Verify checkpoint manifest was stored
            assert checkpoint_id in mock_store._store
        finally:
            self._restore(ab_module, orig_available, orig_scope)

    async def test_load_checkpoint_with_artifacts(self):
        (
            bridge,
            mock_store,
            ab_module,
            orig_available,
            orig_scope,
        ) = await self._make_bridge_with_artifacts()
        try:
            pages = [
                make_page("p1", content="page one"),
                make_page("p2", content="page two"),
            ]
            checkpoint_id = await bridge.store_checkpoint(pages, "test_checkpoint")
            loaded_pages = await bridge.load_checkpoint(checkpoint_id)
            assert len(loaded_pages) == 2
            page_ids = {p.page_id for p in loaded_pages}
            assert "p1" in page_ids
            assert "p2" in page_ids
        finally:
            self._restore(ab_module, orig_available, orig_scope)

    async def test_load_checkpoint_with_artifacts_not_found(self):
        (
            bridge,
            mock_store,
            ab_module,
            orig_available,
            orig_scope,
        ) = await self._make_bridge_with_artifacts()
        try:
            loaded = await bridge.load_checkpoint("nonexistent")
            assert loaded == []
        finally:
            self._restore(ab_module, orig_available, orig_scope)

    async def test_load_checkpoint_manifest_as_string(self):
        """Test load_checkpoint when manifest data comes back as string."""
        (
            bridge,
            mock_store,
            ab_module,
            orig_available,
            orig_scope,
        ) = await self._make_bridge_with_artifacts()
        try:
            pages = [make_page("p1", content="page one")]
            checkpoint_id = await bridge.store_checkpoint(pages, "test_checkpoint")
            # Replace stored bytes with string
            stored_bytes = mock_store._store[checkpoint_id]
            mock_store._store[checkpoint_id] = stored_bytes.decode("utf-8")
            loaded_pages = await bridge.load_checkpoint(checkpoint_id)
            assert len(loaded_pages) == 1
            assert loaded_pages[0].page_id == "p1"
        finally:
            self._restore(ab_module, orig_available, orig_scope)

    async def test_get_stats_with_artifacts(self):
        (
            bridge,
            mock_store,
            ab_module,
            orig_available,
            orig_scope,
        ) = await self._make_bridge_with_artifacts()
        try:
            stats = bridge.get_stats()
            assert stats.backend == "chuk-artifacts"
            assert stats.persistent is True
            assert stats.session_id == "test-session"
        finally:
            self._restore(ab_module, orig_available, orig_scope)

    async def test_store_with_artifacts_no_storage_scope(self):
        """Test _store_with_artifacts when StorageScope is None."""
        import chuk_ai_session_manager.memory.artifacts_bridge as ab_module

        orig_available = ab_module.ARTIFACTS_AVAILABLE
        orig_scope = ab_module.StorageScope

        ab_module.ARTIFACTS_AVAILABLE = True
        ab_module.StorageScope = None  # Simulate missing StorageScope

        try:
            bridge = ArtifactsBridge()
            mock_store = MockArtifactStore()
            await bridge.configure(artifact_store=mock_store, session_id="test-session")
            page = make_page("p1", content="no scope test")
            artifact_id = await bridge.store_page(page, StorageTier.L3)
            assert artifact_id.startswith("art_")
        finally:
            ab_module.ARTIFACTS_AVAILABLE = orig_available
            ab_module.StorageScope = orig_scope

    async def test_store_checkpoint_no_storage_scope(self):
        """Test store_checkpoint when StorageScope is None."""
        import chuk_ai_session_manager.memory.artifacts_bridge as ab_module

        orig_available = ab_module.ARTIFACTS_AVAILABLE
        orig_scope = ab_module.StorageScope

        ab_module.ARTIFACTS_AVAILABLE = True
        ab_module.StorageScope = None

        try:
            bridge = ArtifactsBridge()
            mock_store = MockArtifactStore()
            await bridge.configure(artifact_store=mock_store, session_id="test-session")
            pages = [make_page("p1", content="page one")]
            checkpoint_id = await bridge.store_checkpoint(pages, "test_checkpoint")
            assert checkpoint_id.startswith("art_")
        finally:
            ab_module.ARTIFACTS_AVAILABLE = orig_available
            ab_module.StorageScope = orig_scope
