# tests/test_memory_manager.py
"""
Tests for MemoryManager orchestrator and SessionManager VM integration.

Covers:
- MemoryManager lifecycle, page creation, working set, eviction
- Fault handling, context building, turn management
- PageLoader protocol, event_to_page helper
- SessionManager enable_vm integration
"""

from chuk_ai_session_manager.memory.manager import MemoryManager, event_to_page
from chuk_ai_session_manager.memory.models import (
    FaultPolicy,
    MemoryPage,
    Modality,
    PageType,
    StorageTier,
    VMMode,
)
from chuk_ai_session_manager.memory.working_set import WorkingSetConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_page(
    page_id="p1",
    content="test content",
    session_id="test-session",
    tier=StorageTier.L0,
    modality=Modality.TEXT,
    page_type=PageType.TRANSCRIPT,
    importance=0.5,
    size_tokens=None,
    **kwargs,
):
    return MemoryPage(
        page_id=page_id,
        session_id=session_id,
        content=content,
        storage_tier=tier,
        modality=modality,
        page_type=page_type,
        importance=importance,
        size_tokens=size_tokens,
        **kwargs,
    )


# ===========================================================================
# TestMemoryManagerLifecycle
# ===========================================================================


class TestMemoryManagerLifecycle:
    """Test MemoryManager initialization and properties."""

    def test_default_init(self):
        vm = MemoryManager()
        assert vm.session_id  # auto-generated UUID
        assert vm.mode == VMMode.STRICT
        assert vm.turn == 0
        assert vm.page_table is not None
        assert vm.working_set is not None
        assert vm.metrics is not None

    def test_custom_session_id(self):
        vm = MemoryManager(session_id="sess-abc")
        assert vm.session_id == "sess-abc"

    def test_custom_config(self):
        config = WorkingSetConfig(max_tokens=2000)
        vm = MemoryManager(config=config)
        assert vm.working_set is not None

    def test_custom_mode(self):
        vm = MemoryManager(mode=VMMode.PASSIVE)
        assert vm.mode == VMMode.PASSIVE

    def test_custom_fault_policy(self):
        policy = FaultPolicy(max_faults_per_turn=5)
        vm = MemoryManager(fault_policy=policy)
        # Verify policy is stored (internal)
        assert vm._fault_policy.max_faults_per_turn == 5

    def test_page_store_starts_empty(self):
        vm = MemoryManager()
        assert len(vm._page_store) == 0
        assert len(vm._page_hints) == 0


# ===========================================================================
# TestPageCreation
# ===========================================================================


class TestPageCreation:
    """Test MemoryManager.create_page()."""

    def test_basic_creation(self):
        vm = MemoryManager(session_id="s1")
        page = vm.create_page("Hello world")

        assert page.content == "Hello world"
        assert page.page_type == PageType.TRANSCRIPT
        assert page.modality == Modality.TEXT
        assert page.storage_tier == StorageTier.L1
        assert page.session_id == "s1"

    def test_custom_page_id(self):
        vm = MemoryManager()
        page = vm.create_page("data", page_id="custom-id")
        assert page.page_id == "custom-id"

    def test_auto_generated_page_id(self):
        vm = MemoryManager()
        page = vm.create_page("data")
        assert page.page_id.startswith("transcript_")

    def test_page_type_prefix_in_id(self):
        vm = MemoryManager()
        page = vm.create_page("data", page_type=PageType.ARTIFACT)
        assert page.page_id.startswith("artifact_")

    def test_with_hint(self):
        vm = MemoryManager()
        page = vm.create_page("data", hint="weather info")
        assert vm._page_hints[page.page_id] == "weather info"

    def test_without_hint(self):
        vm = MemoryManager()
        page = vm.create_page("data")
        assert page.page_id not in vm._page_hints

    def test_registered_in_page_table(self):
        vm = MemoryManager()
        page = vm.create_page("data")
        entry = vm.page_table.lookup(page.page_id)
        assert entry is not None

    def test_stored_in_page_store(self):
        vm = MemoryManager()
        page = vm.create_page("data")
        assert page.page_id in vm._page_store

    def test_mutation_logged(self):
        vm = MemoryManager()
        vm.create_page("data")
        summary = vm._mutation_log.get_summary()
        assert summary["total_mutations"] >= 1

    def test_with_size_tokens(self):
        vm = MemoryManager()
        page = vm.create_page("data", size_tokens=42)
        assert page.size_tokens == 42

    def test_with_importance(self):
        vm = MemoryManager()
        page = vm.create_page("important", importance=0.9)
        assert page.importance == 0.9

    def test_with_provenance(self):
        vm = MemoryManager()
        page = vm.create_page("data", provenance=["src1", "src2"])
        assert page.provenance == ["src1", "src2"]


# ===========================================================================
# TestWorkingSetManagement
# ===========================================================================


class TestWorkingSetManagement:
    """Test add_to_working_set and related operations."""

    async def test_add_page_to_l0(self):
        vm = MemoryManager()
        page = vm.create_page("Hello", size_tokens=10)
        result = await vm.add_to_working_set(page)

        assert result is True
        l0_ids = vm.working_set.get_l0_page_ids()
        assert page.page_id in l0_ids

    async def test_page_table_updated_to_l0(self):
        vm = MemoryManager()
        page = vm.create_page("Hello", size_tokens=10)
        await vm.add_to_working_set(page)

        entry = vm.page_table.lookup(page.page_id)
        assert entry is not None
        assert entry.tier == StorageTier.L0

    async def test_metrics_updated(self):
        vm = MemoryManager()
        page = vm.create_page("Hello", size_tokens=10)
        await vm.add_to_working_set(page)

        assert vm.metrics.tokens_in_working_set > 0

    async def test_multiple_pages(self):
        vm = MemoryManager()
        pages = []
        for i in range(3):
            p = vm.create_page(f"Page {i}", page_id=f"p{i}", size_tokens=10)
            await vm.add_to_working_set(p)
            pages.append(p)

        l0_ids = vm.working_set.get_l0_page_ids()
        assert len(l0_ids) == 3

    async def test_get_l0_pages(self):
        vm = MemoryManager()
        page = vm.create_page("content", size_tokens=10)
        await vm.add_to_working_set(page)

        l0 = vm.get_l0_pages()
        assert len(l0) == 1
        assert l0[0].page_id == page.page_id


# ===========================================================================
# TestEviction
# ===========================================================================


class TestEviction:
    """Test page eviction."""

    async def test_evict_to_l2(self):
        vm = MemoryManager()
        page = vm.create_page("data", page_id="ev1", size_tokens=10)
        await vm.add_to_working_set(page)

        result = await vm.evict_page("ev1", StorageTier.L2)
        assert result is True

        # Should not be in L0 anymore
        l0_ids = vm.working_set.get_l0_page_ids()
        assert "ev1" not in l0_ids

    async def test_evict_updates_page_table(self):
        vm = MemoryManager()
        page = vm.create_page("data", page_id="ev2", size_tokens=10)
        await vm.add_to_working_set(page)
        await vm.evict_page("ev2", StorageTier.L2)

        entry = vm.page_table.lookup("ev2")
        assert entry is not None
        assert entry.tier == StorageTier.L2

    async def test_evict_nonexistent_returns_false(self):
        vm = MemoryManager()
        result = await vm.evict_page("nope", StorageTier.L2)
        assert result is False

    async def test_evict_records_metrics(self):
        vm = MemoryManager()
        page = vm.create_page("data", page_id="ev3", size_tokens=10)
        await vm.add_to_working_set(page)

        before = vm.metrics.evictions_total
        await vm.evict_page("ev3", StorageTier.L2)
        assert vm.metrics.evictions_total == before + 1

    async def test_evict_to_l3_removes_from_store_when_bridge_configured(self):
        """L3/L4 eviction removes content from memory if bridge is not configured."""
        vm = MemoryManager()
        page = vm.create_page("data", page_id="ev4", size_tokens=10)
        await vm.add_to_working_set(page)

        # Without bridge configured, L3 eviction still removes from page_store
        await vm.evict_page("ev4", StorageTier.L3)
        assert "ev4" not in vm._page_store


# ===========================================================================
# TestFaultHandling
# ===========================================================================


class TestFaultHandling:
    """Test handle_fault."""

    async def test_fault_on_known_page(self):
        vm = MemoryManager()
        vm.create_page("recoverable", page_id="f1", size_tokens=10)
        # Page is in page_store and page_table but NOT in working set yet
        # Simulate it being evicted (tier > L0)
        vm._page_table.update_location("f1", tier=StorageTier.L2)

        result = await vm.handle_fault("f1")
        assert result.success is True
        assert result.page is not None

    async def test_fault_adds_to_working_set(self):
        vm = MemoryManager()
        vm.create_page("data", page_id="f2", size_tokens=10)
        vm._page_table.update_location("f2", tier=StorageTier.L2)

        await vm.handle_fault("f2")
        l0_ids = vm.working_set.get_l0_page_ids()
        assert "f2" in l0_ids

    async def test_fault_on_unknown_page(self):
        vm = MemoryManager()
        result = await vm.handle_fault("nonexistent")
        assert result.success is False
        assert result.error is not None

    async def test_fault_policy_limit(self):
        policy = FaultPolicy(max_faults_per_turn=0, max_tokens_per_turn=0)
        vm = MemoryManager(fault_policy=policy)
        vm.create_page("data", page_id="f3", size_tokens=10)
        vm._page_table.update_location("f3", tier=StorageTier.L2)

        result = await vm.handle_fault("f3")
        assert result.success is False
        assert "limit" in (result.error or "").lower()


# ===========================================================================
# TestContextBuilding
# ===========================================================================


class TestContextBuilding:
    """Test build_context."""

    async def test_build_empty_context(self):
        vm = MemoryManager()
        ctx = vm.build_context()

        assert "developer_message" in ctx
        assert "tools" in ctx
        assert "manifest" in ctx
        assert "packed_context" in ctx

    async def test_build_context_with_pages(self):
        vm = MemoryManager()
        page = vm.create_page("Hello world", size_tokens=10)
        await vm.add_to_working_set(page)

        ctx = vm.build_context()
        assert "Hello world" in ctx["developer_message"]

    async def test_build_context_with_system_prompt(self):
        vm = MemoryManager()
        ctx = vm.build_context(system_prompt="You are helpful.")
        assert "You are helpful" in ctx["developer_message"]

    async def test_strict_mode_includes_tools(self):
        vm = MemoryManager(mode=VMMode.STRICT)
        ctx = vm.build_context()
        assert len(ctx["tools"]) > 0

    async def test_passive_mode_no_tools(self):
        vm = MemoryManager(mode=VMMode.PASSIVE)
        ctx = vm.build_context()
        assert len(ctx["tools"]) == 0

    async def test_relaxed_mode_includes_tools(self):
        vm = MemoryManager(mode=VMMode.RELAXED)
        ctx = vm.build_context()
        assert len(ctx["tools"]) > 0

    async def test_mode_override(self):
        vm = MemoryManager(mode=VMMode.STRICT)
        ctx = vm.build_context(mode=VMMode.PASSIVE)
        assert len(ctx["tools"]) == 0


# ===========================================================================
# TestTurnManagement
# ===========================================================================


class TestTurnManagement:
    """Test new_turn."""

    def test_turn_increments(self):
        vm = MemoryManager()
        assert vm.turn == 0
        vm.new_turn()
        assert vm.turn == 1
        vm.new_turn()
        assert vm.turn == 2

    def test_turn_snapshots_context(self):
        vm = MemoryManager()
        vm.new_turn()
        summary = vm._mutation_log.get_summary()
        assert summary["context_snapshots"] >= 1


# ===========================================================================
# TestPinning
# ===========================================================================


class TestPinning:
    """Test page pinning."""

    async def test_pin_page(self):
        vm = MemoryManager()
        page = vm.create_page("pin me", page_id="pin1", size_tokens=10)
        await vm.add_to_working_set(page)
        vm.pin_page("pin1")

        assert vm.working_set.is_pinned("pin1")

    async def test_pin_logs_mutation(self):
        vm = MemoryManager()
        vm.create_page("pin me", page_id="pin2", size_tokens=10)
        vm.pin_page("pin2")

        summary = vm._mutation_log.get_summary()
        assert summary["total_mutations"] >= 2  # create + pin


# ===========================================================================
# TestPageLoaderProtocol
# ===========================================================================


class TestPageLoaderProtocol:
    """Test MemoryManager.load() (PageLoader protocol)."""

    async def test_load_from_memory(self):
        vm = MemoryManager()
        vm.create_page("loadable", page_id="ld1")
        loaded = await vm.load("ld1", StorageTier.L1)
        assert loaded is not None
        assert loaded.page_id == "ld1"

    async def test_load_missing_page(self):
        vm = MemoryManager()
        loaded = await vm.load("missing", StorageTier.L1)
        assert loaded is None

    async def test_load_with_artifact_id_no_bridge(self):
        vm = MemoryManager()
        loaded = await vm.load("x", StorageTier.L3, artifact_id="art-1")
        assert loaded is None  # bridge not configured


# ===========================================================================
# TestEventToPage
# ===========================================================================


class TestEventToPage:
    """Test the event_to_page standalone function."""

    def test_user_message(self):
        page = event_to_page(
            message="Hello",
            role="user",
            session_id="s1",
        )
        assert page.content == "Hello"
        assert page.session_id == "s1"
        assert page.page_type == PageType.TRANSCRIPT
        assert page.modality == Modality.TEXT
        assert page.storage_tier == StorageTier.L1

    def test_assistant_message(self):
        page = event_to_page(
            message="Hi there!",
            role="assistant",
            session_id="s2",
        )
        assert page.content == "Hi there!"
        assert page.session_id == "s2"

    def test_with_event_id(self):
        page = event_to_page(
            message="test",
            role="user",
            session_id="s1",
            event_id="abcdef1234567890",
        )
        assert page.page_id == "msg_abcdef12"

    def test_without_event_id(self):
        page = event_to_page(
            message="test",
            role="user",
            session_id="s1",
        )
        assert page.page_id.startswith("msg_")

    def test_custom_page_type(self):
        page = event_to_page(
            message="result",
            role="tool",
            session_id="s1",
            page_type=PageType.ARTIFACT,
        )
        assert page.page_type == PageType.ARTIFACT

    def test_custom_importance(self):
        page = event_to_page(
            message="critical",
            role="system",
            session_id="s1",
            importance=0.9,
        )
        assert page.importance == 0.9


# ===========================================================================
# TestStats
# ===========================================================================


class TestStats:
    """Test get_stats returns all subsystem keys."""

    async def test_stats_keys(self):
        vm = MemoryManager(session_id="stats-test")
        stats = vm.get_stats()

        expected_keys = [
            "session_id",
            "turn",
            "mode",
            "page_table",
            "working_set",
            "fault_handler",
            "tlb",
            "mutation_log",
            "prefetcher",
            "pack_cache",
            "metrics",
            "pages_in_store",
        ]
        for key in expected_keys:
            assert key in stats, f"Missing stats key: {key}"

    async def test_stats_session_id(self):
        vm = MemoryManager(session_id="my-sess")
        stats = vm.get_stats()
        assert stats["session_id"] == "my-sess"

    async def test_stats_mode(self):
        vm = MemoryManager(mode=VMMode.RELAXED)
        stats = vm.get_stats()
        assert stats["mode"] == "relaxed"

    async def test_stats_pages_count(self):
        vm = MemoryManager()
        vm.create_page("a", page_id="s1")
        vm.create_page("b", page_id="s2")
        stats = vm.get_stats()
        assert stats["pages_in_store"] == 2


# ===========================================================================
# TestBridgeConfiguration
# ===========================================================================


class TestBridgeConfiguration:
    """Test configure_bridge."""

    async def test_configure_bridge(self):
        vm = MemoryManager(session_id="bridge-test")
        assert vm._bridge_configured is False

        await vm.configure_bridge()
        assert vm._bridge_configured is True

    async def test_configure_bridge_custom_session(self):
        vm = MemoryManager(session_id="original")
        await vm.configure_bridge(session_id="override")
        assert vm._bridge_configured is True


# ===========================================================================
# TestSessionManagerVMIntegration
# ===========================================================================


class TestSessionManagerVMIntegration:
    """Test SessionManager with enable_vm=True."""

    async def test_vm_disabled_by_default(self):
        from chuk_ai_session_manager import SessionManager

        sm = SessionManager()
        assert sm.vm is None

    async def test_vm_enabled(self):
        from chuk_ai_session_manager import SessionManager

        sm = SessionManager(enable_vm=True)
        assert sm.vm is not None
        assert isinstance(sm.vm, MemoryManager)

    async def test_vm_custom_mode(self):
        from chuk_ai_session_manager import SessionManager

        sm = SessionManager(enable_vm=True, vm_mode=VMMode.RELAXED)
        assert sm.vm is not None
        assert sm.vm.mode == VMMode.RELAXED

    async def test_vm_custom_config(self):
        from chuk_ai_session_manager import SessionManager

        config = WorkingSetConfig(max_tokens=5000)
        sm = SessionManager(enable_vm=True, vm_config=config)
        assert sm.vm is not None

    async def test_user_says_creates_vm_page(self):
        from chuk_ai_session_manager import SessionManager

        sm = SessionManager(enable_vm=True)
        await sm.user_says("Hello VM!")

        assert sm.vm is not None
        l0 = sm.vm.get_l0_pages()
        assert len(l0) >= 1
        # Check page content
        contents = [p.content for p in l0]
        assert any("Hello VM!" in str(c) for c in contents)

    async def test_ai_responds_creates_vm_page(self):
        from chuk_ai_session_manager import SessionManager

        sm = SessionManager(enable_vm=True)
        await sm.user_says("Question")
        await sm.ai_responds("Answer", model="test-model")

        assert sm.vm is not None
        l0 = sm.vm.get_l0_pages()
        assert len(l0) >= 2

    async def test_tool_used_creates_vm_page(self):
        from chuk_ai_session_manager import SessionManager

        sm = SessionManager(enable_vm=True)
        await sm.tool_used("search", {"q": "test"}, "results here")

        assert sm.vm is not None
        pages = list(sm.vm._page_store.values())
        assert len(pages) >= 1
        # Tool pages should be ARTIFACT type
        tool_pages = [p for p in pages if p.page_type == PageType.ARTIFACT]
        assert len(tool_pages) >= 1

    async def test_get_vm_context(self):
        from chuk_ai_session_manager import SessionManager

        sm = SessionManager(enable_vm=True, system_prompt="Be helpful.")
        await sm.user_says("Hello")

        ctx = sm.get_vm_context()
        assert ctx is not None
        assert "developer_message" in ctx
        assert "tools" in ctx

    async def test_get_vm_context_when_disabled(self):
        from chuk_ai_session_manager import SessionManager

        sm = SessionManager(enable_vm=False)
        ctx = sm.get_vm_context()
        assert ctx is None

    async def test_vm_session_id_matches(self):
        from chuk_ai_session_manager import SessionManager

        sm = SessionManager(enable_vm=True)
        await sm.user_says("init")

        assert sm.vm is not None
        assert sm.vm.session_id == sm.session_id


# ===========================================================================
# TestSearchPages
# ===========================================================================


class TestSearchPages:
    """Test MemoryManager.search_pages()."""

    async def test_search_with_hints(self):
        vm = MemoryManager(session_id="search-test")
        vm.create_page(
            "schema v1", hint="database design", page_id="db1", size_tokens=10
        )
        vm.create_page(
            "api spec", hint="rest api endpoints", page_id="api1", size_tokens=10
        )

        result = await vm.search_pages("database")
        assert len(result.results) >= 1
        found_ids = [r.page_id for r in result.results]
        assert "db1" in found_ids

    async def test_search_no_results(self):
        vm = MemoryManager(session_id="search-empty")
        vm.create_page("hello world", hint="greeting", page_id="g1", size_tokens=10)

        result = await vm.search_pages("zzz_nonexistent_zzz")
        assert len(result.results) == 0

    async def test_search_respects_limit(self):
        vm = MemoryManager(session_id="search-limit")
        for i in range(10):
            vm.create_page(
                f"item {i}",
                hint="common keyword",
                page_id=f"item{i}",
                size_tokens=10,
            )

        result = await vm.search_pages("common", limit=2)
        assert len(result.results) <= 2


# ===========================================================================
# TestEvictSegmentPages
# ===========================================================================


class TestEvictSegmentPages:
    """Test MemoryManager.evict_segment_pages()."""

    async def test_evicts_non_pinned(self):
        vm = MemoryManager(session_id="evict-seg")
        pages = []
        for i in range(3):
            p = vm.create_page(f"page {i}", page_id=f"seg{i}", size_tokens=10)
            await vm.add_to_working_set(p)
            pages.append(p)

        # Pin one page
        vm.pin_page("seg1")

        evicted = await vm.evict_segment_pages()
        assert len(evicted) == 2
        # Pinned page should still be in L0
        l0_ids = vm.working_set.get_l0_page_ids()
        assert "seg1" in l0_ids

    async def test_preserves_pinned(self):
        vm = MemoryManager(session_id="evict-pin")
        p0 = vm.create_page("keep me", page_id="pinned0", size_tokens=10)
        p1 = vm.create_page("evict me", page_id="loose0", size_tokens=10)
        await vm.add_to_working_set(p0)
        await vm.add_to_working_set(p1)

        vm.pin_page("pinned0")
        await vm.evict_segment_pages()

        l0_ids = vm.working_set.get_l0_page_ids()
        assert "pinned0" in l0_ids
        assert "loose0" not in l0_ids

    async def test_returns_evicted_ids(self):
        vm = MemoryManager(session_id="evict-ids")
        for i in range(4):
            p = vm.create_page(f"data {i}", page_id=f"eid{i}", size_tokens=10)
            await vm.add_to_working_set(p)

        vm.pin_page("eid2")
        evicted = await vm.evict_segment_pages()

        assert "eid0" in evicted
        assert "eid1" in evicted
        assert "eid3" in evicted
        assert "eid2" not in evicted


# ===========================================================================
# TestSegmentationHook
# ===========================================================================


class TestSegmentationHook:
    """Test VM integration during session segmentation."""

    async def test_segment_creates_summary_page(self):
        from chuk_ai_session_manager import SessionManager

        sm = SessionManager(
            enable_vm=True,
            infinite_context=True,
            max_turns_per_segment=2,
        )
        # Fill up the segment (2 messages hit the threshold)
        await sm.user_says("First question?")
        await sm.ai_responds("First answer", model="test")

        # This user_says should trigger segmentation
        await sm.user_says("Second question triggers segment")

        assert sm.vm is not None
        # Check that a SUMMARY page was created in the page store
        summary_pages = [
            p for p in sm.vm._page_store.values() if p.page_type == PageType.SUMMARY
        ]
        assert len(summary_pages) >= 1

    async def test_segment_pins_summary(self):
        from chuk_ai_session_manager import SessionManager

        sm = SessionManager(
            enable_vm=True,
            infinite_context=True,
            max_turns_per_segment=2,
        )
        await sm.user_says("Question one?")
        await sm.ai_responds("Answer one", model="test")
        await sm.user_says("Trigger segmentation")

        assert sm.vm is not None
        # Find the summary page and verify it is pinned
        summary_pages = [
            p for p in sm.vm._page_store.values() if p.page_type == PageType.SUMMARY
        ]
        assert len(summary_pages) >= 1
        for sp in summary_pages:
            assert sm.vm.working_set.is_pinned(sp.page_id)

    async def test_segment_evicts_old_pages(self):
        from chuk_ai_session_manager import SessionManager

        sm = SessionManager(
            enable_vm=True,
            infinite_context=True,
            max_turns_per_segment=2,
        )
        await sm.user_says("Old message one")
        await sm.ai_responds("Old response one", model="test")

        assert sm.vm is not None
        # Record L0 page IDs before segmentation
        l0_before = set(sm.vm.working_set.get_l0_page_ids())
        assert len(l0_before) >= 2  # At least user + assistant pages

        # Trigger segmentation
        await sm.user_says("New message triggers segment")

        # Old transcript pages should have been evicted from L0
        l0_after = set(sm.vm.working_set.get_l0_page_ids())
        # The old non-pinned pages should no longer be in L0
        evicted_from_l0 = l0_before - l0_after
        assert len(evicted_from_l0) >= 1

    async def test_segment_updates_vm_session_id(self):
        from chuk_ai_session_manager import SessionManager

        sm = SessionManager(
            enable_vm=True,
            infinite_context=True,
            max_turns_per_segment=2,
        )
        await sm.user_says("Setup question?")
        await sm.ai_responds("Setup answer", model="test")

        old_session_id = sm.session_id

        # Trigger segmentation
        await sm.user_says("Trigger new segment")

        assert sm.vm is not None
        # Session ID should have changed after segmentation
        assert sm.session_id != old_session_id
        # VM session_id should match the new session_id
        assert sm.vm._session_id == sm.session_id


# ===========================================================================
# TestVMAwarePromptBuilder
# ===========================================================================


class TestVMAwarePromptBuilder:
    """Test get_messages_for_llm with VM-aware context."""

    async def test_vm_messages_include_vm_context(self):
        from chuk_ai_session_manager import SessionManager

        sm = SessionManager(enable_vm=True, system_prompt="You are helpful.")
        await sm.user_says("Hello VM!")

        messages = await sm.get_messages_for_llm()
        assert len(messages) >= 2  # system + user

        # First message should be system with VM context
        system_msg = messages[0]
        assert system_msg["role"] == "system"
        content = system_msg["content"]
        assert "VM:CONTEXT" in content or "VM:RULES" in content

    async def test_vm_messages_include_system_false_returns_raw(self):
        from chuk_ai_session_manager import SessionManager

        sm = SessionManager(enable_vm=True, system_prompt="You are helpful.")
        await sm.user_says("Hello!")

        messages = await sm.get_messages_for_llm(include_system=False)
        # Should NOT contain VM context when include_system=False
        for msg in messages:
            assert "VM:CONTEXT" not in msg.get("content", "")
            assert "VM:RULES" not in msg.get("content", "")

    async def test_no_vm_returns_standard(self):
        from chuk_ai_session_manager import SessionManager

        sm = SessionManager(system_prompt="Standard prompt.")
        await sm.user_says("Hello!")

        messages = await sm.get_messages_for_llm()
        assert len(messages) >= 2

        # First message should be standard system prompt
        system_msg = messages[0]
        assert system_msg["role"] == "system"
        assert system_msg["content"] == "Standard prompt."
        assert "VM:CONTEXT" not in system_msg["content"]

    async def test_vm_messages_include_conversation(self):
        from chuk_ai_session_manager import SessionManager

        sm = SessionManager(enable_vm=True, system_prompt="Test prompt.")
        await sm.user_says("User message one")
        await sm.ai_responds("Assistant reply one", model="test")
        await sm.user_says("User message two")

        messages = await sm.get_messages_for_llm()

        # Should have system + user + assistant + user = 4 messages
        assert len(messages) >= 4

        # System message first
        assert messages[0]["role"] == "system"

        # Conversation messages should follow
        roles = [m["role"] for m in messages[1:]]
        assert "user" in roles
        assert "assistant" in roles

        # Verify user and assistant content appears
        all_content = " ".join(m["content"] for m in messages)
        assert "User message one" in all_content
        assert "Assistant reply one" in all_content
        assert "User message two" in all_content
