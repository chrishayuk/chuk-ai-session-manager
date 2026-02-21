# tests/test_memory.py
"""
Tests for the AI Virtual Memory subsystem.

Tests cover:
- Models (MemoryPage, PageTableEntry, TokenBudget, VMMetrics)
- PageTable and TLB
- WorkingSetManager
- ContextPacker
- ManifestBuilder
- PageFaultHandler
- ArtifactsBridge
"""

import pytest

from chuk_ai_session_manager.memory import (
    # Enums
    Actor,
    # Working Set
    AntiThrashPolicy,
    ArtifactsBridge,
    CompressionLevel,
    ContextPackCache,
    # Context Packing
    ContextPacker,
    # Core Models
    FaultPolicy,
    FaultReason,
    FaultRecord,
    FaultResult,
    InMemoryBackend,
    ManifestBuilder,
    ManifestPolicies,
    MemoryABI,
    MemoryPage,
    Modality,
    # Mutation Log
    MutationLogLite,
    MutationType,
    PageFaultHandler,
    PageSearchHandler,
    # Data Structures
    PageTable,
    PageTableEntry,
    PageTLB,
    PageType,
    PinnedSet,
    # Prefetcher
    SimplePrefetcher,
    StorageTier,
    TLBWithPageTable,
    TokenBudget,
    ToolUsagePattern,
    UserExperienceMetrics,
    VMManifest,
    VMMetrics,
    WorkingSetConfig,
    WorkingSetManager,
)

# =============================================================================
# Model Tests
# =============================================================================


class TestMemoryPage:
    """Tests for MemoryPage model."""

    def test_create_text_page(self):
        """Test creating a basic text page."""
        page = MemoryPage(
            page_id="msg_001",
            modality=Modality.TEXT,
            content="Hello, world!",
        )
        assert page.page_id == "msg_001"
        assert page.modality == Modality.TEXT
        assert page.storage_tier == StorageTier.L1  # Default
        assert page.compression_level == CompressionLevel.FULL
        assert not page.dirty

    def test_estimate_tokens(self):
        """Test token estimation for text content."""
        page = MemoryPage(
            page_id="msg_001",
            modality=Modality.TEXT,
            content="This is a test message with some content.",
        )
        tokens = page.estimate_tokens()
        # ~4 chars per token
        assert tokens > 0
        assert tokens == len(page.content) // 4

    def test_mark_accessed(self):
        """Test access tracking."""
        page = MemoryPage(
            page_id="msg_001",
            modality=Modality.TEXT,
        )
        initial_count = page.access_count
        initial_time = page.last_accessed

        page.mark_accessed()

        assert page.access_count == initial_count + 1
        assert page.last_accessed >= initial_time

    def test_dirty_tracking(self):
        """Test dirty bit management."""
        page = MemoryPage(
            page_id="msg_001",
            modality=Modality.TEXT,
        )
        assert not page.dirty

        page.mark_dirty()
        assert page.dirty

        page.mark_clean()
        assert not page.dirty


class TestTokenBudget:
    """Tests for TokenBudget model."""

    def test_default_budget(self):
        """Test default token budget values."""
        budget = TokenBudget()
        assert budget.total_limit == 128000
        assert budget.reserved == 4000
        assert budget.used == 0
        assert budget.available == 128000 - 4000

    def test_add_tokens(self):
        """Test adding tokens to budget."""
        budget = TokenBudget()
        success = budget.add(1000, Modality.TEXT)
        assert success
        assert budget.text_tokens == 1000
        assert budget.used == 1000

    def test_budget_overflow(self):
        """Test that budget prevents overflow."""
        budget = TokenBudget(total_limit=1000, reserved=100)
        # Try to add more than available
        success = budget.add(1000, Modality.TEXT)
        assert not success
        assert budget.used == 0

    def test_utilization(self):
        """Test utilization calculation."""
        budget = TokenBudget(total_limit=1000, reserved=0)
        budget.add(500, Modality.TEXT)
        assert budget.utilization == 0.5


class TestVMMetrics:
    """Tests for VMMetrics model."""

    def test_record_fault(self):
        """Test fault recording."""
        metrics = VMMetrics()
        metrics.record_fault()
        assert metrics.faults_total == 1
        assert metrics.faults_this_turn == 1

    def test_tlb_hit_rate(self):
        """Test TLB hit rate calculation."""
        metrics = VMMetrics()
        metrics.record_tlb_hit()
        metrics.record_tlb_hit()
        metrics.record_tlb_miss()
        assert metrics.tlb_hit_rate == pytest.approx(2 / 3)

    def test_new_turn_resets_counters(self):
        """Test that new_turn resets per-turn counters."""
        metrics = VMMetrics()
        metrics.record_fault()
        metrics.record_eviction()
        metrics.new_turn()
        assert metrics.faults_this_turn == 0
        assert metrics.evictions_this_turn == 0
        assert metrics.faults_total == 1  # Total not reset


# =============================================================================
# PageTable Tests
# =============================================================================


class TestPageTable:
    """Tests for PageTable."""

    def test_register_page(self):
        """Test registering a page."""
        table = PageTable()
        page = MemoryPage(
            page_id="msg_001",
            modality=Modality.TEXT,
            content="Hello",
        )
        entry = table.register(page)
        assert entry.page_id == "msg_001"
        assert "msg_001" in table

    def test_lookup(self):
        """Test looking up a page."""
        table = PageTable()
        page = MemoryPage(page_id="msg_001", modality=Modality.TEXT)
        table.register(page)

        entry = table.lookup("msg_001")
        assert entry is not None
        assert entry.page_id == "msg_001"

        missing = table.lookup("nonexistent")
        assert missing is None

    def test_update_location(self):
        """Test updating page location."""
        table = PageTable()
        page = MemoryPage(
            page_id="msg_001",
            modality=Modality.TEXT,
            storage_tier=StorageTier.L1,
        )
        table.register(page)

        success = table.update_location("msg_001", StorageTier.L2)
        assert success

        entry = table.lookup("msg_001")
        assert entry.tier == StorageTier.L2

    def test_get_by_tier(self):
        """Test getting pages by tier."""
        table = PageTable()

        # Add pages to different tiers
        for i, tier in enumerate([StorageTier.L0, StorageTier.L1, StorageTier.L2]):
            page = MemoryPage(
                page_id=f"msg_{i}",
                modality=Modality.TEXT,
                storage_tier=tier,
            )
            table.register(page)

        l1_pages = table.get_by_tier(StorageTier.L1)
        assert len(l1_pages) == 1
        assert l1_pages[0].page_id == "msg_1"

    def test_dirty_tracking(self):
        """Test dirty page tracking."""
        table = PageTable()
        page = MemoryPage(page_id="msg_001", modality=Modality.TEXT)
        table.register(page)

        assert len(table.get_dirty_pages()) == 0

        table.mark_dirty("msg_001")
        dirty = table.get_dirty_pages()
        assert len(dirty) == 1
        assert dirty[0].page_id == "msg_001"

        table.mark_clean("msg_001")
        assert len(table.get_dirty_pages()) == 0


# =============================================================================
# TLB Tests
# =============================================================================


class TestPageTLB:
    """Tests for PageTLB."""

    def test_insert_and_lookup(self):
        """Test inserting and looking up entries."""
        tlb = PageTLB()
        entry = PageTableEntry(
            page_id="msg_001",
            tier=StorageTier.L1,
            modality=Modality.TEXT,
        )
        tlb.insert(entry)

        result = tlb.lookup("msg_001")
        assert result is not None
        assert result.page_id == "msg_001"
        assert tlb.hits == 1

    def test_miss(self):
        """Test TLB miss."""
        tlb = PageTLB()
        result = tlb.lookup("nonexistent")
        assert result is None
        assert tlb.misses == 1

    def test_lru_eviction(self):
        """Test LRU eviction when full."""
        tlb = PageTLB(max_entries=2)

        # Insert 3 entries
        for i in range(3):
            entry = PageTableEntry(
                page_id=f"msg_{i}",
                tier=StorageTier.L1,
                modality=Modality.TEXT,
            )
            tlb.insert(entry)

        # First entry should be evicted
        assert tlb.lookup("msg_0") is None
        assert tlb.lookup("msg_1") is not None
        assert tlb.lookup("msg_2") is not None

    def test_invalidate(self):
        """Test invalidating an entry."""
        tlb = PageTLB()
        entry = PageTableEntry(
            page_id="msg_001",
            tier=StorageTier.L1,
            modality=Modality.TEXT,
        )
        tlb.insert(entry)

        removed = tlb.invalidate("msg_001")
        assert removed
        assert tlb.lookup("msg_001") is None


class TestTLBWithPageTable:
    """Tests for TLBWithPageTable wrapper."""

    def test_lookup_populates_tlb(self):
        """Test that lookup populates TLB on miss."""
        table = PageTable()
        page = MemoryPage(page_id="msg_001", modality=Modality.TEXT)
        table.register(page)

        wrapper = TLBWithPageTable(table)

        # First lookup - TLB miss, page table hit
        entry = wrapper.lookup("msg_001")
        assert entry is not None
        assert wrapper.tlb.misses == 1

        # Second lookup - TLB hit
        entry = wrapper.lookup("msg_001")
        assert entry is not None
        assert wrapper.tlb.hits == 1


# =============================================================================
# WorkingSetManager Tests
# =============================================================================


class TestWorkingSetManager:
    """Tests for WorkingSetManager."""

    def test_add_to_l0(self):
        """Test adding a page to L0."""
        manager = WorkingSetManager()
        page = MemoryPage(
            page_id="msg_001",
            modality=Modality.TEXT,
            content="Hello, world!",
            size_tokens=100,
        )

        success = manager.add_to_l0(page)
        assert success
        assert manager.is_in_l0("msg_001")
        assert manager.tokens_used == 100

    def test_add_to_l1(self):
        """Test adding a page to L1."""
        manager = WorkingSetManager()
        page = MemoryPage(
            page_id="msg_001",
            modality=Modality.TEXT,
        )

        success = manager.add_to_l1(page)
        assert success
        assert manager.is_in_l1("msg_001")

    def test_eviction_candidates(self):
        """Test getting eviction candidates."""
        manager = WorkingSetManager()

        # Add pages to L0
        for i in range(5):
            page = MemoryPage(
                page_id=f"msg_{i}",
                modality=Modality.TEXT,
                size_tokens=100,
            )
            manager.add_to_l0(page)

        candidates = manager.get_eviction_candidates()
        # First added should be first candidate (lowest score)
        assert len(candidates) > 0
        assert candidates[0][0] == "msg_0"

    def test_needs_eviction(self):
        """Test eviction detection."""
        config = WorkingSetConfig(
            max_l0_tokens=1000,
            reserved_tokens=0,
            eviction_threshold=0.8,
        )
        # Need to set budget limits to match config
        budget = TokenBudget(total_limit=1000, reserved=0)
        manager = WorkingSetManager(config=config, budget=budget)

        # Add pages up to threshold (850/1000 = 0.85 > 0.8)
        page = MemoryPage(
            page_id="msg_001",
            modality=Modality.TEXT,
            size_tokens=850,
        )
        manager.add_to_l0(page)

        assert manager.needs_eviction()


# =============================================================================
# ContextPacker Tests
# =============================================================================


class TestContextPacker:
    """Tests for ContextPacker."""

    def test_pack_text_pages(self):
        """Test packing text pages."""
        packer = ContextPacker()
        pages = [
            MemoryPage(
                page_id="msg_001",
                modality=Modality.TEXT,
                content="Hello from user",
                metadata={"role": "user"},
            ),
            MemoryPage(
                page_id="msg_002",
                modality=Modality.TEXT,
                content="Hello from assistant",
                metadata={"role": "assistant"},
            ),
        ]

        result = packer.pack(pages)
        assert "msg_001" in result.content
        assert "msg_002" in result.content
        assert "Hello from user" in result.content
        assert len(result.pages_included) == 2

    def test_pack_image_page(self):
        """Test packing an image page."""
        packer = ContextPacker()
        page = MemoryPage(
            page_id="img_001",
            modality=Modality.IMAGE,
            caption="A test image",
            dimensions=(800, 600),
        )

        result = packer.pack([page])
        assert "[IMAGE:" in result.content
        assert "800x600" in result.content

    def test_pack_with_wrapper(self):
        """Test packing with VM:CONTEXT wrapper."""
        packer = ContextPacker()
        page = MemoryPage(
            page_id="msg_001",
            modality=Modality.TEXT,
            content="Test",
        )

        result = packer.pack_with_wrapper([page])
        assert result.content.startswith("<VM:CONTEXT>")
        assert result.content.endswith("</VM:CONTEXT>")


# =============================================================================
# ManifestBuilder Tests
# =============================================================================


class TestManifestBuilder:
    """Tests for ManifestBuilder."""

    def test_build_manifest(self):
        """Test building a manifest."""
        table = PageTable()

        # Add working set pages
        for i in range(2):
            page = MemoryPage(
                page_id=f"msg_{i}",
                modality=Modality.TEXT,
                storage_tier=StorageTier.L0,
                size_tokens=100,
            )
            table.register(page)

        # Add available pages
        for i in range(2, 5):
            page = MemoryPage(
                page_id=f"msg_{i}",
                modality=Modality.TEXT,
                storage_tier=StorageTier.L2,
                size_tokens=100,
            )
            table.register(page)

        builder = ManifestBuilder()
        manifest = builder.build(
            session_id="sess_001",
            page_table=table,
            working_set_ids=["msg_0", "msg_1"],
        )

        assert manifest.session_id == "sess_001"
        assert len(manifest.working_set) == 2
        assert len(manifest.available_pages) == 3

    def test_manifest_to_json(self):
        """Test manifest serialization."""
        manifest = VMManifest(
            session_id="sess_001",
            policies=ManifestPolicies(max_faults_per_turn=3),
        )

        json_str = manifest.to_json()
        assert "sess_001" in json_str
        assert "max_faults_per_turn" in json_str

    def test_manifest_wrapped_json(self):
        """Test manifest with VM:MANIFEST_JSON wrapper."""
        manifest = VMManifest(session_id="sess_001")
        wrapped = manifest.to_wrapped_json()
        assert wrapped.startswith("<VM:MANIFEST_JSON>")
        assert wrapped.endswith("</VM:MANIFEST_JSON>")


# =============================================================================
# PageFaultHandler Tests
# =============================================================================


class TestPageFaultHandler:
    """Tests for PageFaultHandler."""

    @pytest.mark.asyncio
    async def test_handle_fault_success(self):
        """Test successful page fault handling."""
        table = PageTable()
        page = MemoryPage(
            page_id="msg_001",
            modality=Modality.TEXT,
            content="Page content",
            storage_tier=StorageTier.L2,
        )
        table.register(page)

        handler = PageFaultHandler()
        handler.configure(page_table=table)
        handler.store_page(page)

        result = await handler.handle_fault("msg_001", target_level=0)
        assert result.success
        assert result.page is not None
        assert result.page.content == "Page content"

    @pytest.mark.asyncio
    async def test_handle_fault_not_found(self):
        """Test page fault for non-existent page."""
        table = PageTable()
        handler = PageFaultHandler()
        handler.configure(page_table=table)

        result = await handler.handle_fault("nonexistent")
        assert not result.success
        assert "nonexistent" in result.error.lower() or "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_fault_limit(self):
        """Test fault limit enforcement."""
        table = PageTable()
        handler = PageFaultHandler(max_faults_per_turn=1)
        handler.configure(page_table=table)

        # First fault
        for i in range(2):
            page = MemoryPage(
                page_id=f"msg_{i}",
                modality=Modality.TEXT,
                content=f"Content {i}",
            )
            table.register(page)
            handler.store_page(page)

        result1 = await handler.handle_fault("msg_0")
        assert result1.success

        # Second fault should fail
        result2 = await handler.handle_fault("msg_1")
        assert not result2.success
        assert "limit" in result2.error.lower()

    def test_build_tool_result(self):
        """Test building tool result envelope."""
        handler = PageFaultHandler()
        page = MemoryPage(
            page_id="msg_001",
            modality=Modality.TEXT,
            content="Test content",
            size_tokens=50,
        )
        fault_result = FaultResult(
            success=True,
            page=page,
            source_tier=StorageTier.L2,
            latency_ms=10.5,
        )

        tool_result = handler.build_tool_result(fault_result)
        # tool_result.page is now a PageData model
        assert tool_result.page.page_id == "msg_001"
        assert tool_result.effects.promoted_to_working_set
        assert tool_result.effects.tokens_est == 50


class TestPageSearchHandler:
    """Tests for PageSearchHandler."""

    @pytest.mark.asyncio
    async def test_search_by_hint(self):
        """Test searching pages by hint."""
        table = PageTable()
        for i in range(5):
            page = MemoryPage(
                page_id=f"msg_{i}",
                modality=Modality.TEXT,
            )
            table.register(page)

        handler = PageSearchHandler()
        handler.configure(page_table=table)
        handler.set_hint("msg_0", "discussion about memory")
        handler.set_hint("msg_1", "memory architecture")
        handler.set_hint("msg_2", "unrelated topic")

        result = await handler.search("memory", limit=10)
        assert len(result.results) == 2
        assert result.total_available == 5


# =============================================================================
# ArtifactsBridge Tests
# =============================================================================


class TestArtifactsBridge:
    """Tests for ArtifactsBridge."""

    @pytest.mark.asyncio
    async def test_in_memory_backend(self):
        """Test in-memory storage backend."""
        bridge = ArtifactsBridge()
        await bridge.configure()

        page = MemoryPage(
            page_id="msg_001",
            modality=Modality.TEXT,
            content="Test content",
        )

        # Store
        artifact_id = await bridge.store_page(page, StorageTier.L3)
        assert artifact_id is not None

        # Load
        loaded = await bridge.load_page(artifact_id)
        assert loaded is not None
        assert loaded.page_id == "msg_001"
        assert loaded.content == "Test content"

    @pytest.mark.asyncio
    async def test_delete_page(self):
        """Test deleting a stored page."""
        bridge = ArtifactsBridge()
        await bridge.configure()

        page = MemoryPage(
            page_id="msg_001",
            modality=Modality.TEXT,
        )

        artifact_id = await bridge.store_page(page)
        deleted = await bridge.delete_page(artifact_id)
        assert deleted

        # Should not be loadable
        loaded = await bridge.load_page(artifact_id)
        assert loaded is None

    @pytest.mark.asyncio
    async def test_checkpoint(self):
        """Test checkpoint storage and loading."""
        bridge = ArtifactsBridge()
        await bridge.configure()

        pages = [MemoryPage(page_id=f"msg_{i}", modality=Modality.TEXT, content=f"Content {i}") for i in range(3)]

        checkpoint_id = await bridge.store_checkpoint(pages, "test_checkpoint")
        assert checkpoint_id is not None

        loaded_pages = await bridge.load_checkpoint(checkpoint_id)
        assert len(loaded_pages) == 3
        assert {p.page_id for p in loaded_pages} == {"msg_0", "msg_1", "msg_2"}


class TestInMemoryBackend:
    """Tests for InMemoryBackend."""

    @pytest.mark.asyncio
    async def test_store_and_load(self):
        """Test basic store and load."""
        backend = InMemoryBackend()
        page = MemoryPage(
            page_id="msg_001",
            modality=Modality.TEXT,
            content="Hello",
        )

        artifact_id = await backend.store(page, StorageTier.L3)
        loaded = await backend.load(artifact_id)

        assert loaded is not None
        assert loaded.page_id == "msg_001"

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing all pages."""
        backend = InMemoryBackend()
        page = MemoryPage(page_id="msg_001", modality=Modality.TEXT)

        artifact_id = await backend.store(page, StorageTier.L3)
        backend.clear()

        loaded = await backend.load(artifact_id)
        assert loaded is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestVMIntegration:
    """Integration tests for the complete VM system."""

    @pytest.mark.asyncio
    async def test_full_fault_flow(self):
        """Test complete page fault flow from storage to context."""
        # Setup storage
        bridge = ArtifactsBridge()
        await bridge.configure()

        # Create and store a page
        page = MemoryPage(
            page_id="msg_001",
            modality=Modality.TEXT,
            content="Important conversation content",
            storage_tier=StorageTier.L3,
        )
        artifact_id = await bridge.store_page(page, StorageTier.L3)

        # Setup page table
        table = PageTable()
        entry = PageTableEntry(
            page_id="msg_001",
            tier=StorageTier.L3,
            artifact_id=artifact_id,
            modality=Modality.TEXT,
        )
        table.register_entry(entry)

        # Setup fault handler
        handler = PageFaultHandler()
        handler.configure(page_table=table)
        handler.store_page(page)  # For simple testing

        # Trigger fault
        result = await handler.handle_fault("msg_001", target_level=0)
        assert result.success
        assert result.page.content == "Important conversation content"

        # Build tool result
        tool_result = handler.build_tool_result(result)
        json_str = tool_result.to_json()
        assert "msg_001" in json_str

    def test_manifest_and_context_generation(self):
        """Test generating both manifest and context."""
        # Setup pages
        table = PageTable()
        working_set_pages = []

        for i in range(3):
            page = MemoryPage(
                page_id=f"msg_{i}",
                modality=Modality.TEXT,
                content=f"Message {i} content",
                storage_tier=StorageTier.L0,
                size_tokens=100,
                metadata={"role": "user" if i % 2 == 0 else "assistant"},
            )
            table.register(page)
            working_set_pages.append(page)

        # Add available pages
        for i in range(3, 6):
            page = MemoryPage(
                page_id=f"msg_{i}",
                modality=Modality.TEXT,
                storage_tier=StorageTier.L2,
            )
            table.register(page)

        # Build manifest
        builder = ManifestBuilder()
        manifest = builder.build(
            session_id="sess_001",
            page_table=table,
            working_set_ids=["msg_0", "msg_1", "msg_2"],
        )

        # Pack context
        packer = ContextPacker()
        context = packer.pack_with_wrapper(working_set_pages)

        # Verify both
        assert len(manifest.working_set) == 3
        assert len(manifest.available_pages) == 3
        assert "<VM:CONTEXT>" in context.content
        assert "msg_0" in context.content


# =============================================================================
# Page Type and Provenance Tests
# =============================================================================


class TestPageTypeProvenance:
    """Tests for page type taxonomy and provenance tracking."""

    def test_page_type_enum(self):
        """Test PageType enum values."""
        assert PageType.TRANSCRIPT == "transcript"
        assert PageType.SUMMARY == "summary"
        assert PageType.ARTIFACT == "artifact"
        assert PageType.CLAIM == "claim"
        assert PageType.PROCEDURE == "procedure"
        assert PageType.INDEX == "index"

    def test_memory_page_with_page_type(self):
        """Test MemoryPage with page_type field."""
        page = MemoryPage(
            page_id="claim_001",
            modality=Modality.TEXT,
            content="User prefers dark mode",
            page_type=PageType.CLAIM,
        )
        assert page.page_type == PageType.CLAIM

    def test_memory_page_with_provenance(self):
        """Test MemoryPage with provenance tracking."""
        # Create a claim derived from a transcript
        page = MemoryPage(
            page_id="claim_001",
            modality=Modality.TEXT,
            content="User prefers dark mode",
            page_type=PageType.CLAIM,
            provenance=["msg_001", "msg_002"],
        )
        assert page.provenance == ["msg_001", "msg_002"]

    def test_memory_page_with_represents(self):
        """Test MemoryPage with represents field for compressed pages."""
        page = MemoryPage(
            page_id="summary_001",
            modality=Modality.TEXT,
            content="Summary of messages 1-10",
            page_type=PageType.SUMMARY,
            represents="msg_001",  # page_id this is a compressed version of
            representation_level=1,  # reduced representation
        )
        assert page.represents == "msg_001"
        assert page.representation_level == 1

    def test_page_table_entry_with_page_type(self):
        """Test PageTableEntry with page_type."""
        entry = PageTableEntry(
            page_id="claim_001",
            tier=StorageTier.L1,
            modality=Modality.TEXT,
            page_type=PageType.CLAIM,
        )
        assert entry.page_type == PageType.CLAIM


# =============================================================================
# Fault Policy Tests
# =============================================================================


class TestFaultPolicy:
    """Tests for fault policy models."""

    def test_fault_policy_defaults(self):
        """Test FaultPolicy default values."""
        policy = FaultPolicy()
        assert policy.max_faults_per_turn == 3
        assert policy.max_fault_tokens_per_turn == 8192

    def test_fault_policy_can_fault(self):
        """Test FaultPolicy can_fault method."""
        policy = FaultPolicy(
            max_faults_per_turn=2,
            max_fault_tokens_per_turn=4000,
        )
        # Can fault initially
        assert policy.can_fault(1000)

        # Record faults
        policy.record_fault(1000)
        policy.record_fault(1500)

        # Cannot fault after reaching limit
        assert not policy.can_fault(1000)

    def test_fault_policy_new_turn(self):
        """Test FaultPolicy new_turn reset."""
        policy = FaultPolicy(max_faults_per_turn=2)
        policy.record_fault(500)
        policy.record_fault(500)

        policy.new_turn()

        assert policy.faults_this_turn == 0
        assert policy.tokens_used_this_turn == 0
        assert policy.can_fault(500)

    def test_fault_record(self):
        """Test FaultRecord model."""
        record = FaultRecord(
            page_id="msg_001",
            turn=5,
            reason=FaultReason.USER_REQUESTED_RECALL,
            tokens_loaded=500,
            latency_ms=25.5,
        )
        assert record.page_id == "msg_001"
        assert record.reason == FaultReason.USER_REQUESTED_RECALL
        assert record.tokens_loaded == 500

    def test_fault_reason_enum(self):
        """Test FaultReason enum values."""
        assert FaultReason.USER_REQUESTED_RECALL == "user_requested_recall"
        assert FaultReason.RESOLVE_REFERENCE == "resolve_reference"
        assert FaultReason.TOOL_PREREQUISITE == "tool_prereq"
        assert FaultReason.SPECULATIVE == "speculative"


# =============================================================================
# Memory ABI Tests
# =============================================================================


class TestMemoryABI:
    """Tests for Memory ABI model."""

    def test_memory_abi_defaults(self):
        """Test MemoryABI default values."""
        abi = MemoryABI()
        assert abi.max_context_tokens == 128000
        assert abi.reserved_tokens == 2000
        assert abi.tool_schema_tokens_reserved == 0
        assert abi.faults_allowed is True

    def test_memory_abi_custom(self):
        """Test MemoryABI with custom values."""
        abi = MemoryABI(
            max_context_tokens=200000,
            reserved_tokens=5000,
            tool_schema_tokens_reserved=1000,
        )
        assert abi.max_context_tokens == 200000
        assert abi.reserved_tokens == 5000
        assert abi.tool_schema_tokens_reserved == 1000

    def test_memory_abi_available_tokens(self):
        """Test available tokens calculation."""
        abi = MemoryABI(
            max_context_tokens=100000,
            reserved_tokens=2000,
            tool_schema_tokens_reserved=500,
        )
        # 100000 - 2000 - 500 = 97500
        assert abi.available_tokens == 97500


# =============================================================================
# User Experience Metrics Tests
# =============================================================================


class TestUserExperienceMetrics:
    """Tests for UX metrics model."""

    def test_recall_success_rate_empty(self):
        """Test recall success rate with no attempts."""
        metrics = UserExperienceMetrics()
        assert metrics.recall_success_rate() == 1.0

    def test_recall_success_rate_with_attempts(self):
        """Test recall success rate calculation."""
        metrics = UserExperienceMetrics()
        metrics.record_recall_attempt(
            turn=1,
            query="What did we discuss?",
            page_ids_cited=["p1"],
            user_corrected=False,
        )
        metrics.record_recall_attempt(turn=2, query="What about X?", page_ids_cited=["p2"], user_corrected=False)
        metrics.record_recall_attempt(turn=3, query="Tell me about Y", page_ids_cited=["p3"], user_corrected=True)
        # 2 successes (not corrected) out of 3 = 2/3
        assert metrics.recall_success_rate() == pytest.approx(2 / 3)

    def test_thrash_index(self):
        """Test thrash index calculation."""
        metrics = UserExperienceMetrics()
        # Record faults - same page faulted multiple times = thrash
        metrics.record_fault("page_a", FaultReason.RESOLVE_REFERENCE, turn=1, tokens_loaded=100)
        metrics.record_fault("page_b", FaultReason.RESOLVE_REFERENCE, turn=2, tokens_loaded=100)
        metrics.record_fault("page_a", FaultReason.RESOLVE_REFERENCE, turn=3, tokens_loaded=100)  # repeat
        metrics.record_fault("page_a", FaultReason.RESOLVE_REFERENCE, turn=4, tokens_loaded=100)  # repeat

        thrash = metrics.thrash_index(window_turns=5)
        # 2 thrash faults (page_a repeated twice) / 5 turns = 0.4
        assert thrash == pytest.approx(2 / 5)

    def test_effective_tokens_ratio(self):
        """Test effective tokens ratio."""
        metrics = UserExperienceMetrics()
        metrics.record_turn_context(
            turn=1,
            tokens_in_context=10000,
            pages_referenced=["p1", "p2", "p3", "p4", "p5"],
        )
        # 5 pages * 200 tokens estimate = 1000 / 10000 = 0.1
        assert metrics.effective_tokens_ratio(1) == pytest.approx(0.1)

    def test_record_fault_for_metrics(self):
        """Test recording faults for tracking."""
        metrics = UserExperienceMetrics()
        metrics.record_fault("page_1", FaultReason.USER_REQUESTED_RECALL, turn=5, tokens_loaded=500)

        assert len(metrics.fault_history) == 1
        assert metrics.fault_history[0].page_id == "page_1"

    def test_fault_reason_breakdown(self):
        """Test getting fault reason breakdown."""
        metrics = UserExperienceMetrics()
        metrics.record_fault("p1", FaultReason.USER_REQUESTED_RECALL, turn=1, tokens_loaded=100)
        metrics.record_fault("p2", FaultReason.RESOLVE_REFERENCE, turn=2, tokens_loaded=100)
        metrics.record_fault("p3", FaultReason.RESOLVE_REFERENCE, turn=3, tokens_loaded=100)

        breakdown = metrics.get_fault_reason_breakdown()
        assert breakdown[FaultReason.USER_REQUESTED_RECALL] == 1
        assert breakdown[FaultReason.RESOLVE_REFERENCE] == 2


# =============================================================================
# Pinned Set Tests
# =============================================================================


class TestPinnedSet:
    """Tests for PinnedSet."""

    def test_pin_and_unpin(self):
        """Test pinning and unpinning pages."""
        pinned = PinnedSet()
        pinned.pin("msg_001")
        assert pinned.is_pinned("msg_001")

        pinned.unpin("msg_001")
        assert not pinned.is_pinned("msg_001")

    def test_auto_pin(self):
        """Test auto-pinning functionality."""
        pinned = PinnedSet(
            auto_pin_last_n_turns=2,
            auto_pin_system_prompt=True,
        )

        # Auto-pin some pages
        pinned.auto_pin("system_prompt")
        pinned.auto_pin("msg_recent")

        assert pinned.is_pinned("system_prompt")
        assert pinned.is_pinned("msg_recent")

    def test_clear_auto_pins(self):
        """Test clearing auto-pins."""
        pinned = PinnedSet()
        pinned.auto_pin("auto_1")
        pinned.auto_pin("auto_2")
        pinned.pin("explicit")

        pinned.clear_auto_pins()

        assert not pinned.is_pinned("auto_1")
        assert not pinned.is_pinned("auto_2")
        assert pinned.is_pinned("explicit")  # Explicit pins remain

    def test_get_all_pinned(self):
        """Test getting all pinned pages."""
        pinned = PinnedSet()
        pinned.pin("msg_001")
        pinned.pin("msg_002")
        pinned.auto_pin("sys")

        all_pinned = pinned.get_all_pinned()
        assert "msg_001" in all_pinned
        assert "msg_002" in all_pinned
        assert "sys" in all_pinned

    def test_count(self):
        """Test counting pinned pages."""
        pinned = PinnedSet()
        assert pinned.count() == 0

        pinned.pin("msg_001")
        pinned.pin("msg_002")
        assert pinned.count() == 2


# =============================================================================
# Anti-Thrash Policy Tests
# =============================================================================


class TestAntiThrashPolicy:
    """Tests for AntiThrashPolicy."""

    def test_eviction_cooldown(self):
        """Test eviction cooldown prevents immediate re-eviction."""
        policy = AntiThrashPolicy(eviction_cooldown_turns=3)

        policy.record_eviction("msg_001", turn=1)

        # Can't evict again within cooldown
        assert not policy.can_evict("msg_001", current_turn=2)
        assert not policy.can_evict("msg_001", current_turn=3)
        # Can evict after cooldown
        assert policy.can_evict("msg_001", current_turn=5)

    def test_fault_protection(self):
        """Test fault protection prevents eviction of recently faulted pages."""
        policy = AntiThrashPolicy(fault_protection_turns=2)

        policy.record_fault("msg_001", turn=5)

        # Can't evict recently faulted page
        assert not policy.can_evict("msg_001", current_turn=5)
        assert not policy.can_evict("msg_001", current_turn=6)
        # Can evict after protection period
        assert policy.can_evict("msg_001", current_turn=8)

    def test_eviction_penalty(self):
        """Test eviction penalty calculation."""
        policy = AntiThrashPolicy()

        # No history - no penalty
        assert policy.get_eviction_penalty("msg_001", current_turn=1) == 0.0

        # Record fault - should add penalty
        policy.record_fault("msg_001", turn=1)
        penalty = policy.get_eviction_penalty("msg_001", current_turn=1)
        assert penalty > 0


# =============================================================================
# Context Pack Cache Tests
# =============================================================================


class TestContextPackCache:
    """Tests for ContextPackCache."""

    def test_put_and_get(self):
        """Test storing and retrieving packed context."""
        from chuk_ai_session_manager.memory.pack_cache import PackedContext

        cache = ContextPackCache(max_entries=10)
        packed = PackedContext(
            vm_context="<VM:CONTEXT>...</VM:CONTEXT>",
            vm_manifest_json="{}",
            page_ids=["msg_001", "msg_002"],
            tokens_used=500,
        )

        working_set_hash = ContextPackCache.compute_working_set_hash(["msg_001", "msg_002"])
        cache.put("sess_001", "gpt-4", 8000, working_set_hash, packed)

        result = cache.get("sess_001", "gpt-4", 8000, working_set_hash)
        assert result is not None
        assert result.page_ids == ["msg_001", "msg_002"]
        assert result.tokens_used == 500

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = ContextPackCache()
        result = cache.get("sess_001", "gpt-4", 8000, "nonexistent_hash")
        assert result is None

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        from chuk_ai_session_manager.memory.pack_cache import PackedContext

        cache = ContextPackCache(max_entries=2)

        for i in range(3):
            packed = PackedContext(
                vm_context=f"context_{i}",
                vm_manifest_json="{}",
            )
            hash_val = f"hash_{i}"
            cache.put(f"sess_{i}", "gpt-4", 8000, hash_val, packed)

        # First entry should be evicted
        assert cache.get("sess_0", "gpt-4", 8000, "hash_0") is None
        # Later entries should still exist
        assert cache.get("sess_1", "gpt-4", 8000, "hash_1") is not None
        assert cache.get("sess_2", "gpt-4", 8000, "hash_2") is not None

    def test_invalidate_session(self):
        """Test invalidating all entries for a session."""
        from chuk_ai_session_manager.memory.pack_cache import PackedContext

        cache = ContextPackCache()

        # Add entries for multiple sessions
        for sess_id in ["sess_a", "sess_a", "sess_b"]:
            packed = PackedContext(
                vm_context="test",
                vm_manifest_json="{}",
            )
            cache.put(sess_id, "gpt-4", 8000, f"hash_{sess_id}_{cache.size}", packed)

        # Invalidate session A
        removed = cache.invalidate_session("sess_a")
        assert removed == 2

    def test_working_set_hash(self):
        """Test working set hash computation."""
        hash1 = ContextPackCache.compute_working_set_hash(["a", "b", "c"])
        hash2 = ContextPackCache.compute_working_set_hash(["c", "b", "a"])
        hash3 = ContextPackCache.compute_working_set_hash(["a", "b", "d"])

        # Same pages, different order should produce same hash
        assert hash1 == hash2
        # Different pages should produce different hash
        assert hash1 != hash3

    def test_hit_rate(self):
        """Test hit rate calculation."""
        from chuk_ai_session_manager.memory.pack_cache import PackedContext

        cache = ContextPackCache()
        packed = PackedContext(vm_context="test", vm_manifest_json="{}")
        cache.put("sess", "gpt-4", 8000, "hash", packed)

        # 2 hits, 1 miss
        cache.get("sess", "gpt-4", 8000, "hash")
        cache.get("sess", "gpt-4", 8000, "hash")
        cache.get("sess", "gpt-4", 8000, "wrong_hash")

        assert cache.hit_rate == pytest.approx(2 / 3)


# =============================================================================
# Mutation Log Tests
# =============================================================================


class TestMutationLogLite:
    """Tests for MutationLogLite."""

    def test_record_mutation(self):
        """Test recording a mutation."""
        log = MutationLogLite(session_id="sess_001")
        mutation = log.record_mutation(
            page_id="msg_001",
            mutation_type=MutationType.CREATE,
            tier_after=StorageTier.L0,
            actor=Actor.USER,
            turn=1,
        )
        assert mutation.page_id == "msg_001"
        assert mutation.mutation_type == MutationType.CREATE
        assert log.mutation_count() == 1

    def test_get_history(self):
        """Test getting mutation history for a page."""
        log = MutationLogLite()

        log.record_mutation("msg_001", MutationType.CREATE, StorageTier.L0, turn=1)
        log.record_mutation("msg_001", MutationType.EVICT, StorageTier.L2, turn=3)
        log.record_mutation("msg_001", MutationType.FAULT_IN, StorageTier.L0, turn=5)

        history = log.get_history("msg_001")
        assert len(history) == 3
        assert history[0].mutation_type == MutationType.CREATE
        assert history[2].mutation_type == MutationType.FAULT_IN

    def test_record_context_at_turn(self):
        """Test recording context snapshot at turn."""
        log = MutationLogLite()

        log.record_context_at_turn(5, ["msg_001", "msg_002", "msg_003"])

        context = log.get_context_at_turn(5)
        assert len(context) == 3
        assert "msg_001" in context

    def test_get_context_missing_turn(self):
        """Test getting context for non-existent turn."""
        log = MutationLogLite()
        context = log.get_context_at_turn(999)
        assert context == []

    def test_get_mutations_by_type(self):
        """Test filtering mutations by type."""
        log = MutationLogLite()

        log.record_mutation("msg_001", MutationType.CREATE, StorageTier.L0)
        log.record_mutation("msg_002", MutationType.EVICT, StorageTier.L2)
        log.record_mutation("msg_003", MutationType.EVICT, StorageTier.L2)

        evictions = log.get_mutations_by_type(MutationType.EVICT)
        assert len(evictions) == 2

    def test_get_mutations_by_actor(self):
        """Test filtering mutations by actor."""
        log = MutationLogLite()

        log.record_mutation("msg_001", MutationType.CREATE, StorageTier.L0, actor=Actor.USER)
        log.record_mutation("msg_002", MutationType.EVICT, StorageTier.L2, actor=Actor.SYSTEM)

        user_mutations = log.get_mutations_by_actor(Actor.USER)
        assert len(user_mutations) == 1

    def test_get_summary(self):
        """Test getting summary statistics."""
        log = MutationLogLite()

        log.record_mutation("msg_001", MutationType.CREATE, StorageTier.L0)
        log.record_mutation("msg_002", MutationType.CREATE, StorageTier.L0)
        log.record_mutation("msg_001", MutationType.EVICT, StorageTier.L2)

        summary = log.get_summary()
        assert summary["total_mutations"] == 3
        assert summary["unique_pages"] == 2
        assert summary["creates"] == 2
        assert summary["evictions"] == 1

    def test_clear(self):
        """Test clearing the log."""
        log = MutationLogLite()
        log.record_mutation("msg_001", MutationType.CREATE, StorageTier.L0)
        log.record_context_at_turn(1, ["msg_001"])

        log.clear()

        assert log.mutation_count() == 0
        assert log.page_count() == 0


# =============================================================================
# Prefetcher Tests
# =============================================================================


class TestSimplePrefetcher:
    """Tests for SimplePrefetcher."""

    def test_record_page_access(self):
        """Test recording page accesses."""
        prefetcher = SimplePrefetcher()
        prefetcher.record_page_access("msg_001")
        prefetcher.record_page_access("msg_001")
        prefetcher.record_page_access("msg_002")

        stats = prefetcher.get_stats()
        assert stats["pages_tracked"] == 2
        assert stats["total_accesses"] == 3

    def test_record_tool_call(self):
        """Test recording tool calls."""
        prefetcher = SimplePrefetcher()
        prefetcher.record_tool_call(
            tool_name="weather_api",
            turn=1,
            pages_accessed_before=["location_claim"],
        )
        prefetcher.record_tool_call(
            tool_name="weather_api",
            turn=3,
            pages_accessed_before=["location_claim", "user_prefs"],
        )

        likely = prefetcher.get_likely_tools()
        assert "weather_api" in likely

        prereqs = prefetcher.get_tool_prereq_pages("weather_api")
        assert "location_claim" in prereqs

    def test_get_top_claims(self):
        """Test getting top accessed claim pages."""
        prefetcher = SimplePrefetcher(max_claims_to_prefetch=2)

        # Simulate page accesses
        for _ in range(5):
            prefetcher.record_page_access("claim_001")
        for _ in range(3):
            prefetcher.record_page_access("claim_002")
        for _ in range(1):
            prefetcher.record_page_access("claim_003")

        # Without page table, returns top pages
        top = prefetcher.get_top_claims(limit=2)
        assert len(top) == 2
        assert top[0] == "claim_001"
        assert top[1] == "claim_002"

    def test_set_last_segment_summary(self):
        """Test setting last segment summary."""
        prefetcher = SimplePrefetcher()
        prefetcher.set_last_segment_summary("summary_001")

        # Should be included in prefetch
        assert prefetcher._last_segment_summary_id == "summary_001"

    @pytest.mark.asyncio
    async def test_prefetch_on_turn_start(self):
        """Test prefetch on turn start."""
        prefetcher = SimplePrefetcher()
        prefetcher.set_last_segment_summary("summary_001")
        prefetcher.record_page_access("claim_001")
        prefetcher.record_page_access("claim_001")

        pages = await prefetcher.prefetch_on_turn_start("sess_001")

        # Should include summary and top claims
        assert "summary_001" in pages
        assert "claim_001" in pages

    def test_clear(self):
        """Test clearing prefetcher state."""
        prefetcher = SimplePrefetcher()
        prefetcher.record_page_access("msg_001")
        prefetcher.record_tool_call("tool", 1)
        prefetcher.set_last_segment_summary("summary")

        prefetcher.clear()

        stats = prefetcher.get_stats()
        assert stats["pages_tracked"] == 0
        assert stats["tools_tracked"] == 0


class TestToolUsagePattern:
    """Tests for ToolUsagePattern model."""

    def test_tool_usage_pattern(self):
        """Test ToolUsagePattern model."""
        pattern = ToolUsagePattern(
            tool_name="weather_api",
            call_count=5,
            last_turn=10,
            prereq_pages=["location", "prefs"],
        )
        assert pattern.tool_name == "weather_api"
        assert pattern.call_count == 5
        assert len(pattern.prereq_pages) == 2


# =============================================================================
# Working Set Manager with Pinning Tests
# =============================================================================


class TestWorkingSetManagerPinning:
    """Tests for WorkingSetManager with pinning and anti-thrash."""

    def test_eviction_respects_pinning(self):
        """Test that pinned pages are not evicted."""
        manager = WorkingSetManager()

        # Add pages
        for i in range(5):
            page = MemoryPage(
                page_id=f"msg_{i}",
                modality=Modality.TEXT,
                size_tokens=100,
            )
            manager.add_to_l0(page)

        # Pin some pages
        manager.pinned_set.pin("msg_0")
        manager.pinned_set.pin("msg_1")

        # Get eviction candidates
        candidates = manager.get_eviction_candidates()
        candidate_ids = [c[0] for c in candidates]

        # Pinned pages should not be in candidates
        assert "msg_0" not in candidate_ids
        assert "msg_1" not in candidate_ids

    def test_current_turn_tracking(self):
        """Test current turn tracking in WorkingSetManager."""
        manager = WorkingSetManager()
        # WorkingSetManager has current_turn field
        assert hasattr(manager, "current_turn")
        assert manager.current_turn == 0

        # Update turn manually
        manager.current_turn = 5
        assert manager.current_turn == 5
