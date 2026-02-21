# chuk_ai_session_manager/memory/manager.py
"""
MemoryManager - Orchestrator for the AI Virtual Memory subsystem.

Ties together all VM primitives into a single cohesive API:
- Page lifecycle (create, add, evict, fault)
- Context building (pack pages into VM:CONTEXT format)
- Turn management (advance counters, snapshot state)
- Metrics and diagnostics

This is the "kernel" that SessionManager talks to.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from .artifacts_bridge import ArtifactsBridge
from .compressor import CompressionResult, CompressorRegistry
from .context_packer import ContextPacker
from .demand_paging import DemandPagingPrePass
from .fault_handler import (
    FaultResult,
    PageFaultHandler,
    PageSearchHandler,
    SearchResult,
)
from .manifest import ManifestBuilder
from .models import (
    Actor,
    CompressionLevel,
    FaultPolicy,
    MemoryPage,
    Modality,
    MutationType,
    PageType,
    StorageTier,
    VMMetrics,
    VMMode,
)
from .mutation_log import MutationLogLite
from .pack_cache import ContextPackCache
from .page_table import PageTable
from .prefetcher import SimplePrefetcher
from .tlb import PageTLB, TLBWithPageTable
from .vm_prompts import build_vm_developer_message, get_vm_tools_as_dicts
from .working_set import WorkingSetConfig, WorkingSetManager

logger = logging.getLogger(__name__)


def event_to_page(
    message: str,
    role: str,
    session_id: str,
    event_id: str | None = None,
    page_type: PageType = PageType.TRANSCRIPT,
    importance: float = 0.5,
) -> MemoryPage:
    """
    Convert session event data into a MemoryPage.

    Pure function that accepts raw strings to avoid circular imports
    between the memory module and session models.

    Args:
        message: The text content.
        role: "user", "assistant", "tool", or "system".
        session_id: The owning session ID.
        event_id: Optional event ID for deterministic page_id generation.
        page_type: Page classification (default TRANSCRIPT).
        importance: Importance score 0.0-1.0 (default 0.5).
    """
    page_id = f"msg_{event_id[:8]}" if event_id else f"msg_{uuid.uuid4().hex[:8]}"

    return MemoryPage(
        page_id=page_id,
        session_id=session_id,
        content=message,
        page_type=page_type,
        modality=Modality.TEXT,
        importance=importance,
        storage_tier=StorageTier.L1,
    )


class MemoryManager:
    """
    Orchestrator for the AI Virtual Memory subsystem.

    Owns and coordinates all VM primitives. Provides a clean API
    for SessionManager integration.

    Usage::

        vm = MemoryManager(session_id="sess_123")

        page = vm.create_page("Hello!", page_type=PageType.TRANSCRIPT)
        await vm.add_to_working_set(page)

        ctx = vm.build_context(system_prompt="You are helpful.")
        # ctx["developer_message"] has VM:RULES + VM:MANIFEST_JSON + VM:CONTEXT
        # ctx["tools"] has page_fault + search_pages tool defs
    """

    def __init__(
        self,
        session_id: str = "",
        config: WorkingSetConfig | None = None,
        fault_policy: FaultPolicy | None = None,
        mode: VMMode = VMMode.STRICT,
        eviction_policy: Any | None = None,
        compressor_registry: CompressorRegistry | None = None,
    ) -> None:
        self._session_id = session_id or str(uuid.uuid4())
        self._mode = mode
        self._turn = 0
        self._lock = asyncio.Lock()

        # Config
        ws_config = config or WorkingSetConfig()
        self._fault_policy = fault_policy or FaultPolicy()

        # Core data structures
        self._page_table = PageTable()
        self._tlb = PageTLB()
        self._tlb_with_pt = TLBWithPageTable(self._page_table, self._tlb)
        self._working_set = WorkingSetManager(config=ws_config)
        self._fault_handler = PageFaultHandler(
            max_faults_per_turn=self._fault_policy.max_faults_per_turn,
        )
        self._context_packer = ContextPacker()
        self._manifest_builder = ManifestBuilder()
        self._mutation_log = MutationLogLite(session_id=self._session_id)
        self._prefetcher = SimplePrefetcher()
        self._pack_cache = ContextPackCache()
        self._metrics = VMMetrics()

        # Eviction policy (optional, delegates to WorkingSetManager default)
        if eviction_policy is not None:
            self._working_set.set_eviction_policy(eviction_policy)

        # Compressor registry (optional, enables compress-before-evict)
        self._compressor_registry = compressor_registry

        # Storage
        self._bridge = ArtifactsBridge(session_id=self._session_id)
        self._bridge_configured = False

        # In-memory page store (pages with content loaded)
        self._page_store: dict[str, MemoryPage] = {}

        # Page hints for search
        self._page_hints: dict[str, str] = {}

        # Search handler
        self._search_handler = PageSearchHandler()
        self._search_handler.configure(page_table=self._page_table)

        # Demand paging pre-pass
        self._demand_pager = DemandPagingPrePass()

        # Wire fault handler with self as PageLoader
        self._fault_handler.configure(
            page_table=self._page_table,
            tlb=self._tlb,
            loader=self,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def mode(self) -> VMMode:
        return self._mode

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def page_table(self) -> PageTable:
        return self._page_table

    @property
    def working_set(self) -> WorkingSetManager:
        return self._working_set

    @property
    def metrics(self) -> VMMetrics:
        return self._metrics

    # ------------------------------------------------------------------
    # PageLoader protocol implementation
    # ------------------------------------------------------------------

    async def load(
        self,
        page_id: str,
        tier: StorageTier,
        artifact_id: str | None = None,
    ) -> MemoryPage | None:
        """PageLoader protocol — load a page from the appropriate tier."""
        # Check in-memory store first
        page = self._page_store.get(page_id)
        if page:
            return page

        # For persistent tiers, delegate to ArtifactsBridge
        if artifact_id and self._bridge_configured:
            return await self._bridge.load_page(artifact_id)

        return None

    # ------------------------------------------------------------------
    # Bridge configuration
    # ------------------------------------------------------------------

    async def configure_bridge(
        self,
        artifact_store: Any | None = None,
        session_id: str | None = None,
    ) -> None:
        """Configure ArtifactsBridge for persistent storage."""
        await self._bridge.configure(
            artifact_store=artifact_store,
            session_id=session_id or self._session_id,
        )
        self._bridge_configured = True

    # ------------------------------------------------------------------
    # Page lifecycle
    # ------------------------------------------------------------------

    def create_page(
        self,
        content: Any,
        page_type: PageType = PageType.TRANSCRIPT,
        modality: Modality = Modality.TEXT,
        importance: float = 0.5,
        provenance: list[str] | None = None,
        page_id: str | None = None,
        size_tokens: int | None = None,
        hint: str | None = None,
    ) -> MemoryPage:
        """
        Create a new MemoryPage, register in page table, store in page_store.

        Does NOT add to working set — call add_to_working_set() separately.
        """
        pid = page_id or f"{page_type.value}_{uuid.uuid4().hex[:8]}"

        page = MemoryPage(
            page_id=pid,
            session_id=self._session_id,
            content=content,
            page_type=page_type,
            modality=modality,
            importance=importance,
            provenance=provenance or [],
            storage_tier=StorageTier.L1,
            size_tokens=size_tokens,
        )

        # Register in page table
        self._page_table.register(page)

        # Store in memory
        self._page_store[pid] = page

        # Store hint for search
        if hint:
            self._page_hints[pid] = hint
            self._search_handler.set_hint(pid, hint)

        # Log mutation
        self._mutation_log.record_mutation(
            page_id=pid,
            mutation_type=MutationType.CREATE,
            tier_after=StorageTier.L1,
            actor=Actor.SYSTEM,
            cause="create_page",
            turn=self._turn,
        )

        # Record access for prefetcher
        self._prefetcher.record_page_access(pid)

        return page

    async def add_to_working_set(self, page: MemoryPage) -> bool:
        """
        Add a page to L0 (context window), running eviction if needed.

        Returns True if successfully added.
        """
        async with self._lock:
            tokens = page.size_tokens or page.estimate_tokens()

            # Run eviction if needed
            if not self._working_set.can_fit(tokens):
                await self._run_eviction(tokens)

            success = self._working_set.add_to_l0(page)
            if success:
                self._page_table.update_location(page.page_id, tier=StorageTier.L0)
                self._tlb.invalidate(page.page_id)

                self._metrics.tokens_in_working_set = self._working_set.tokens_used

                # Invalidate pack cache (working set changed)
                self._pack_cache.invalidate_session(self._session_id)

            return success

    async def evict_page(
        self,
        page_id: str,
        target_tier: StorageTier = StorageTier.L2,
    ) -> bool:
        """Evict a page from working set to a lower tier."""
        page = self._page_store.get(page_id)
        if not page:
            return False

        old_tier = page.storage_tier
        artifact_id = None

        # Store to persistent storage for L3/L4
        if target_tier in (StorageTier.L3, StorageTier.L4) and self._bridge_configured:
            artifact_id = await self._bridge.store_page(page, target_tier)

        # Remove from working set
        self._working_set.remove_from_l0(page_id, page)

        # Update page table
        self._page_table.update_location(page_id, tier=target_tier, artifact_id=artifact_id)
        self._tlb.invalidate(page_id)

        # Update page
        page.storage_tier = target_tier
        if artifact_id:
            page.artifact_id = artifact_id

        # For L3/L4, remove content from memory
        if target_tier in (StorageTier.L3, StorageTier.L4):
            self._page_store.pop(page_id, None)

        # Record
        self._working_set.record_eviction(page_id)
        self._metrics.record_eviction()
        self._mutation_log.record_mutation(
            page_id=page_id,
            mutation_type=MutationType.EVICT,
            tier_before=old_tier,
            tier_after=target_tier,
            actor=Actor.SYSTEM,
            cause="eviction_pressure",
            turn=self._turn,
        )

        return True

    async def compress_page(
        self,
        page_id: str,
        target_level: CompressionLevel,
    ) -> CompressionResult | None:
        """
        Compress a page to the target compression level.

        Updates page in store, page table, mutation log, and metrics.
        Returns None if no compressor registry or page not found.
        """
        if self._compressor_registry is None:
            return None

        page = self._page_store.get(page_id)
        if not page:
            return None

        result = await self._compressor_registry.compress_page(page, target_level)

        if result.tokens_saved > 0:
            # Update page in store
            self._page_store[page_id] = result.page

            # Update working set L0 token accounting
            self._working_set.update_page_tokens(
                page_id,
                old_tokens=result.original_tokens,
                new_tokens=result.compressed_tokens,
                modality=result.page.modality,
            )

            # Update page table
            self._page_table.update_compression(page_id, target_level)

            # Invalidate caches
            self._tlb.invalidate(page_id)
            self._pack_cache.invalidate_session(self._session_id)

            # Record
            self._metrics.record_compression(result.tokens_saved)
            self._mutation_log.record_mutation(
                page_id=page_id,
                mutation_type=MutationType.COMPRESS,
                tier_after=page.storage_tier,
                actor=Actor.SYSTEM,
                cause=f"compress_{target_level.name.lower()}",
                turn=self._turn,
            )

        return result

    async def _run_eviction(self, tokens_needed: int) -> list[str]:
        """Run eviction to free tokens. Returns evicted page IDs."""
        evicted_ids: list[str] = []
        tokens_freed = 0
        target = self._working_set.calculate_eviction_target(tokens_needed)

        candidates = self._working_set.get_eviction_candidates(
            tokens_needed=tokens_needed,
            from_tier=StorageTier.L0,
            page_table=self._page_table,
        )

        for page_id, _score in candidates:
            if tokens_freed >= target:
                break

            page = self._page_store.get(page_id)
            if not page:
                continue

            page_tokens = page.size_tokens or page.estimate_tokens()

            # Try compression before eviction
            if self._compressor_registry is not None and page.compression_level < CompressionLevel.ABSTRACT:
                next_level = CompressionLevel(page.compression_level + 1)
                result = await self.compress_page(page_id, next_level)
                if result and result.tokens_saved > 0:
                    tokens_freed += result.tokens_saved
                    continue  # Skip eviction — compression freed enough

            await self.evict_page(page_id, StorageTier.L2)
            tokens_freed += page_tokens
            evicted_ids.append(page_id)

        return evicted_ids

    async def evict_segment_pages(
        self,
        target_tier: StorageTier = StorageTier.L2,
    ) -> list[str]:
        """
        Evict all non-pinned L0 pages to target tier.

        Used on segment rollover to clear old transcript pages
        while preserving pinned pages (summaries, claims).
        """
        evicted: list[str] = []
        for page_id in list(self._working_set.get_l0_page_ids()):
            if self._working_set.is_pinned(page_id):
                continue
            await self.evict_page(page_id, target_tier)
            evicted.append(page_id)
        return evicted

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search_pages(
        self,
        query: str,
        modality: str | None = None,
        limit: int = 5,
    ) -> SearchResult:
        """Search for pages matching a query using hints."""
        return await self._search_handler.search(query, modality, limit)

    # ------------------------------------------------------------------
    # Demand paging
    # ------------------------------------------------------------------

    async def demand_pre_pass(self, user_message: str) -> list[str]:
        """
        Run DemandPagingPrePass and fault candidate pages into L0.

        Called before the LLM sees the user message, so relevant
        context is already in the working set.

        Pre-fetches are transparent to the model: they do not consume
        the model's per-turn fault budget.

        Returns list of page_ids successfully faulted in.
        """
        candidates = self._demand_pager.get_prefetch_candidates(
            message=user_message,
            page_table=self._page_table,
            page_hints=self._page_hints,
            working_set_ids=set(self._working_set.get_l0_page_ids()),
        )
        faulted: list[str] = []
        for page_id in candidates:
            result = await self.handle_fault(page_id)
            if result.success:
                faulted.append(page_id)

        # Reset per-turn fault budgets — pre-fetches are a system
        # optimisation and should not reduce the model's fault quota.
        self._fault_handler.faults_this_turn = 0
        self._fault_policy.faults_this_turn = 0
        self._fault_policy.tokens_used_this_turn = 0

        return faulted

    # ------------------------------------------------------------------
    # Fault handling
    # ------------------------------------------------------------------

    async def handle_fault(
        self,
        page_id: str,
        target_level: int = 2,
    ) -> FaultResult:
        """
        Handle a page fault request.

        Checks fault policy, delegates to PageFaultHandler,
        and promotes the result to working set.
        """
        estimated_tokens = 500
        if not self._fault_policy.can_fault(estimated_tokens):
            return FaultResult(
                success=False,
                error="Fault policy limit reached for this turn",
            )

        result = await self._fault_handler.handle_fault(page_id, target_level)

        if result.success and result.page:
            page_tokens = result.page.size_tokens or result.page.estimate_tokens()
            self._fault_policy.record_fault(page_tokens)
            self._metrics.record_fault()

            # Store in page_store
            self._page_store[result.page.page_id] = result.page

            # Record anti-thrash
            self._working_set.record_fault(page_id)

            # Add to working set
            await self.add_to_working_set(result.page)

            # Log mutation
            self._mutation_log.record_mutation(
                page_id=page_id,
                mutation_type=MutationType.FAULT_IN,
                tier_after=StorageTier.L0,
                actor=Actor.MODEL,
                cause="page_fault",
                turn=self._turn,
            )

        return result

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def get_l0_pages(self) -> list[MemoryPage]:
        """Get all pages currently in L0 (context window), in order."""
        pages = []
        for page_id in self._working_set.get_l0_page_ids():
            page = self._page_store.get(page_id)
            if page:
                pages.append(page)
        return pages

    def build_context(
        self,
        mode: VMMode | None = None,
        system_prompt: str = "",
        model_id: str = "",
        token_budget: int | None = None,
    ) -> dict[str, Any]:
        """
        Build the complete VM context for an LLM call.

        Returns:
            dict with keys:
            - "developer_message": str — VM:RULES + VM:MANIFEST_JSON + VM:CONTEXT
            - "tools": list — VM tool definitions (page_fault, search_pages)
            - "manifest": VMManifest
            - "packed_context": PackedContext
        """
        vm_mode = mode or self._mode
        l0_pages = self.get_l0_pages()

        # Pack context
        packed = self._context_packer.pack(l0_pages, token_budget)

        # Build manifest
        manifest = self._manifest_builder.build(
            session_id=self._session_id,
            page_table=self._page_table,
            working_set_ids=self._working_set.get_l0_page_ids(),
            hint_generator=lambda entry: self._page_hints.get(entry.page_id, ""),
        )

        manifest_json = manifest.to_json()

        # Build developer message
        developer_message = build_vm_developer_message(
            mode=vm_mode,
            manifest_json=manifest_json,
            context=packed.content,
            system_prompt=system_prompt,
        )

        # Tools (only for non-passive modes)
        tools = get_vm_tools_as_dicts() if vm_mode != VMMode.PASSIVE else []

        return {
            "developer_message": developer_message,
            "tools": tools,
            "manifest": manifest,
            "packed_context": packed,
        }

    # ------------------------------------------------------------------
    # Turn management
    # ------------------------------------------------------------------

    def new_turn(self) -> None:
        """Advance to a new turn. Resets per-turn counters."""
        self._turn += 1
        self._working_set.new_turn()
        self._fault_handler.new_turn()
        self._fault_policy.new_turn()
        self._metrics.new_turn()

        # Snapshot context for mutation log
        self._mutation_log.record_context_at_turn(
            turn=self._turn,
            page_ids=self._working_set.get_l0_page_ids(),
        )

    # ------------------------------------------------------------------
    # Pinning
    # ------------------------------------------------------------------

    def pin_page(self, page_id: str) -> None:
        """Pin a page so it is never evicted."""
        self._working_set.pin_page(page_id)
        self._mutation_log.record_mutation(
            page_id=page_id,
            mutation_type=MutationType.PIN,
            tier_after=StorageTier.L0,
            actor=Actor.SYSTEM,
            cause="explicit_pin",
            turn=self._turn,
        )

    def update_session_id(self, session_id: str) -> None:
        """Update the session ID (e.g. after segment rollover)."""
        self._session_id = session_id

    def set_last_segment_summary(self, page_id: str) -> None:
        """Notify the prefetcher about a new segment summary page."""
        self._prefetcher.set_last_segment_summary(page_id)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive stats across all subsystems."""
        return {
            "session_id": self._session_id,
            "turn": self._turn,
            "mode": self._mode.value,
            "page_table": self._page_table.get_stats().model_dump(),
            "working_set": self._working_set.get_stats().model_dump(),
            "fault_handler": self._fault_handler.get_metrics().model_dump(),
            "tlb": self._tlb.get_stats().model_dump(),
            "mutation_log": self._mutation_log.get_summary(),
            "prefetcher": self._prefetcher.get_stats(),
            "pack_cache": self._pack_cache.get_stats(),
            "metrics": self._metrics.model_dump(),
            "pages_in_store": len(self._page_store),
        }
