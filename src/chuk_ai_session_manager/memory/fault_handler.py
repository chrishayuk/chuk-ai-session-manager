# chuk_ai_session_manager/memory/fault_handler.py
"""
Page Fault Handler for AI Virtual Memory.

The PageFaultHandler resolves requests for pages not currently in L0.
When the model calls page_fault(page_id, target_level), this handler:
1. Looks up the page location
2. Loads from the appropriate tier
3. Compresses to the requested level
4. Returns the canonical tool result envelope

Design principles:
- Async-native: All I/O operations are async
- Pydantic-native: All models are BaseModel subclasses
- No magic strings: Uses enums for all categorical values
- Metrics-aware: Tracks fault counts and latencies
"""

import time
from typing import Callable, Dict, List, Optional, Protocol

from pydantic import BaseModel, Field, PrivateAttr

from .models import (
    ALL_COMPRESSION_LEVELS,
    AudioContent,
    CompressionLevel,
    FaultEffects,
    FaultMetrics,
    ImageContent,
    MemoryPage,
    Modality,
    PageData,
    PageMeta,
    SearchResultEntry,
    StorageTier,
    StructuredContent,
    TextContent,
    VideoContent,
    VMMetrics,
)
from .page_table import PageTable
from .tlb import PageTLB


class PageLoader(Protocol):
    """Protocol for loading pages from storage tiers."""

    async def load(
        self,
        page_id: str,
        tier: StorageTier,
        artifact_id: Optional[str] = None,
    ) -> Optional[MemoryPage]:
        """Load a page from storage."""
        ...


class PageCompressor(Protocol):
    """Protocol for compressing pages to different levels."""

    async def compress(
        self,
        page: MemoryPage,
        target_level: CompressionLevel,
    ) -> MemoryPage:
        """Compress a page to the target level."""
        ...


class FaultResult(BaseModel):
    """Result of a page fault resolution."""

    success: bool
    page: Optional[MemoryPage] = None
    error: Optional[str] = None
    source_tier: Optional[StorageTier] = None
    latency_ms: float = 0.0
    was_compressed: bool = False


class VMToolResult(BaseModel):
    """
    Canonical envelope for VM tool results.

    This is the format returned to the model via role="tool" messages.
    """

    page: PageData
    effects: FaultEffects = Field(default_factory=FaultEffects)

    def to_json(self) -> str:
        """Serialize to JSON for tool response."""
        return self.model_dump_json()


class VMToolError(BaseModel):
    """Error response for VM tool calls."""

    error: str
    page_id: Optional[str] = None


class PageFaultHandler(BaseModel):
    """
    Handles page fault resolution.

    When the model needs a page that's not in the working set,
    this handler fetches it from the appropriate tier.
    """

    # Dependencies (set after construction)
    page_table: Optional[PageTable] = None
    tlb: Optional[PageTLB] = None

    # Page storage (maps page_id -> MemoryPage for L2+ pages)
    # In a deployed system, this would be replaced by ArtifactsBridge
    page_store: Dict[str, MemoryPage] = Field(default_factory=dict)

    # Metrics
    metrics: VMMetrics = Field(default_factory=VMMetrics)

    # Configuration
    max_faults_per_turn: int = Field(
        default=2, description="Maximum faults allowed per turn"
    )
    faults_this_turn: int = Field(default=0, description="Faults issued this turn")

    # Optional async loader/compressor (private attrs)
    _loader: Optional[PageLoader] = PrivateAttr(default=None)
    _compressor: Optional[PageCompressor] = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    def configure(
        self,
        page_table: PageTable,
        tlb: Optional[PageTLB] = None,
        loader: Optional[PageLoader] = None,
        compressor: Optional[PageCompressor] = None,
    ) -> None:
        """Configure the handler with dependencies."""
        self.page_table = page_table
        self.tlb = tlb
        self._loader = loader
        self._compressor = compressor

    def new_turn(self) -> None:
        """Reset per-turn counters."""
        self.faults_this_turn = 0
        self.metrics.new_turn()

    def can_fault(self) -> bool:
        """Check if more faults are allowed this turn."""
        return self.faults_this_turn < self.max_faults_per_turn

    async def handle_fault(
        self,
        page_id: str,
        target_level: int = 2,
    ) -> FaultResult:
        """
        Handle a page fault request.

        Args:
            page_id: ID of the page to load
            target_level: Compression level (0=full, 1=reduced, 2=abstract, 3=ref)

        Returns:
            FaultResult with the loaded page or error
        """
        start_time = time.time()

        # Check fault limit
        if not self.can_fault():
            return FaultResult(
                success=False,
                error=f"Fault limit exceeded ({self.max_faults_per_turn} per turn)",
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Ensure we have a page table
        if self.page_table is None:
            return FaultResult(
                success=False,
                error="PageFaultHandler not configured",
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Look up page entry (TLB first, then page table)
        entry = None
        if self.tlb:
            entry = self.tlb.lookup(page_id)
            if entry:
                self.metrics.record_tlb_hit()
            else:
                self.metrics.record_tlb_miss()
                entry = self.page_table.lookup(page_id)
                if entry:
                    self.tlb.insert(entry)
        else:
            entry = self.page_table.lookup(page_id)

        if not entry:
            return FaultResult(
                success=False,
                error=f"Page not found: {page_id}",
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Load the page
        page = await self._load_page(page_id, entry.tier, entry.artifact_id)
        if not page:
            return FaultResult(
                success=False,
                error=f"Failed to load page from {entry.tier.value}",
                source_tier=entry.tier,
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Compress if needed
        target = CompressionLevel(target_level)
        was_compressed = False
        if page.compression_level != target:
            page = await self._compress_page(page, target)
            was_compressed = True

        # Update metrics
        self.faults_this_turn += 1
        self.metrics.record_fault()
        self.page_table.mark_accessed(page_id)

        latency = (time.time() - start_time) * 1000
        return FaultResult(
            success=True,
            page=page,
            source_tier=entry.tier,
            latency_ms=latency,
            was_compressed=was_compressed,
        )

    async def _load_page(
        self,
        page_id: str,
        tier: StorageTier,
        artifact_id: Optional[str],
    ) -> Optional[MemoryPage]:
        """Load a page from storage."""
        # Use custom loader if available
        if self._loader:
            return await self._loader.load(page_id, tier, artifact_id)

        # Default: check in-memory store
        return self.page_store.get(page_id)

    async def _compress_page(
        self,
        page: MemoryPage,
        target_level: CompressionLevel,
    ) -> MemoryPage:
        """Compress a page to the target level."""
        # Use custom compressor if available
        if self._compressor:
            return await self._compressor.compress(page, target_level)

        # Default: stub compression (just update level, don't transform)
        page.compression_level = target_level
        return page

    def store_page(self, page: MemoryPage) -> None:
        """Store a page in the local page store (for testing/simple usage)."""
        self.page_store[page.page_id] = page

    def build_tool_result(
        self,
        fault_result: FaultResult,
        evictions: Optional[List[str]] = None,
    ) -> VMToolResult:
        """
        Build the canonical tool result envelope for a fault result.

        This is what gets returned to the model in the tool response.
        """
        if not fault_result.success or not fault_result.page:
            # Return error as a minimal PageData
            return VMToolResult(
                page=PageData(
                    page_id="error",
                    modality=Modality.TEXT,
                    level=CompressionLevel.FULL,
                    tier=StorageTier.L0,
                    content=TextContent(text=fault_result.error or "Unknown error"),
                    meta=PageMeta(),
                ),
                effects=FaultEffects(promoted_to_working_set=False),
            )

        page = fault_result.page
        content = self._format_content_for_modality(page)
        meta = self._build_meta(page, fault_result)

        return VMToolResult(
            page=PageData(
                page_id=page.page_id,
                modality=page.modality,
                level=(
                    CompressionLevel(page.compression_level)
                    if isinstance(page.compression_level, int)
                    else page.compression_level
                ),
                tier=StorageTier.L1,  # Promoted to L1 after fault
                content=content,
                meta=meta,
            ),
            effects=FaultEffects(
                promoted_to_working_set=True,
                tokens_est=page.size_tokens or page.estimate_tokens(),
                evictions=evictions,
            ),
        )

    def _format_content_for_modality(
        self, page: MemoryPage
    ) -> TextContent | ImageContent | AudioContent | VideoContent | StructuredContent:
        """Format page content based on modality."""
        if page.modality == Modality.TEXT:
            return TextContent(text=page.content or "")

        elif page.modality == Modality.IMAGE:
            result = ImageContent()
            if page.caption:
                result.caption = page.caption
            if page.content and isinstance(page.content, str):
                if page.content.startswith("http"):
                    result.url = page.content
                elif page.content.startswith("data:"):
                    result.base64 = page.content
                else:
                    result.caption = page.content
            return result

        elif page.modality == Modality.AUDIO:
            result = AudioContent()
            if page.transcript:
                result.transcript = page.transcript
            elif page.content and isinstance(page.content, str):
                result.transcript = page.content
            if page.duration_seconds:
                result.duration_seconds = page.duration_seconds
            return result

        elif page.modality == Modality.VIDEO:
            result = VideoContent()
            if page.transcript:
                result.transcript = page.transcript
            if page.duration_seconds:
                result.duration_seconds = page.duration_seconds
            scenes = page.metadata.get("scenes", [])
            if scenes:
                result.scenes = scenes
            return result

        elif page.modality == Modality.STRUCTURED:
            return StructuredContent(
                data=page.content if isinstance(page.content, dict) else {}
            )

        else:
            return TextContent(text=str(page.content) if page.content else "")

    def _build_meta(
        self,
        page: MemoryPage,
        fault_result: FaultResult,
    ) -> PageMeta:
        """Build metadata for tool result."""
        meta = PageMeta(
            source_tier=fault_result.source_tier if fault_result.source_tier else None,
        )

        if page.mime_type:
            meta.mime_type = page.mime_type

        if page.size_bytes:
            meta.size_bytes = page.size_bytes

        if page.dimensions:
            meta.dimensions = list(page.dimensions)

        if page.duration_seconds:
            meta.duration_seconds = page.duration_seconds

        if fault_result.latency_ms:
            meta.latency_ms = round(fault_result.latency_ms, 2)

        return meta

    def get_metrics(self) -> FaultMetrics:
        """Get fault handler metrics."""
        return FaultMetrics(
            faults_this_turn=self.faults_this_turn,
            max_faults_per_turn=self.max_faults_per_turn,
            faults_remaining=self.max_faults_per_turn - self.faults_this_turn,
            total_faults=self.metrics.faults_total,
            tlb_hit_rate=self.metrics.tlb_hit_rate,
        )


class SearchResult(BaseModel):
    """Result of a page search operation."""

    results: List[SearchResultEntry] = Field(default_factory=list)
    total_available: int = 0

    def to_json(self) -> str:
        """Serialize to JSON for tool response."""
        return self.model_dump_json()


class PageSearchHandler(BaseModel):
    """
    Handles search_pages tool calls.

    Searches available pages by query and returns metadata (not content).
    """

    page_table: Optional[PageTable] = None

    # Optional search function
    _search_fn: Optional[Callable] = PrivateAttr(default=None)

    # Page hints (for simple text search)
    page_hints: Dict[str, str] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    def configure(
        self,
        page_table: PageTable,
        search_fn: Optional[Callable] = None,
    ) -> None:
        """Configure the search handler."""
        self.page_table = page_table
        self._search_fn = search_fn

    def set_hint(self, page_id: str, hint: str) -> None:
        """Set a search hint for a page."""
        self.page_hints[page_id] = hint

    async def search(
        self,
        query: str,
        modality: Optional[str] = None,
        limit: int = 5,
    ) -> SearchResult:
        """
        Search for pages matching a query.

        Args:
            query: Search query (keyword or semantic)
            modality: Optional filter by modality
            limit: Maximum results

        Returns:
            SearchResult with matching page metadata
        """
        if self.page_table is None:
            return SearchResult(results=[], total_available=0)

        # Use custom search if available
        if self._search_fn:
            return await self._search_fn(query, modality, limit)

        # Default: simple text matching on hints
        results: List[SearchResultEntry] = []
        query_lower = query.lower()

        for page_id, entry in self.page_table.entries.items():
            # Filter by modality if specified
            if modality:
                try:
                    if entry.modality != Modality(modality):
                        continue
                except ValueError:
                    continue

            # Check hint for match
            hint = self.page_hints.get(page_id, "")
            if query_lower in hint.lower() or query_lower in page_id.lower():
                relevance = 1.0 if query_lower in page_id.lower() else 0.8
                results.append(
                    SearchResultEntry(
                        page_id=page_id,
                        modality=entry.modality.value,
                        tier=entry.tier.value,
                        levels=ALL_COMPRESSION_LEVELS,
                        hint=hint,
                        relevance=relevance,
                    )
                )

            if len(results) >= limit:
                break

        # Sort by relevance
        results.sort(key=lambda x: x.relevance, reverse=True)

        return SearchResult(
            results=results[:limit],
            total_available=len(self.page_table.entries),
        )
