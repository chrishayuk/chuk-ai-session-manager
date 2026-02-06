# tests/test_fault_handler_coverage.py
"""Tests for memory/fault_handler.py to achieve >90% coverage."""

import pytest
from unittest.mock import AsyncMock

from chuk_ai_session_manager.memory.fault_handler import (
    FaultResult,
    PageFaultHandler,
    PageSearchHandler,
    SearchResult,
    VMToolError,
    VMToolResult,
)
from chuk_ai_session_manager.memory.models import (
    AudioContent,
    CompressionLevel,
    ImageContent,
    MemoryPage,
    Modality,
    PageData,
    PageMeta,
    PageTableEntry,
    StorageTier,
    StructuredContent,
    TextContent,
    VideoContent,
)
from chuk_ai_session_manager.memory.page_table import PageTable
from chuk_ai_session_manager.memory.tlb import PageTLB


def _make_page(page_id="p1", modality=Modality.TEXT, content="hello", **kw):
    return MemoryPage(
        page_id=page_id,
        session_id="s1",
        modality=modality,
        content=content,
        **kw,
    )


def _make_entry(page_id="p1", tier=StorageTier.L2, modality=Modality.TEXT, **kw):
    return PageTableEntry(page_id=page_id, tier=tier, modality=modality, **kw)


# ── FaultResult / VMToolResult / VMToolError models ──


class TestFaultResult:
    def test_defaults(self):
        r = FaultResult(success=True)
        assert r.success
        assert r.page is None
        assert r.error is None
        assert r.latency_ms == 0.0
        assert r.was_compressed is False

    def test_with_page(self):
        page = _make_page()
        r = FaultResult(success=True, page=page, source_tier=StorageTier.L2)
        assert r.page.page_id == "p1"
        assert r.source_tier == StorageTier.L2


class TestVMToolResult:
    def test_to_json(self):
        tr = VMToolResult(
            page=PageData(
                page_id="p1",
                modality="text",
                level=0,
                tier="l0",
                content=TextContent(text="hi"),
                meta=PageMeta(),
            )
        )
        j = tr.to_json()
        assert "p1" in j
        assert "hi" in j


class TestVMToolError:
    def test_fields(self):
        e = VMToolError(error="oops", page_id="p1")
        assert e.error == "oops"
        assert e.page_id == "p1"


# ── PageFaultHandler ──


class TestPageFaultHandlerBasic:
    def test_configure(self):
        handler = PageFaultHandler()
        pt = PageTable()
        tlb = PageTLB()
        handler.configure(pt, tlb)
        assert handler.page_table is pt
        assert handler.tlb is tlb

    def test_new_turn(self):
        handler = PageFaultHandler()
        handler.faults_this_turn = 5
        handler.new_turn()
        assert handler.faults_this_turn == 0

    def test_can_fault(self):
        handler = PageFaultHandler(max_faults_per_turn=2)
        assert handler.can_fault()
        handler.faults_this_turn = 2
        assert not handler.can_fault()

    def test_store_page(self):
        handler = PageFaultHandler()
        page = _make_page("x1")
        handler.store_page(page)
        assert "x1" in handler.page_store

    def test_get_metrics(self):
        handler = PageFaultHandler(max_faults_per_turn=3)
        handler.faults_this_turn = 1
        m = handler.get_metrics()
        assert m.faults_this_turn == 1
        assert m.max_faults_per_turn == 3
        assert m.faults_remaining == 2


class TestHandleFault:
    @pytest.fixture
    def handler_with_table(self):
        handler = PageFaultHandler()
        pt = PageTable()
        page = _make_page(
            "pg1", compression_level=CompressionLevel.FULL, storage_tier=StorageTier.L2
        )
        handler.store_page(page)
        pt.register(page)
        handler.configure(pt)
        return handler

    @pytest.mark.asyncio
    async def test_fault_limit_exceeded(self):
        handler = PageFaultHandler(max_faults_per_turn=1)
        handler.faults_this_turn = 1
        r = await handler.handle_fault("p1")
        assert not r.success
        assert "limit exceeded" in r.error

    @pytest.mark.asyncio
    async def test_no_page_table(self):
        handler = PageFaultHandler()
        r = await handler.handle_fault("p1")
        assert not r.success
        assert "not configured" in r.error

    @pytest.mark.asyncio
    async def test_page_not_found(self):
        handler = PageFaultHandler()
        handler.configure(PageTable())
        r = await handler.handle_fault("missing")
        assert not r.success
        assert "not found" in r.error

    @pytest.mark.asyncio
    async def test_successful_fault_no_tlb(self, handler_with_table):
        r = await handler_with_table.handle_fault("pg1", target_level=0)
        assert r.success
        assert r.page.page_id == "pg1"
        assert r.source_tier == StorageTier.L2

    @pytest.mark.asyncio
    async def test_successful_fault_with_compression(self, handler_with_table):
        # Target a different level than the page's current
        r = await handler_with_table.handle_fault("pg1", target_level=2)
        assert r.success
        assert r.was_compressed

    @pytest.mark.asyncio
    async def test_fault_with_tlb_hit(self):
        handler = PageFaultHandler()
        pt = PageTable()
        tlb = PageTLB()
        page = _make_page("pg1", storage_tier=StorageTier.L2)
        handler.store_page(page)
        entry = pt.register(page)
        tlb.insert(entry)  # Pre-populate TLB
        handler.configure(pt, tlb)

        r = await handler.handle_fault("pg1", target_level=0)
        assert r.success
        assert handler.metrics.tlb_hit_rate > 0

    @pytest.mark.asyncio
    async def test_fault_with_tlb_miss(self):
        handler = PageFaultHandler()
        pt = PageTable()
        tlb = PageTLB()
        page = _make_page("pg1", storage_tier=StorageTier.L2)
        handler.store_page(page)
        pt.register(page)
        # TLB is empty - miss
        handler.configure(pt, tlb)

        r = await handler.handle_fault("pg1", target_level=0)
        assert r.success
        # TLB miss recorded, entry inserted after

    @pytest.mark.asyncio
    async def test_load_failure(self):
        handler = PageFaultHandler()
        pt = PageTable()
        page = _make_page("pg1", storage_tier=StorageTier.L2)
        pt.register(page)
        handler.configure(pt)
        # No page in page_store → load returns None
        r = await handler.handle_fault("pg1")
        assert not r.success
        assert "Failed to load" in r.error

    @pytest.mark.asyncio
    async def test_custom_loader(self):
        handler = PageFaultHandler()
        pt = PageTable()
        page = _make_page("pg1", storage_tier=StorageTier.L2)
        pt.register(page)

        loader = AsyncMock()
        loader.load = AsyncMock(return_value=_make_page("pg1"))
        handler.configure(pt, loader=loader)

        r = await handler.handle_fault("pg1", target_level=0)
        assert r.success
        loader.load.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_custom_compressor(self):
        handler = PageFaultHandler()
        pt = PageTable()
        page = _make_page(
            "pg1", compression_level=CompressionLevel.FULL, storage_tier=StorageTier.L2
        )
        handler.store_page(page)
        pt.register(page)

        compressed = _make_page("pg1", compression_level=CompressionLevel.ABSTRACT)
        compressor = AsyncMock()
        compressor.compress = AsyncMock(return_value=compressed)
        handler.configure(pt, compressor=compressor)

        r = await handler.handle_fault("pg1", target_level=2)
        assert r.success
        assert r.was_compressed
        compressor.compress.assert_awaited_once()


# ── build_tool_result ──


class TestBuildToolResult:
    def test_error_result(self):
        handler = PageFaultHandler()
        fr = FaultResult(success=False, error="boom")
        tr = handler.build_tool_result(fr)
        assert tr.page.page_id == "error"
        assert "boom" in tr.page.content.text
        assert tr.effects.promoted_to_working_set is False

    def test_success_text(self):
        handler = PageFaultHandler()
        page = _make_page("pg1", content="hello", size_tokens=50)
        fr = FaultResult(
            success=True, page=page, source_tier=StorageTier.L2, latency_ms=5.0
        )
        tr = handler.build_tool_result(fr, evictions=["old1"])
        assert tr.page.page_id == "pg1"
        assert tr.effects.promoted_to_working_set
        assert tr.effects.evictions == ["old1"]

    def test_success_with_int_compression_level(self):
        handler = PageFaultHandler()
        page = _make_page("pg1", compression_level=0)
        fr = FaultResult(success=True, page=page, source_tier=StorageTier.L2)
        tr = handler.build_tool_result(fr)
        assert tr.page.level == 0


# ── _format_content_for_modality ──


class TestFormatContentForModality:
    def test_text(self):
        handler = PageFaultHandler()
        page = _make_page("p1", content="hi")
        result = handler._format_content_for_modality(page)
        assert isinstance(result, TextContent)
        assert result.text == "hi"

    def test_text_none_content(self):
        handler = PageFaultHandler()
        page = _make_page("p1", content=None)
        result = handler._format_content_for_modality(page)
        assert isinstance(result, TextContent)
        assert result.text == ""

    def test_image_with_caption(self):
        handler = PageFaultHandler()
        page = _make_page("p1", modality=Modality.IMAGE, content=None, caption="A cat")
        result = handler._format_content_for_modality(page)
        assert isinstance(result, ImageContent)
        assert result.caption == "A cat"

    def test_image_url(self):
        handler = PageFaultHandler()
        page = _make_page("p1", modality=Modality.IMAGE, content="http://img.png")
        result = handler._format_content_for_modality(page)
        assert isinstance(result, ImageContent)
        assert result.url == "http://img.png"

    def test_image_base64(self):
        handler = PageFaultHandler()
        page = _make_page(
            "p1", modality=Modality.IMAGE, content="data:image/png;base64,abc"
        )
        result = handler._format_content_for_modality(page)
        assert isinstance(result, ImageContent)
        assert result.base64 == "data:image/png;base64,abc"

    def test_image_plain_text(self):
        handler = PageFaultHandler()
        page = _make_page("p1", modality=Modality.IMAGE, content="a diagram")
        result = handler._format_content_for_modality(page)
        assert isinstance(result, ImageContent)
        assert result.caption == "a diagram"

    def test_audio_with_transcript(self):
        handler = PageFaultHandler()
        page = _make_page(
            "p1",
            modality=Modality.AUDIO,
            content=None,
            transcript="hello world",
            duration_seconds=90.0,
        )
        result = handler._format_content_for_modality(page)
        assert isinstance(result, AudioContent)
        assert result.transcript == "hello world"
        assert result.duration_seconds == 90.0

    def test_audio_content_as_transcript(self):
        handler = PageFaultHandler()
        page = _make_page("p1", modality=Modality.AUDIO, content="spoken words")
        result = handler._format_content_for_modality(page)
        assert isinstance(result, AudioContent)
        assert result.transcript == "spoken words"

    def test_video(self):
        handler = PageFaultHandler()
        page = _make_page(
            "p1",
            modality=Modality.VIDEO,
            content=None,
            transcript="video text",
            duration_seconds=120.0,
            metadata={"scenes": ["s1", "s2"]},
        )
        result = handler._format_content_for_modality(page)
        assert isinstance(result, VideoContent)
        assert result.transcript == "video text"
        assert result.duration_seconds == 120.0
        assert result.scenes == ["s1", "s2"]

    def test_video_no_scenes(self):
        handler = PageFaultHandler()
        page = _make_page("p1", modality=Modality.VIDEO, content=None)
        result = handler._format_content_for_modality(page)
        assert isinstance(result, VideoContent)

    def test_structured_dict(self):
        handler = PageFaultHandler()
        page = _make_page("p1", modality=Modality.STRUCTURED, content={"key": "val"})
        result = handler._format_content_for_modality(page)
        assert isinstance(result, StructuredContent)
        assert result.data == {"key": "val"}

    def test_structured_non_dict(self):
        handler = PageFaultHandler()
        page = _make_page("p1", modality=Modality.STRUCTURED, content="not a dict")
        result = handler._format_content_for_modality(page)
        assert isinstance(result, StructuredContent)
        assert result.data == {}

    def test_unknown_modality(self):
        """Unknown modality falls through to else branch returning TextContent."""
        handler = PageFaultHandler()
        # Use a modality that has no dedicated handler path - we can't easily
        # create one without extending the enum, so we test with content=None
        # on a text page and verify the fallback. Let's test the else branch
        # by creating a page with None content for a custom scenario.
        page = _make_page("p1", content="fallback")
        # Monkeypatch modality to something unhandled
        object.__setattr__(page, "modality", "other")
        result = handler._format_content_for_modality(page)
        assert isinstance(result, TextContent)
        assert result.text == "fallback"


# ── _build_meta ──


class TestBuildMeta:
    def test_basic(self):
        handler = PageFaultHandler()
        page = _make_page("p1")
        fr = FaultResult(success=True, source_tier=StorageTier.L2, latency_ms=5.0)
        meta = handler._build_meta(page, fr)
        assert meta.source_tier == StorageTier.L2.value
        assert meta.latency_ms == 5.0

    def test_with_mime_type(self):
        handler = PageFaultHandler()
        page = _make_page("p1", mime_type="image/png")
        fr = FaultResult(success=True, source_tier=StorageTier.L1)
        meta = handler._build_meta(page, fr)
        assert meta.mime_type == "image/png"

    def test_with_size_bytes(self):
        handler = PageFaultHandler()
        page = _make_page("p1", size_bytes=1024)
        fr = FaultResult(success=True, source_tier=StorageTier.L1)
        meta = handler._build_meta(page, fr)
        assert meta.size_bytes == 1024

    def test_with_dimensions(self):
        handler = PageFaultHandler()
        page = _make_page("p1", modality=Modality.IMAGE, dimensions=(800, 600))
        fr = FaultResult(success=True, source_tier=StorageTier.L1)
        meta = handler._build_meta(page, fr)
        assert meta.dimensions == [800, 600]

    def test_with_duration(self):
        handler = PageFaultHandler()
        page = _make_page("p1", modality=Modality.AUDIO, duration_seconds=42.5)
        fr = FaultResult(success=True, source_tier=StorageTier.L1)
        meta = handler._build_meta(page, fr)
        assert meta.duration_seconds == 42.5

    def test_no_source_tier(self):
        handler = PageFaultHandler()
        page = _make_page("p1")
        fr = FaultResult(success=True, source_tier=None)
        meta = handler._build_meta(page, fr)
        assert meta.source_tier == "unknown"


# ── SearchResult ──


class TestSearchResult:
    def test_to_json(self):
        sr = SearchResult(results=[], total_available=3)
        j = sr.to_json()
        assert '"total_available":3' in j or '"total_available": 3' in j


# ── PageSearchHandler ──


class TestPageSearchHandler:
    def test_configure(self):
        sh = PageSearchHandler()
        pt = PageTable()
        sh.configure(pt)
        assert sh.page_table is pt

    def test_set_hint(self):
        sh = PageSearchHandler()
        sh.set_hint("p1", "cats and dogs")
        assert sh.page_hints["p1"] == "cats and dogs"

    @pytest.mark.asyncio
    async def test_search_no_page_table(self):
        sh = PageSearchHandler()
        r = await sh.search("test")
        assert r.results == []
        assert r.total_available == 0

    @pytest.mark.asyncio
    async def test_search_custom_fn(self):
        sh = PageSearchHandler()
        pt = PageTable()
        custom = AsyncMock(return_value=SearchResult(results=[], total_available=5))
        sh.configure(pt, search_fn=custom)
        r = await sh.search("query", modality="text", limit=3)
        custom.assert_awaited_once_with("query", "text", 3)
        assert r.total_available == 5

    @pytest.mark.asyncio
    async def test_search_hint_match(self):
        sh = PageSearchHandler()
        pt = PageTable()
        entry = _make_entry("p1", tier=StorageTier.L2)
        pt.register_entry(entry)
        sh.configure(pt)
        sh.set_hint("p1", "cats and dogs")

        r = await sh.search("cats")
        assert len(r.results) == 1
        assert r.results[0].page_id == "p1"
        assert r.results[0].relevance == 0.8

    @pytest.mark.asyncio
    async def test_search_page_id_match(self):
        sh = PageSearchHandler()
        pt = PageTable()
        entry = _make_entry("cats_page", tier=StorageTier.L2)
        pt.register_entry(entry)
        sh.configure(pt)

        r = await sh.search("cats")
        assert len(r.results) == 1
        assert r.results[0].relevance == 1.0

    @pytest.mark.asyncio
    async def test_search_modality_filter(self):
        sh = PageSearchHandler()
        pt = PageTable()
        pt.register_entry(_make_entry("p1", modality=Modality.TEXT))
        pt.register_entry(_make_entry("p2", modality=Modality.IMAGE))
        sh.configure(pt)
        sh.set_hint("p1", "hello")
        sh.set_hint("p2", "hello")

        r = await sh.search("hello", modality="image")
        assert len(r.results) == 1
        assert r.results[0].page_id == "p2"

    @pytest.mark.asyncio
    async def test_search_limit(self):
        sh = PageSearchHandler()
        pt = PageTable()
        for i in range(10):
            pt.register_entry(_make_entry(f"p{i}", modality=Modality.TEXT))
            sh.set_hint(f"p{i}", "match")
        sh.configure(pt)

        r = await sh.search("match", limit=3)
        assert len(r.results) == 3

    @pytest.mark.asyncio
    async def test_search_no_match(self):
        sh = PageSearchHandler()
        pt = PageTable()
        pt.register_entry(_make_entry("p1"))
        sh.configure(pt)
        sh.set_hint("p1", "dogs")

        r = await sh.search("xyz")
        assert len(r.results) == 0

    @pytest.mark.asyncio
    async def test_search_results_sorted_by_relevance(self):
        sh = PageSearchHandler()
        pt = PageTable()
        # p1 matches via hint (0.8), cats_page matches via id (1.0)
        pt.register_entry(_make_entry("p1", modality=Modality.TEXT))
        pt.register_entry(_make_entry("cats_page", modality=Modality.TEXT))
        sh.configure(pt)
        sh.set_hint("p1", "cats rule")

        r = await sh.search("cats")
        assert len(r.results) == 2
        assert r.results[0].page_id == "cats_page"  # Higher relevance
        assert r.results[1].page_id == "p1"
