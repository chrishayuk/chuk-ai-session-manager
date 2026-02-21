# tests/test_compressor.py
"""
Tests for the compressor module and its integration with MemoryManager.

Covers:
- CompressionResult creation and properties
- TextCompressor across all compression level transitions
- PassthroughCompressor (no-op) behaviour
- ImageCompressor with captions and references
- CompressorRegistry routing and defaults
- MemoryManager.compress_page() integration
- Eviction-with-compression flow in MemoryManager._run_eviction()
"""

import pytest

from chuk_ai_session_manager.memory.compressor import (
    CompressionResult,
    CompressorRegistry,
    ImageCompressor,
    PassthroughCompressor,
    TextCompressor,
)
from chuk_ai_session_manager.memory.manager import MemoryManager
from chuk_ai_session_manager.memory.models import (
    CompressionLevel,
    MemoryPage,
    Modality,
    MutationType,
    PageType,
    TokenBudget,
)
from chuk_ai_session_manager.memory.working_set import WorkingSetConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_page(
    page_id: str = "p1",
    content: str = "Hello world " * 100,
    compression_level: CompressionLevel = CompressionLevel.FULL,
    page_type: PageType = PageType.TRANSCRIPT,
    modality: Modality = Modality.TEXT,
    size_tokens: int | None = None,
    caption: str | None = None,
) -> MemoryPage:
    """Build a MemoryPage suitable for text compression tests."""
    return MemoryPage(
        page_id=page_id,
        session_id="s1",
        content=content,
        page_type=page_type,
        modality=modality,
        compression_level=compression_level,
        size_tokens=size_tokens,
        caption=caption,
    )


def _image_page(
    page_id: str = "img1",
    caption: str | None = "A cat sitting on a mat",
    size_tokens: int = 200,
) -> MemoryPage:
    """Build a MemoryPage suitable for image compression tests."""
    return MemoryPage(
        page_id=page_id,
        session_id="s1",
        content="<binary-placeholder>",
        page_type=PageType.ARTIFACT,
        modality=Modality.IMAGE,
        compression_level=CompressionLevel.FULL,
        size_tokens=size_tokens,
        caption=caption,
    )


# ===========================================================================
# TestCompressionResult
# ===========================================================================


class TestCompressionResult:
    """Tests for CompressionResult dataclass-like model."""

    def test_creation(self):
        page = _text_page()
        result = CompressionResult(
            page=page,
            original_tokens=400,
            compressed_tokens=200,
            level_before=CompressionLevel.FULL,
            level_after=CompressionLevel.REDUCED,
        )
        assert result.original_tokens == 400
        assert result.compressed_tokens == 200
        assert result.level_before == CompressionLevel.FULL
        assert result.level_after == CompressionLevel.REDUCED

    def test_tokens_saved_property(self):
        page = _text_page()
        result = CompressionResult(
            page=page,
            original_tokens=1000,
            compressed_tokens=350,
            level_before=CompressionLevel.FULL,
            level_after=CompressionLevel.ABSTRACT,
        )
        assert result.tokens_saved == 650

    def test_level_transition(self):
        """tokens_saved is zero when levels are the same (no-op)."""
        page = _text_page()
        result = CompressionResult(
            page=page,
            original_tokens=400,
            compressed_tokens=400,
            level_before=CompressionLevel.FULL,
            level_after=CompressionLevel.FULL,
        )
        assert result.tokens_saved == 0
        assert result.level_before == result.level_after


# ===========================================================================
# TestTextCompressor
# ===========================================================================


class TestTextCompressor:
    """Tests for TextCompressor across all compression transitions."""

    @pytest.mark.asyncio
    async def test_full_to_reduced(self):
        page = _text_page(content="Hello world " * 100)
        compressor = TextCompressor()
        result = await compressor.compress(page, CompressionLevel.REDUCED)

        assert result.level_after == CompressionLevel.REDUCED
        assert result.compressed_tokens < result.original_tokens
        assert result.page.compression_level == CompressionLevel.REDUCED

    @pytest.mark.asyncio
    async def test_full_to_abstract(self):
        page = _text_page(content="The quick brown fox. Jumped over the lazy dog. " * 20)
        compressor = TextCompressor()
        result = await compressor.compress(page, CompressionLevel.ABSTRACT)

        assert result.level_after == CompressionLevel.ABSTRACT
        assert result.compressed_tokens < result.original_tokens
        assert result.page.compression_level == CompressionLevel.ABSTRACT

    @pytest.mark.asyncio
    async def test_full_to_reference(self):
        page = _text_page(content="Important discussion about memory management.\nMore details follow.")
        compressor = TextCompressor()
        result = await compressor.compress(page, CompressionLevel.REFERENCE)

        assert result.level_after == CompressionLevel.REFERENCE
        assert result.page.compression_level == CompressionLevel.REFERENCE
        # Reference format: [page_type] first_line
        assert "[transcript]" in str(result.page.content)

    @pytest.mark.asyncio
    async def test_reduced_to_abstract(self):
        page = _text_page(
            content="Reduced text. " * 50,
            compression_level=CompressionLevel.REDUCED,
        )
        compressor = TextCompressor()
        result = await compressor.compress(page, CompressionLevel.ABSTRACT)

        assert result.level_before == CompressionLevel.REDUCED
        assert result.level_after == CompressionLevel.ABSTRACT

    @pytest.mark.asyncio
    async def test_reduced_to_reference(self):
        page = _text_page(
            content="Reduced content here.\nSecond line.",
            compression_level=CompressionLevel.REDUCED,
        )
        compressor = TextCompressor()
        result = await compressor.compress(page, CompressionLevel.REFERENCE)

        assert result.level_after == CompressionLevel.REFERENCE
        assert "[transcript]" in str(result.page.content)

    @pytest.mark.asyncio
    async def test_abstract_to_reference(self):
        page = _text_page(
            content="Summary of conversation about memory.",
            compression_level=CompressionLevel.ABSTRACT,
        )
        compressor = TextCompressor()
        result = await compressor.compress(page, CompressionLevel.REFERENCE)

        assert result.level_before == CompressionLevel.ABSTRACT
        assert result.level_after == CompressionLevel.REFERENCE

    @pytest.mark.asyncio
    async def test_same_level_noop(self):
        """Compressing to the same level should be a no-op."""
        page = _text_page(
            content="Already reduced.",
            compression_level=CompressionLevel.REDUCED,
        )
        compressor = TextCompressor()
        result = await compressor.compress(page, CompressionLevel.REDUCED)

        assert result.level_before == CompressionLevel.REDUCED
        assert result.level_after == CompressionLevel.REDUCED
        assert result.tokens_saved == 0
        # Page should be the exact same object (no copy)
        assert result.page is page

    @pytest.mark.asyncio
    async def test_lower_level_noop(self):
        """Compressing to a lower level should also be a no-op."""
        page = _text_page(
            content="Already abstract.",
            compression_level=CompressionLevel.ABSTRACT,
        )
        compressor = TextCompressor()
        result = await compressor.compress(page, CompressionLevel.REDUCED)

        assert result.level_after == CompressionLevel.ABSTRACT
        assert result.tokens_saved == 0

    @pytest.mark.asyncio
    async def test_with_summarize_fn(self):
        """When summarize_fn is provided, it should be called for reduction."""

        async def mock_summarize(text: str, max_tokens: int) -> str:
            return f"Summary ({max_tokens} tokens)"

        compressor = TextCompressor(summarize_fn=mock_summarize)
        page = _text_page(content="A very long piece of text. " * 100)
        result = await compressor.compress(page, CompressionLevel.REDUCED)

        assert result.level_after == CompressionLevel.REDUCED
        assert "Summary (" in str(result.page.content)
        assert "tokens)" in str(result.page.content)

    @pytest.mark.asyncio
    async def test_empty_content(self):
        """Empty/None content should not raise."""
        page = _text_page(content="", size_tokens=0)
        compressor = TextCompressor()
        result = await compressor.compress(page, CompressionLevel.REDUCED)

        # Even empty content produces a result
        assert result.level_after == CompressionLevel.REDUCED
        assert result.compressed_tokens >= 1  # min of 1


# ===========================================================================
# TestPassthroughCompressor
# ===========================================================================


class TestPassthroughCompressor:
    """Tests for PassthroughCompressor (no-op compression)."""

    @pytest.mark.asyncio
    async def test_updates_level(self):
        page = _text_page(content="Audio transcript stub", compression_level=CompressionLevel.FULL)
        compressor = PassthroughCompressor(modality=Modality.AUDIO)
        result = await compressor.compress(page, CompressionLevel.REDUCED)

        assert result.level_after == CompressionLevel.REDUCED
        assert result.page.compression_level == CompressionLevel.REDUCED

    @pytest.mark.asyncio
    async def test_content_unchanged(self):
        original_content = "This content should not change"
        page = _text_page(content=original_content)
        compressor = PassthroughCompressor(modality=Modality.TEXT)
        result = await compressor.compress(page, CompressionLevel.ABSTRACT)

        assert str(result.page.content) == original_content
        assert result.tokens_saved == 0

    def test_modality(self):
        compressor = PassthroughCompressor(modality=Modality.VIDEO)
        assert compressor.supported_modality == Modality.VIDEO

    def test_can_compress(self):
        compressor = PassthroughCompressor(modality=Modality.AUDIO)
        assert compressor.can_compress(CompressionLevel.FULL, CompressionLevel.REDUCED) is True
        assert compressor.can_compress(CompressionLevel.REDUCED, CompressionLevel.FULL) is False
        assert compressor.can_compress(CompressionLevel.ABSTRACT, CompressionLevel.ABSTRACT) is False


# ===========================================================================
# TestImageCompressor
# ===========================================================================


class TestImageCompressor:
    """Tests for ImageCompressor across compression transitions."""

    @pytest.mark.asyncio
    async def test_full_to_reduced_noop(self):
        """FULL -> REDUCED for images just updates the level; tokens stay the same."""
        page = _image_page()
        compressor = ImageCompressor()
        result = await compressor.compress(page, CompressionLevel.REDUCED)

        assert result.level_after == CompressionLevel.REDUCED
        assert result.compressed_tokens == result.original_tokens
        assert result.page.compression_level == CompressionLevel.REDUCED

    @pytest.mark.asyncio
    async def test_to_abstract_uses_caption(self):
        page = _image_page(caption="A cat sitting on a mat")
        compressor = ImageCompressor()
        result = await compressor.compress(page, CompressionLevel.ABSTRACT)

        assert result.level_after == CompressionLevel.ABSTRACT
        assert result.page.content == "A cat sitting on a mat"
        assert result.compressed_tokens < result.original_tokens

    @pytest.mark.asyncio
    async def test_to_abstract_no_caption_placeholder(self):
        page = _image_page(caption=None)
        compressor = ImageCompressor()
        result = await compressor.compress(page, CompressionLevel.ABSTRACT)

        assert result.level_after == CompressionLevel.ABSTRACT
        assert result.page.content == "[IMAGE]"

    @pytest.mark.asyncio
    async def test_to_reference(self):
        page = _image_page(page_id="img42")
        compressor = ImageCompressor()
        result = await compressor.compress(page, CompressionLevel.REFERENCE)

        assert result.level_after == CompressionLevel.REFERENCE
        assert "img42" in str(result.page.content)
        assert "image:" in str(result.page.content)

    def test_modality(self):
        compressor = ImageCompressor()
        assert compressor.supported_modality == Modality.IMAGE


# ===========================================================================
# TestCompressorRegistry
# ===========================================================================


class TestCompressorRegistry:
    """Tests for CompressorRegistry routing and defaults."""

    def test_default_has_all_modalities(self):
        registry = CompressorRegistry.default()
        # Should have entries for TEXT, IMAGE, AUDIO, VIDEO, STRUCTURED
        for modality in Modality:
            compressor = registry.get(modality)
            assert compressor is not None

    def test_register_custom(self):
        registry = CompressorRegistry()
        custom = PassthroughCompressor(modality=Modality.TEXT)
        registry.register(Modality.TEXT, custom)
        assert registry.get(Modality.TEXT) is custom

    def test_get_returns_registered(self):
        registry = CompressorRegistry.default()
        text_compressor = registry.get(Modality.TEXT)
        assert isinstance(text_compressor, TextCompressor)

    def test_get_returns_passthrough_for_unknown(self):
        """Registry with no entries should return PassthroughCompressor for any modality."""
        registry = CompressorRegistry()
        compressor = registry.get(Modality.AUDIO)
        assert isinstance(compressor, PassthroughCompressor)

    @pytest.mark.asyncio
    async def test_compress_routes_correctly(self):
        registry = CompressorRegistry.default()
        text_page = _text_page(content="Route me to TextCompressor. " * 50)
        result = await registry.compress_page(text_page, CompressionLevel.REDUCED)

        assert result.level_after == CompressionLevel.REDUCED
        assert result.compressed_tokens < result.original_tokens

    @pytest.mark.asyncio
    async def test_compress_text(self):
        registry = CompressorRegistry.default()
        page = _text_page(content="Sentence one. Sentence two. " * 40)
        result = await registry.compress_page(page, CompressionLevel.ABSTRACT)

        assert result.level_after == CompressionLevel.ABSTRACT
        assert result.page.compression_level == CompressionLevel.ABSTRACT

    @pytest.mark.asyncio
    async def test_compress_image(self):
        registry = CompressorRegistry.default()
        page = _image_page(caption="Sunset over the ocean")
        result = await registry.compress_page(page, CompressionLevel.ABSTRACT)

        assert result.level_after == CompressionLevel.ABSTRACT
        assert result.page.content == "Sunset over the ocean"

    @pytest.mark.asyncio
    async def test_compress_unknown_modality(self):
        """An empty registry should still compress via PassthroughCompressor."""
        registry = CompressorRegistry()
        page = _text_page(content="Some structured data", modality=Modality.STRUCTURED)
        result = await registry.compress_page(page, CompressionLevel.REDUCED)

        assert result.level_after == CompressionLevel.REDUCED
        # Passthrough: content unchanged, tokens unchanged
        assert result.tokens_saved == 0


# ===========================================================================
# TestManagerCompression
# ===========================================================================


class TestManagerCompression:
    """Tests for MemoryManager.compress_page() integration."""

    @pytest.mark.asyncio
    async def test_compress_updates_store(self):
        mgr = MemoryManager(
            session_id="s1",
            compressor_registry=CompressorRegistry.default(),
        )
        page = mgr.create_page("Hello world " * 100, page_type=PageType.TRANSCRIPT)
        await mgr.add_to_working_set(page)

        result = await mgr.compress_page(page.page_id, CompressionLevel.REDUCED)

        assert result is not None
        assert result.level_after == CompressionLevel.REDUCED
        # The store should now hold the compressed page
        stored = mgr._page_store[page.page_id]
        assert stored.compression_level == CompressionLevel.REDUCED

    @pytest.mark.asyncio
    async def test_compress_updates_page_table(self):
        mgr = MemoryManager(
            session_id="s1",
            compressor_registry=CompressorRegistry.default(),
        )
        page = mgr.create_page("Content for table test. " * 80, page_type=PageType.TRANSCRIPT)
        await mgr.add_to_working_set(page)

        await mgr.compress_page(page.page_id, CompressionLevel.ABSTRACT)

        entry = mgr._page_table.entries.get(page.page_id)
        assert entry is not None
        assert entry.compression_level == CompressionLevel.ABSTRACT

    @pytest.mark.asyncio
    async def test_compress_logs_mutation(self):
        mgr = MemoryManager(
            session_id="s1",
            compressor_registry=CompressorRegistry.default(),
        )
        page = mgr.create_page("Log mutation test. " * 80, page_type=PageType.TRANSCRIPT)
        await mgr.add_to_working_set(page)

        await mgr.compress_page(page.page_id, CompressionLevel.REDUCED)

        mutations = mgr._mutation_log.get_all_mutations()
        compress_mutations = [m for m in mutations if m.mutation_type == MutationType.COMPRESS]
        assert len(compress_mutations) >= 1
        assert compress_mutations[-1].page_id == page.page_id
        assert "compress_reduced" in (compress_mutations[-1].cause or "")

    @pytest.mark.asyncio
    async def test_no_registry_returns_none(self):
        """Without a compressor registry, compress_page returns None."""
        mgr = MemoryManager(session_id="s1")
        page = mgr.create_page("No registry test.", page_type=PageType.TRANSCRIPT)
        await mgr.add_to_working_set(page)

        result = await mgr.compress_page(page.page_id, CompressionLevel.REDUCED)
        assert result is None

    @pytest.mark.asyncio
    async def test_unknown_page_returns_none(self):
        mgr = MemoryManager(
            session_id="s1",
            compressor_registry=CompressorRegistry.default(),
        )
        result = await mgr.compress_page("nonexistent_page", CompressionLevel.REDUCED)
        assert result is None

    @pytest.mark.asyncio
    async def test_records_metrics(self):
        mgr = MemoryManager(
            session_id="s1",
            compressor_registry=CompressorRegistry.default(),
        )
        page = mgr.create_page("Metrics tracking test. " * 80, page_type=PageType.TRANSCRIPT)
        await mgr.add_to_working_set(page)

        assert mgr.metrics.compressions_total == 0

        await mgr.compress_page(page.page_id, CompressionLevel.REDUCED)

        assert mgr.metrics.compressions_total >= 1
        assert mgr.metrics.tokens_saved_by_compression > 0


# ===========================================================================
# TestManagerEvictionWithCompression
# ===========================================================================


class TestManagerEvictionWithCompression:
    """Tests for compress-before-evict logic in MemoryManager._run_eviction()."""

    @pytest.mark.asyncio
    async def test_compresses_before_evicting(self):
        """When a compressor registry is present, eviction should try compression first."""
        config = WorkingSetConfig(max_l0_tokens=500, reserved_tokens=0)
        mgr = MemoryManager(
            session_id="s1",
            config=config,
            compressor_registry=CompressorRegistry.default(),
        )
        # Constrain the budget so eviction triggers with small pages
        mgr._working_set.budget = TokenBudget(total_limit=500, reserved=0)

        # Add several pages that together exceed the 500-token budget
        pages = []
        for i in range(5):
            p = mgr.create_page(
                f"Page {i} content. " * 40,
                page_type=PageType.TRANSCRIPT,
                page_id=f"page_{i}",
            )
            await mgr.add_to_working_set(p)
            pages.append(p)

        # At this point eviction (with compression) should have fired.
        # Verify that at least one compression or eviction happened.
        assert mgr.metrics.compressions_total >= 1 or mgr.metrics.evictions_total >= 1

    @pytest.mark.asyncio
    async def test_no_registry_evicts_directly(self):
        """Without a compressor registry, eviction proceeds without compression."""
        config = WorkingSetConfig(max_l0_tokens=500, reserved_tokens=0)
        mgr = MemoryManager(
            session_id="s1",
            config=config,
            # No compressor_registry
        )
        # Constrain the budget so eviction triggers with small pages
        mgr._working_set.budget = TokenBudget(total_limit=500, reserved=0)

        pages = []
        for i in range(5):
            p = mgr.create_page(
                f"Page {i} content. " * 40,
                page_type=PageType.TRANSCRIPT,
                page_id=f"page_{i}",
            )
            await mgr.add_to_working_set(p)
            pages.append(p)

        # No compressions should have happened
        assert mgr.metrics.compressions_total == 0
        # But eviction should have happened to make room
        assert mgr.metrics.evictions_total >= 1


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


class TestTextCompressorProperties:
    """Cover TextCompressor.supported_modality and can_compress (lines 142, 149)."""

    def test_supported_modality(self):
        c = TextCompressor()
        assert c.supported_modality == Modality.TEXT

    def test_can_compress_higher(self):
        c = TextCompressor()
        assert c.can_compress(CompressionLevel.FULL, CompressionLevel.REDUCED) is True

    def test_can_compress_same(self):
        c = TextCompressor()
        assert c.can_compress(CompressionLevel.REDUCED, CompressionLevel.REDUCED) is False

    def test_can_compress_lower(self):
        c = TextCompressor()
        assert c.can_compress(CompressionLevel.ABSTRACT, CompressionLevel.REDUCED) is False

    @pytest.mark.asyncio
    async def test_abstract_with_summarize_fn(self):
        """Cover summarize_fn path for _to_abstract (line 209)."""

        async def mock_summarize(text: str, max_tokens: int) -> str:
            return f"Abstract summary ({max_tokens})"

        c = TextCompressor(summarize_fn=mock_summarize)
        page = MemoryPage(
            page_id="p1",
            session_id="s1",
            content="Long text. " * 200,
            page_type=PageType.TRANSCRIPT,
            modality=Modality.TEXT,
        )
        result = await c.compress(page, CompressionLevel.ABSTRACT)
        assert "Abstract summary" in str(result.page.content)
        assert result.level_after == CompressionLevel.ABSTRACT


class TestImageCompressorProperties:
    """Cover ImageCompressor.can_compress and no-op branches (lines 249, 260)."""

    def test_can_compress_higher(self):
        c = ImageCompressor()
        assert c.can_compress(CompressionLevel.FULL, CompressionLevel.REDUCED) is True

    def test_can_compress_same(self):
        c = ImageCompressor()
        assert c.can_compress(CompressionLevel.REDUCED, CompressionLevel.REDUCED) is False

    @pytest.mark.asyncio
    async def test_noop_when_at_target(self):
        """Cover the target_level <= level_before branch (line 260)."""
        c = ImageCompressor()
        page = MemoryPage(
            page_id="img1",
            session_id="s1",
            content="image data",
            page_type=PageType.ARTIFACT,
            modality=Modality.IMAGE,
            compression_level=CompressionLevel.ABSTRACT,
        )
        result = await c.compress(page, CompressionLevel.REDUCED)
        assert result.level_after == CompressionLevel.ABSTRACT
        assert result.tokens_saved == 0


class TestPassthroughCompressorNoop:
    """Cover PassthroughCompressor no-op branch (line 350)."""

    @pytest.mark.asyncio
    async def test_noop_when_at_target(self):
        c = PassthroughCompressor(Modality.AUDIO)
        page = MemoryPage(
            page_id="a1",
            session_id="s1",
            content="audio ref",
            page_type=PageType.ARTIFACT,
            modality=Modality.AUDIO,
            compression_level=CompressionLevel.REDUCED,
        )
        result = await c.compress(page, CompressionLevel.FULL)
        assert result.level_after == CompressionLevel.REDUCED
        assert result.tokens_saved == 0
