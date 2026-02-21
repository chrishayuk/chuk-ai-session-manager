# chuk_ai_session_manager/memory/compressor.py
"""
Compressor protocol and implementations for AI Virtual Memory.

Provides per-modality compression that transforms page content through
the CompressionLevel hierarchy (FULL → REDUCED → ABSTRACT → REFERENCE)
to reduce token usage without evicting pages.

Usage::

    from chuk_ai_session_manager.memory.compressor import (
        CompressorRegistry,
        TextCompressor,
    )

    # Default registry (text + image + passthrough stubs)
    registry = CompressorRegistry.default()

    # With LLM-based summarization
    async def my_summarizer(text: str, max_tokens: int) -> str:
        return await call_llm(f"Summarize in {max_tokens} tokens: {text}")

    registry = CompressorRegistry.default()
    registry.register(Modality.TEXT, TextCompressor(summarize_fn=my_summarizer))
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from .models import CompressionLevel, MemoryPage, Modality

logger = logging.getLogger(__name__)

# =============================================================================
# Types
# =============================================================================

SummarizeFn = Callable[[str, int], Awaitable[str]]
"""Callback: (text, max_tokens) -> summarized text."""


# =============================================================================
# Models
# =============================================================================


class CompressionResult(BaseModel):
    """Result of compressing a page."""

    model_config = {"arbitrary_types_allowed": True}

    page: MemoryPage
    original_tokens: int
    compressed_tokens: int
    level_before: CompressionLevel
    level_after: CompressionLevel

    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.compressed_tokens


# =============================================================================
# Protocol
# =============================================================================


@runtime_checkable
class Compressor(Protocol):
    """
    Protocol for modality-specific page compression.

    Compressors transform page content to a lower CompressionLevel.
    Must be async to support LLM-based summarization callbacks.
    """

    @property
    def supported_modality(self) -> Modality: ...

    async def compress(
        self,
        page: MemoryPage,
        target_level: CompressionLevel,
    ) -> CompressionResult:
        """
        Compress page content to the target level.

        Should NOT modify the input page. Returns a CompressionResult
        containing a new/copied page with compressed content.
        """
        ...

    def can_compress(
        self,
        current_level: CompressionLevel,
        target_level: CompressionLevel,
    ) -> bool:
        """Check if this compressor supports the given transition."""
        ...


# =============================================================================
# Implementations
# =============================================================================


class TextCompressorConfig(BaseModel):
    """Configuration for text compression."""

    reduced_ratio: float = Field(default=0.5, description="Keep this fraction of chars for REDUCED")
    abstract_max_tokens: int = Field(default=200, description="Max tokens for ABSTRACT")
    reference_max_tokens: int = Field(default=50, description="Max tokens for REFERENCE")


class TextCompressor:
    """
    Text page compressor.

    Without summarize_fn: truncation-based (works offline, no API keys).
    With summarize_fn: LLM-based summarization for higher quality.
    """

    def __init__(
        self,
        config: TextCompressorConfig | None = None,
        summarize_fn: SummarizeFn | None = None,
    ) -> None:
        self.config = config or TextCompressorConfig()
        self._summarize_fn = summarize_fn

    @property
    def supported_modality(self) -> Modality:
        return Modality.TEXT

    def can_compress(
        self,
        current_level: CompressionLevel,
        target_level: CompressionLevel,
    ) -> bool:
        return target_level > current_level

    async def compress(
        self,
        page: MemoryPage,
        target_level: CompressionLevel,
    ) -> CompressionResult:
        content = str(page.content) if page.content else ""
        original_tokens = page.size_tokens or page.estimate_tokens()
        level_before = page.compression_level

        if target_level <= level_before:
            # No-op: already at or below target level
            return CompressionResult(
                page=page,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                level_before=level_before,
                level_after=level_before,
            )

        if target_level == CompressionLevel.REDUCED:
            compressed = await self._to_reduced(content)
        elif target_level == CompressionLevel.ABSTRACT:
            compressed = await self._to_abstract(content)
        else:  # REFERENCE
            compressed = self._to_reference(page)

        compressed_tokens = max(1, len(compressed) // 4)

        new_page = page.model_copy(
            update={
                "content": compressed,
                "compression_level": target_level,
                "size_tokens": compressed_tokens,
            }
        )

        return CompressionResult(
            page=new_page,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            level_before=level_before,
            level_after=target_level,
        )

    async def _to_reduced(self, content: str) -> str:
        """FULL → REDUCED: truncate or summarize."""
        max_chars = int(len(content) * self.config.reduced_ratio)
        if self._summarize_fn:
            max_tokens = max(1, max_chars // 4)
            return await self._summarize_fn(content, max_tokens)
        if len(content) <= max_chars:
            return content
        return content[:max_chars].rstrip() + "..."

    async def _to_abstract(self, content: str) -> str:
        """→ ABSTRACT: first N sentences or summarize."""
        max_chars = self.config.abstract_max_tokens * 4
        if self._summarize_fn:
            return await self._summarize_fn(content, self.config.abstract_max_tokens)
        # Extract first few sentences
        sentences = re.split(r"(?<=[.!?])\s+", content)
        result = ""
        for sentence in sentences:
            if len(result) + len(sentence) > max_chars:
                break
            result += sentence + " "
        return result.strip() or content[:max_chars].rstrip() + "..."

    def _to_reference(self, page: MemoryPage) -> str:
        """→ REFERENCE: type tag + brief topic."""
        content = str(page.content) if page.content else ""
        max_chars = self.config.reference_max_tokens * 4
        page_type = page.page_type.value if page.page_type else "page"
        # Take first line or first N chars as topic
        first_line = content.split("\n")[0][:max_chars]
        return f"[{page_type}] {first_line}"


class ImageCompressor:
    """
    Image page compressor.

    FULL → REDUCED: No-op (images stored by reference).
    → ABSTRACT: Uses caption field if available.
    → REFERENCE: Page type + modality tag.
    """

    @property
    def supported_modality(self) -> Modality:
        return Modality.IMAGE

    def can_compress(
        self,
        current_level: CompressionLevel,
        target_level: CompressionLevel,
    ) -> bool:
        return target_level > current_level

    async def compress(
        self,
        page: MemoryPage,
        target_level: CompressionLevel,
    ) -> CompressionResult:
        original_tokens = page.size_tokens or page.estimate_tokens()
        level_before = page.compression_level

        if target_level <= level_before:
            return CompressionResult(
                page=page,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                level_before=level_before,
                level_after=level_before,
            )

        if target_level == CompressionLevel.REDUCED:
            # Images are already stored by reference; no real compression
            new_page = page.model_copy(update={"compression_level": CompressionLevel.REDUCED})
            return CompressionResult(
                page=new_page,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                level_before=level_before,
                level_after=CompressionLevel.REDUCED,
            )

        if target_level == CompressionLevel.ABSTRACT:
            caption = page.caption or "[IMAGE]"
            compressed_tokens = max(1, len(caption) // 4)
            new_page = page.model_copy(
                update={
                    "content": caption,
                    "compression_level": CompressionLevel.ABSTRACT,
                    "size_tokens": compressed_tokens,
                }
            )
            return CompressionResult(
                page=new_page,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                level_before=level_before,
                level_after=CompressionLevel.ABSTRACT,
            )

        # REFERENCE
        page_type = page.page_type.value if page.page_type else "image"
        ref = f"[{page_type}] image:{page.page_id}"
        compressed_tokens = max(1, len(ref) // 4)
        new_page = page.model_copy(
            update={
                "content": ref,
                "compression_level": CompressionLevel.REFERENCE,
                "size_tokens": compressed_tokens,
            }
        )
        return CompressionResult(
            page=new_page,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            level_before=level_before,
            level_after=CompressionLevel.REFERENCE,
        )


class PassthroughCompressor:
    """
    No-op compressor for modalities without real compression support.

    Updates compression_level metadata but leaves content unchanged.
    Default fallback for unregistered modalities.
    """

    def __init__(self, modality: Modality = Modality.TEXT) -> None:
        self._modality = modality

    @property
    def supported_modality(self) -> Modality:
        return self._modality

    def can_compress(
        self,
        current_level: CompressionLevel,
        target_level: CompressionLevel,
    ) -> bool:
        return target_level > current_level

    async def compress(
        self,
        page: MemoryPage,
        target_level: CompressionLevel,
    ) -> CompressionResult:
        original_tokens = page.size_tokens or page.estimate_tokens()
        level_before = page.compression_level

        if target_level <= level_before:
            return CompressionResult(
                page=page,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                level_before=level_before,
                level_after=level_before,
            )

        new_page = page.model_copy(update={"compression_level": target_level})
        return CompressionResult(
            page=new_page,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            level_before=level_before,
            level_after=target_level,
        )


# =============================================================================
# Registry
# =============================================================================


class CompressorRegistry:
    """
    Registry mapping Modality → Compressor.

    Falls back to PassthroughCompressor for unregistered modalities.
    """

    def __init__(
        self,
        compressors: dict[Modality, Compressor] | None = None,
    ) -> None:
        self._compressors: dict[Modality, Compressor] = compressors or {}

    def register(self, modality: Modality, compressor: Compressor) -> None:
        """Register a compressor for a modality."""
        self._compressors[modality] = compressor

    def get(self, modality: Modality) -> Compressor:
        """Get the compressor for a modality (or passthrough fallback)."""
        return self._compressors.get(modality, PassthroughCompressor(modality))

    async def compress_page(
        self,
        page: MemoryPage,
        target_level: CompressionLevel,
    ) -> CompressionResult:
        """Route to the appropriate compressor by modality."""
        compressor = self.get(page.modality)
        return await compressor.compress(page, target_level)

    @classmethod
    def default(cls) -> CompressorRegistry:
        """Create a registry with default compressors for all modalities."""
        return cls(
            compressors={
                Modality.TEXT: TextCompressor(),
                Modality.IMAGE: ImageCompressor(),
                Modality.AUDIO: PassthroughCompressor(Modality.AUDIO),
                Modality.VIDEO: PassthroughCompressor(Modality.VIDEO),
                Modality.STRUCTURED: PassthroughCompressor(Modality.STRUCTURED),
            }
        )
