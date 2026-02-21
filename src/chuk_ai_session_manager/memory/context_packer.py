# chuk_ai_session_manager/memory/context_packer.py
"""
Context Packer for AI Virtual Memory.

The ContextPacker transforms the working set into the VM:CONTEXT block -
a compact, human-readable representation of mapped pages that goes into
the model's context window.

Design principles:
- Human-readable: Format is easy for models to parse and reference
- Token-efficient: Compact representation with clear structure
- Cross-modal: Handles text, images, audio, video with appropriate formats
- Pydantic-native: All models are BaseModel subclasses
- No magic strings: Uses enums for all categorical values
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .models import (
    ContextPrefix,
    FormattedPage,
    MemoryPage,
    MessageRole,
    Modality,
    PageType,
)


class PackedContext(BaseModel):
    """Result of packing the working set into context."""

    content: str = Field(..., description="The VM:CONTEXT block content")
    tokens_est: int = Field(default=0, description="Estimated token count")
    pages_included: list[str] = Field(default_factory=list, description="Page IDs included")
    pages_truncated: list[str] = Field(default_factory=list, description="Page IDs that were truncated")
    pages_omitted: list[str] = Field(default_factory=list, description="Page IDs omitted due to budget")


class ContextPackerConfig(BaseModel):
    """Configuration for context packing."""

    include_page_ids: bool = Field(default=True, description="Include page IDs in output")
    include_timestamps: bool = Field(default=False, description="Include timestamps")
    max_text_length: int = Field(default=0, description="Max chars per text page (0=unlimited)")


class ContextPacker(BaseModel):
    """
    Packs MemoryPages into the VM:CONTEXT format.

    The output format is:
    ```
    <VM:CONTEXT>
    U (msg_301): "User message text here"
    A (msg_302): "Assistant response here"
    T (tool_result_045): {"calculator": {"result": 4}}
    S (summary_seg_02): "Key points: 1) First point, 2) Second point..."
    I (img_045): [IMAGE: architecture diagram, 1200x800]
    D (audio_012): [AUDIO: 5:42 duration, transcript: "So the key insight is..."]
    V (video_007): [VIDEO: 12:30 duration, 8 scenes, topic: "system walkthrough"]
    </VM:CONTEXT>
    ```
    """

    config: ContextPackerConfig = Field(default_factory=ContextPackerConfig)

    def pack(
        self,
        pages: list[MemoryPage],
        token_budget: int | None = None,
    ) -> PackedContext:
        """
        Pack a list of pages into VM:CONTEXT format.

        Args:
            pages: List of MemoryPages to pack
            token_budget: Optional token limit (will truncate/omit if exceeded)

        Returns:
            PackedContext with the formatted content
        """
        lines: list[str] = []
        pages_included: list[str] = []
        pages_truncated: list[str] = []
        pages_omitted: list[str] = []
        tokens_used = 0

        for page in pages:
            # Format the page
            formatted = self._format_page(page)

            # Check budget
            if token_budget and tokens_used + formatted.tokens_est > token_budget:
                # Try truncating
                if page.modality == Modality.TEXT and self.config.max_text_length == 0:
                    # Can truncate text
                    remaining = token_budget - tokens_used
                    if remaining > 50:  # Minimum useful content
                        formatted = self._format_page(page, max_tokens=remaining)
                        pages_truncated.append(page.page_id)
                    else:
                        pages_omitted.append(page.page_id)
                        continue
                else:
                    pages_omitted.append(page.page_id)
                    continue

            lines.append(formatted.content)
            pages_included.append(page.page_id)
            tokens_used += formatted.tokens_est

        return PackedContext(
            content="\n".join(lines),
            tokens_est=tokens_used,
            pages_included=pages_included,
            pages_truncated=pages_truncated,
            pages_omitted=pages_omitted,
        )

    def _format_page(
        self,
        page: MemoryPage,
        max_tokens: int | None = None,
    ) -> FormattedPage:
        """
        Format a single page for VM:CONTEXT.

        Returns FormattedPage with formatted content and token estimate.
        """
        if page.modality == Modality.TEXT:
            return self._format_text(page, max_tokens)
        elif page.modality == Modality.IMAGE:
            return self._format_image(page)
        elif page.modality == Modality.AUDIO:
            return self._format_audio(page)
        elif page.modality == Modality.VIDEO:
            return self._format_video(page)
        elif page.modality == Modality.STRUCTURED:
            return self._format_structured(page, max_tokens)
        else:
            return self._format_generic(page)

    def _format_text(
        self,
        page: MemoryPage,
        max_tokens: int | None = None,
    ) -> FormattedPage:
        """Format a text page."""
        content = page.content or ""

        # Determine prefix based on metadata
        prefix = self._get_text_prefix(page)

        # Truncate if needed
        max_chars = self.config.max_text_length
        if max_tokens:
            max_chars = max_tokens * 4  # ~4 chars per token

        if max_chars > 0 and len(content) > max_chars:
            content = content[:max_chars] + "..."

        # Format
        if self.config.include_page_ids:
            line = f'{prefix.value} ({page.page_id}): "{content}"'
        else:
            line = f'{prefix.value}: "{content}"'

        tokens_est = len(line) // 4
        return FormattedPage(content=line, tokens_est=tokens_est)

    def _get_text_prefix(self, page: MemoryPage) -> ContextPrefix:
        """Determine the prefix for a text page based on metadata."""
        role = page.metadata.get(MessageRole.USER.value, "")
        if not role:
            role = page.metadata.get("role", "")

        page_type = page.metadata.get("type", "")

        if role == MessageRole.USER.value:
            return ContextPrefix.USER
        elif role == MessageRole.ASSISTANT.value:
            return ContextPrefix.ASSISTANT
        elif role == MessageRole.TOOL.value or page_type == PageType.TRANSCRIPT.value:
            return ContextPrefix.TOOL
        elif page_type == PageType.SUMMARY.value or PageType.SUMMARY.value in page.page_id:
            return ContextPrefix.SUMMARY
        else:
            return ContextPrefix.USER  # Default

    def _format_image(self, page: MemoryPage) -> FormattedPage:
        """Format an image page."""
        parts = ["[IMAGE:"]

        # Add caption if available
        if page.caption:
            parts.append(f" {page.caption}")
        elif page.content and isinstance(page.content, str):
            parts.append(f" {page.content}")

        # Add dimensions
        if page.dimensions:
            parts.append(f", {page.dimensions[0]}x{page.dimensions[1]}")

        parts.append("]")
        description = "".join(parts)

        if self.config.include_page_ids:
            line = f"{ContextPrefix.IMAGE.value} ({page.page_id}): {description}"
        else:
            line = f"{ContextPrefix.IMAGE.value}: {description}"

        tokens_est = len(line) // 4
        return FormattedPage(content=line, tokens_est=tokens_est)

    def _format_audio(self, page: MemoryPage) -> FormattedPage:
        """Format an audio page."""
        parts = ["[AUDIO:"]

        # Duration
        if page.duration_seconds:
            mins = int(page.duration_seconds // 60)
            secs = int(page.duration_seconds % 60)
            parts.append(f" {mins}:{secs:02d} duration")

        # Transcript excerpt
        if page.transcript:
            excerpt = page.transcript[:200]
            if len(page.transcript) > 200:
                excerpt += "..."
            parts.append(f', transcript: "{excerpt}"')
        elif page.content and isinstance(page.content, str):
            excerpt = page.content[:200]
            if len(page.content) > 200:
                excerpt += "..."
            parts.append(f', transcript: "{excerpt}"')

        parts.append("]")
        description = "".join(parts)

        if self.config.include_page_ids:
            line = f"{ContextPrefix.AUDIO.value} ({page.page_id}): {description}"
        else:
            line = f"{ContextPrefix.AUDIO.value}: {description}"

        tokens_est = len(line) // 4
        return FormattedPage(content=line, tokens_est=tokens_est)

    def _format_video(self, page: MemoryPage) -> FormattedPage:
        """Format a video page."""
        parts = ["[VIDEO:"]

        # Duration
        if page.duration_seconds:
            mins = int(page.duration_seconds // 60)
            secs = int(page.duration_seconds % 60)
            parts.append(f" {mins}:{secs:02d} duration")

        # Scene count from metadata
        scene_count = page.metadata.get("scene_count")
        if scene_count:
            parts.append(f", {scene_count} scenes")

        # Topic/description
        topic = page.metadata.get("topic") or page.caption
        if topic:
            parts.append(f', topic: "{topic}"')

        parts.append("]")
        description = "".join(parts)

        if self.config.include_page_ids:
            line = f"{ContextPrefix.VIDEO.value} ({page.page_id}): {description}"
        else:
            line = f"{ContextPrefix.VIDEO.value}: {description}"

        tokens_est = len(line) // 4
        return FormattedPage(content=line, tokens_est=tokens_est)

    def _format_structured(
        self,
        page: MemoryPage,
        max_tokens: int | None = None,
    ) -> FormattedPage:
        """Format a structured (JSON) page."""
        import json

        content = page.content
        if isinstance(content, dict):
            content = json.dumps(content, separators=(",", ":"))
        elif content is None:
            content = "{}"

        # Truncate if needed
        max_chars = max_tokens * 4 if max_tokens else 0
        if max_chars > 0 and len(str(content)) > max_chars:
            content = str(content)[:max_chars] + "..."

        if self.config.include_page_ids:
            line = f"{ContextPrefix.STRUCTURED.value} ({page.page_id}): {content}"
        else:
            line = f"{ContextPrefix.STRUCTURED.value}: {content}"

        tokens_est = len(line) // 4
        return FormattedPage(content=line, tokens_est=tokens_est)

    def _format_generic(self, page: MemoryPage) -> FormattedPage:
        """Format any other page type."""
        content = str(page.content)[:500] if page.content else "[no content]"

        if self.config.include_page_ids:
            line = f"{ContextPrefix.UNKNOWN.value} ({page.page_id}): {content}"
        else:
            line = f"{ContextPrefix.UNKNOWN.value}: {content}"

        tokens_est = len(line) // 4
        return FormattedPage(content=line, tokens_est=tokens_est)

    def pack_with_wrapper(
        self,
        pages: list[MemoryPage],
        token_budget: int | None = None,
    ) -> PackedContext:
        """
        Pack pages and wrap with VM:CONTEXT tags.

        This is the complete format for inclusion in a developer message.
        """
        result = self.pack(pages, token_budget)
        result.content = f"<VM:CONTEXT>\n{result.content}\n</VM:CONTEXT>"
        # Add wrapper tokens to estimate
        result.tokens_est += 10
        return result
