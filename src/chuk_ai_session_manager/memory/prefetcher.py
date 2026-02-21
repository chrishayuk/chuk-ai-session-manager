# chuk_ai_session_manager/memory/prefetcher.py
"""
Simple Prefetcher for AI Virtual Memory.

Don't wait for ML prediction. Start with dumb heuristics that work:
1. Last segment summary (almost always needed for "what did we discuss")
2. Most-referenced claim pages (high access_count claims)
3. Tool traces for likely tools (based on recent tool usage)

Keep it simple: no ML prediction required.
"""

from collections import Counter
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, PrivateAttr

from .models import PageType

if TYPE_CHECKING:
    from .page_table import PageTable


class ToolUsagePattern(BaseModel):
    """Track tool usage patterns for prefetch prediction."""

    tool_name: str
    call_count: int = Field(default=0)
    last_turn: int = Field(default=0)
    prereq_pages: list[str] = Field(default_factory=list)  # Pages often read before calling


class SimplePrefetcher(BaseModel):
    """
    Basic prefetch that doesn't need prediction models.
    Good enough for v0.9.
    """

    # How many claim pages to prefetch
    max_claims_to_prefetch: int = Field(default=3)

    # How many recent tools to consider
    max_recent_tools: int = Field(default=3)

    # Tool usage tracking
    _tool_usage: dict[str, ToolUsagePattern] = PrivateAttr(default_factory=dict)

    # Recent page accesses per tool (tool_name -> list of page_ids accessed before call)
    _tool_prereqs: dict[str, list[str]] = PrivateAttr(default_factory=dict)

    # Page access counts (for finding high-value claims)
    _page_access_counts: Counter = PrivateAttr(default_factory=Counter)

    # Last segment summary page ID (updated when segments roll)
    _last_segment_summary_id: str | None = PrivateAttr(default=None)

    def record_page_access(self, page_id: str) -> None:
        """Record a page access for statistics."""
        self._page_access_counts[page_id] += 1

    def record_tool_call(
        self,
        tool_name: str,
        turn: int,
        pages_accessed_before: list[str] | None = None,
    ) -> None:
        """
        Record a tool call and what pages were accessed before it.

        This builds up patterns like:
        "When calling weather_tool, we usually read location claims first"
        """
        if tool_name not in self._tool_usage:
            self._tool_usage[tool_name] = ToolUsagePattern(tool_name=tool_name)

        pattern = self._tool_usage[tool_name]
        pattern.call_count += 1
        pattern.last_turn = turn

        # Track pages accessed before this tool call
        if pages_accessed_before:
            if tool_name not in self._tool_prereqs:
                self._tool_prereqs[tool_name] = []
            self._tool_prereqs[tool_name].extend(pages_accessed_before)

    def set_last_segment_summary(self, page_id: str) -> None:
        """Update the last segment summary page ID."""
        self._last_segment_summary_id = page_id

    def get_likely_tools(self, recent_turns: int = 5) -> list[str]:  # noqa: ARG002
        """Get tools likely to be called based on recent usage."""
        # Sort by recency and frequency
        tools = list(self._tool_usage.values())
        tools.sort(key=lambda t: (t.last_turn, t.call_count), reverse=True)
        return [t.tool_name for t in tools[: self.max_recent_tools]]

    def get_tool_prereq_pages(self, tool_name: str) -> list[str]:
        """
        Get pages commonly accessed before calling a tool.

        Returns most frequently accessed prereq pages.
        """
        prereqs = self._tool_prereqs.get(tool_name, [])
        if not prereqs:
            return []

        # Count occurrences and return top ones
        counts = Counter(prereqs)
        return [page_id for page_id, _ in counts.most_common(3)]

    def get_top_claims(
        self,
        page_table: "PageTable | None" = None,
        limit: int | None = None,
    ) -> list[str]:
        """
        Get most-referenced claim pages.

        If page_table is provided, filters to only claim-type pages.
        """
        if limit is None:
            limit = self.max_claims_to_prefetch

        # Get top accessed pages
        top_pages = [page_id for page_id, _ in self._page_access_counts.most_common(limit * 3)]

        if page_table is None:
            return top_pages[:limit]

        # Filter to claims only
        claims = []
        for page_id in top_pages:
            entry = page_table.lookup(page_id)
            if entry and entry.page_type == PageType.CLAIM:
                claims.append(page_id)
                if len(claims) >= limit:
                    break

        return claims

    async def prefetch_on_turn_start(
        self,
        session_id: str,  # noqa: ARG002 — reserved for storage-backed prefetch
        page_table: "PageTable | None" = None,
    ) -> list[str]:
        """
        Get pages to prefetch at the start of a turn.

        Returns list of page_ids to prefetch.
        """
        pages_to_prefetch: list[str] = []
        seen: set[str] = set()

        def add_page(page_id: str) -> None:
            if page_id and page_id not in seen:
                pages_to_prefetch.append(page_id)
                seen.add(page_id)

        # 1. Last segment summary (almost always needed for "what did we discuss")
        if self._last_segment_summary_id:
            add_page(self._last_segment_summary_id)

        # 2. Most-referenced claim pages (high access_count claims)
        for claim_id in self.get_top_claims(page_table):
            add_page(claim_id)

        # 3. Tool prereqs for likely tools
        likely_tools = self.get_likely_tools()
        for tool_name in likely_tools:
            for prereq_id in self.get_tool_prereq_pages(tool_name):
                add_page(prereq_id)

        return pages_to_prefetch

    def clear(self) -> None:
        """Clear all tracking data."""
        self._tool_usage.clear()
        self._tool_prereqs.clear()
        self._page_access_counts.clear()
        self._last_segment_summary_id = None

    def get_stats(self) -> dict[str, int]:
        """Get prefetcher statistics."""
        return {
            "tools_tracked": len(self._tool_usage),
            "pages_tracked": len(self._page_access_counts),
            "total_accesses": sum(self._page_access_counts.values()),
        }
