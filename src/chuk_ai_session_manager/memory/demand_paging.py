# chuk_ai_session_manager/memory/demand_paging.py
"""
Demand Paging Pre-Pass for AI Virtual Memory.

Scans user messages for recall signals (e.g. "what did we discuss",
"remember when", "go back to") and returns page IDs to prefetch
into L0 before the LLM call.

This makes context recall feel seamless — the system proactively
faults relevant pages before the model even sees the message.
"""

from __future__ import annotations

import logging
import re

from .models import PageType
from .page_table import PageTable

logger = logging.getLogger(__name__)

# Default recall signal patterns (case-insensitive)
DEFAULT_RECALL_SIGNALS: list[str] = [
    r"(as we|what did we|earlier|previously|remember when)",
    r"(that (decision|choice|plan|approach) (about|for|regarding))",
    r"(go back to|revisit|return to)",
    r"(remind me|refresh my memory)",
    r"(you (said|mentioned|suggested|recommended))",
]


class DemandPagingPrePass:
    """
    Scan user messages for signals that old context will be needed.

    Pure/stateless — accepts dependencies as arguments rather than
    holding references to MemoryManager internals.

    Usage::

        pager = DemandPagingPrePass()
        candidates = pager.get_prefetch_candidates(
            message="What did we decide about the database?",
            page_table=vm.page_table,
            page_hints=vm._page_hints,
            working_set_ids=set(vm.working_set.get_l0_page_ids()),
        )
        # candidates = ["claim_db_choice_001", "summary_seg1_abc"]
    """

    def __init__(
        self,
        recall_signals: list[str] | None = None,
        max_prefetch_pages: int = 5,
    ) -> None:
        signals = recall_signals or DEFAULT_RECALL_SIGNALS
        self._recall_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in signals]
        self._max_prefetch_pages = max_prefetch_pages

    # ------------------------------------------------------------------
    # Recall signal detection
    # ------------------------------------------------------------------

    def has_recall_signal(self, message: str) -> bool:
        """Check if the message contains any recall signal patterns."""
        return any(pattern.search(message) for pattern in self._recall_patterns)

    # ------------------------------------------------------------------
    # Topic extraction
    # ------------------------------------------------------------------

    def extract_topics(self, message: str) -> list[str]:
        """
        Extract candidate topic keywords from the message.

        Strategy: keep alphanumeric tokens >= 4 chars. Short tokens
        (the, is, we, ...) are naturally excluded by the length filter,
        so no stop-word list is needed.
        """
        tokens = re.split(r"[^a-zA-Z0-9]+", message.lower())
        return [t for t in tokens if len(t) >= 4]

    # ------------------------------------------------------------------
    # Prefetch candidate generation
    # ------------------------------------------------------------------

    def get_prefetch_candidates(
        self,
        message: str,
        page_table: PageTable,
        page_hints: dict[str, str],
        working_set_ids: set[str],
    ) -> list[str]:
        """
        Analyze user message and return page IDs to prefetch into L0.

        Strategy:
        1. If recall signal detected → include claim pages not in L0
        2. Extract topics → match against page_hints (substring)
        3. Include summary pages not in L0
        4. Deduplicate, cap at max_prefetch_pages

        Args:
            message: The user's message to analyze.
            page_table: The VM page table for querying page metadata.
            page_hints: Dict mapping page_id -> hint string.
            working_set_ids: Set of page_ids already in L0.

        Returns:
            List of page_ids to fault into L0.
        """
        candidates: list[str] = []
        seen: set[str] = set()

        def _add(page_id: str) -> None:
            if page_id not in seen and page_id not in working_set_ids:
                seen.add(page_id)
                candidates.append(page_id)

        has_recall = self.has_recall_signal(message)

        # 1. Recall signal → include claim pages
        if has_recall:
            for entry in page_table.get_by_type(PageType.CLAIM):
                _add(entry.page_id)

        # 2. Topic matching against hints
        topics = self.extract_topics(message)
        if topics:
            for page_id, hint in page_hints.items():
                hint_lower = hint.lower()
                for topic in topics:
                    if topic in hint_lower:
                        _add(page_id)
                        break  # one match per page is enough

        # 3. Include summary pages (useful context even without recall)
        if has_recall:
            for entry in page_table.get_by_type(PageType.SUMMARY):
                _add(entry.page_id)

        return candidates[: self._max_prefetch_pages]
