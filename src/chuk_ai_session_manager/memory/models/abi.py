# chuk_ai_session_manager/memory/models/abi.py
"""Memory ABI, recall tracking, and UX metrics models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from chuk_ai_session_manager.memory.models.enums import (
    CompressionLevel,
    FaultReason,
    Modality,
    PageType,
)
from chuk_ai_session_manager.memory.models.fault import FaultRecord

# =============================================================================
# Memory ABI Models
# =============================================================================


class PageManifestEntry(BaseModel):
    """Entry in the memory manifest for a page."""

    page_id: str
    modality: Modality
    page_type: PageType
    compression_level: CompressionLevel
    tokens: int
    importance: float
    provenance: list[str] = Field(default_factory=list)  # source page_ids
    can_evict: bool = Field(default=True)
    can_compress: bool = Field(default=True)


class MemoryABI(BaseModel):
    """
    Application Binary Interface for memory negotiation.

    Lets different models negotiate memory requirements.
    Smaller models survive with aggressive compression.
    Tool processors reason about memory cost.
    """

    # What's in context
    pages: list[PageManifestEntry] = Field(default_factory=list)

    # Capabilities
    faults_allowed: bool = Field(default=True)
    upgrade_budget_tokens: int = Field(default=2048, description="Tokens reserved for fault resolution")

    # Constraints
    max_context_tokens: int = Field(default=128000)
    reserved_tokens: int = Field(default=2000, description="System prompt, etc.")

    # Tool schema budget (often the hidden token hog)
    tool_schema_tokens_reserved: int = Field(default=0, description="Tokens consumed by tool definitions")
    active_toolset_hash: str | None = Field(default=None, description="For cache invalidation when tools change")

    # Preferences
    modality_weights: dict[Modality, float] = Field(
        default_factory=lambda: {
            Modality.TEXT: 1.0,
            Modality.IMAGE: 0.8,
            Modality.AUDIO: 0.6,
            Modality.VIDEO: 0.4,
        }
    )

    @property
    def available_tokens(self) -> int:
        """Tokens available for content after reservations."""
        return max(
            0,
            self.max_context_tokens - self.reserved_tokens - self.tool_schema_tokens_reserved,
        )


# =============================================================================
# UX Metrics Models
# =============================================================================


class RecallAttempt(BaseModel):
    """Record of a recall attempt for tracking success rate."""

    turn: int
    query: str  # What user asked to recall
    page_ids_cited: list[str] = Field(default_factory=list)
    user_corrected: bool = Field(default=False)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class UserExperienceMetrics(BaseModel):
    """
    Metrics that correlate with user satisfaction.

    These tell you whether the system "feels good" to users.
    """

    # Recall tracking
    recall_attempts: list[RecallAttempt] = Field(default_factory=list)

    # Fault history for thrash calculation
    fault_history: list[FaultRecord] = Field(default_factory=list)

    # Page references per turn (for effective tokens)
    pages_referenced_per_turn: dict[int, list[str]] = Field(default_factory=dict)
    tokens_in_context_per_turn: dict[int, int] = Field(default_factory=dict)

    def recall_success_rate(self) -> float:
        """
        Success rate for recall attempts.
        Success = no correction needed.
        """
        if not self.recall_attempts:
            return 1.0
        successes = sum(1 for r in self.recall_attempts if not r.user_corrected)
        return successes / len(self.recall_attempts)

    def thrash_index(self, window_turns: int = 5) -> float:
        """
        Faults after first fault in a topic window.
        Low = stable working set. High = constantly missing what we need.
        """
        if not self.fault_history:
            return 0.0

        # Get recent faults
        if self.fault_history:
            max_turn = max(f.turn for f in self.fault_history)
            min_turn = max(0, max_turn - window_turns)
            recent_faults = [f for f in self.fault_history if f.turn >= min_turn]
        else:
            return 0.0

        if not recent_faults:
            return 0.0

        # Count first faults vs thrash faults
        seen_pages: set = set()
        thrash_faults = 0
        for fault in recent_faults:
            if fault.page_id in seen_pages:
                thrash_faults += 1
            else:
                seen_pages.add(fault.page_id)

        return thrash_faults / window_turns if window_turns > 0 else 0.0

    def effective_tokens_ratio(self, turn: int) -> float:
        """
        What fraction of context tokens actually contributed to the answer?
        """
        context_tokens = self.tokens_in_context_per_turn.get(turn, 0)
        if context_tokens == 0:
            return 0.0

        referenced_pages = self.pages_referenced_per_turn.get(turn, [])
        # This is a simplified calculation - in reality you'd sum tokens of referenced pages
        # For now, estimate based on count
        referenced_estimate = len(referenced_pages) * 200  # ~200 tokens per referenced page
        return min(1.0, referenced_estimate / context_tokens)

    def record_recall_attempt(
        self,
        turn: int,
        query: str,
        page_ids_cited: list[str],
        user_corrected: bool = False,
    ) -> None:
        """Record a recall attempt."""
        self.recall_attempts.append(
            RecallAttempt(
                turn=turn,
                query=query,
                page_ids_cited=page_ids_cited,
                user_corrected=user_corrected,
            )
        )

    def record_fault(
        self,
        page_id: str,
        reason: FaultReason,
        turn: int,
        tokens_loaded: int,
        latency_ms: float = 0.0,
    ) -> None:
        """Record a fault for thrash tracking."""
        self.fault_history.append(
            FaultRecord(
                page_id=page_id,
                reason=reason,
                turn=turn,
                tokens_loaded=tokens_loaded,
                latency_ms=latency_ms,
            )
        )

    def record_turn_context(
        self,
        turn: int,
        tokens_in_context: int,
        pages_referenced: list[str],
    ) -> None:
        """Record context state for effective tokens calculation."""
        self.tokens_in_context_per_turn[turn] = tokens_in_context
        self.pages_referenced_per_turn[turn] = pages_referenced

    def get_fault_reason_breakdown(self) -> dict[FaultReason, int]:
        """Get count of faults by reason."""
        breakdown: dict[FaultReason, int] = dict.fromkeys(FaultReason, 0)
        for fault in self.fault_history:
            breakdown[fault.reason] = breakdown.get(fault.reason, 0) + 1
        return breakdown
