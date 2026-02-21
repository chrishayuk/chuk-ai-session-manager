# chuk_ai_session_manager/memory/models/fault.py
"""Fault policy, fault records, and mutation log models."""

from datetime import datetime

from pydantic import BaseModel, Field

from chuk_ai_session_manager.memory.models.enums import (
    Actor,
    FaultConfidenceThreshold,
    FaultReason,
    MutationType,
    StorageTier,
)

# =============================================================================
# Fault Policy Models
# =============================================================================


class FaultPolicy(BaseModel):
    """
    Guardrails to prevent fault spirals and budget blowouts.
    """

    # Existing
    max_faults_per_turn: int = Field(default=3)

    # Token budget for fault resolution
    max_fault_tokens_per_turn: int = Field(default=8192, description="Don't let faults blow the token budget")

    # Confidence threshold - only fault if explicitly needed
    fault_confidence_threshold: FaultConfidenceThreshold = Field(default=FaultConfidenceThreshold.REFERENCED)

    # Track tokens used this turn for fault resolution
    tokens_used_this_turn: int = Field(default=0)
    faults_this_turn: int = Field(default=0)

    def can_fault(self, estimated_tokens: int) -> bool:
        """Check if a fault is allowed under current policy."""
        if self.faults_this_turn >= self.max_faults_per_turn:
            return False
        return self.tokens_used_this_turn + estimated_tokens <= self.max_fault_tokens_per_turn

    def record_fault(self, tokens: int) -> None:
        """Record a fault and its token cost."""
        self.faults_this_turn += 1
        self.tokens_used_this_turn += tokens

    def new_turn(self) -> None:
        """Reset for new turn."""
        self.faults_this_turn = 0
        self.tokens_used_this_turn = 0


class FaultRecord(BaseModel):
    """Record of a single page fault for metrics."""

    page_id: str
    reason: FaultReason
    turn: int
    tokens_loaded: int
    latency_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Mutation Log Models
# =============================================================================


class PageMutation(BaseModel):
    """
    Immutable record of a page change.

    Enables debugging, replay, and grounding story:
    - "What was in context for turn T?"
    - "Who changed what and why?"
    """

    mutation_id: str
    page_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    turn: int = Field(default=0)

    mutation_type: MutationType

    # Context at mutation time
    tier_before: StorageTier | None = None
    tier_after: StorageTier

    # Who caused it
    actor: Actor
    cause: str | None = Field(
        default=None,
        description="e.g., 'eviction_pressure', 'page_fault', 'explicit_request'",
    )
