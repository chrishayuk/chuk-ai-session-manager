# chuk_ai_session_manager/memory/mutation_log.py
"""
Lite Mutation Log for AI Virtual Memory.

Append-only log of page operations for:
- Debugging: "What was in context for turn T?"
- Replay: Reconstruct state for testing
- Grounding story: Prove what the model saw

This is a lite version focused on basic tracking.
Full event-sourcing with time-travel comes in v0.15.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime

from pydantic import BaseModel, Field, PrivateAttr

from .models import (
    Actor,
    MutationType,
    PageMutation,
    StorageTier,
)


class ContextSnapshot(BaseModel):
    """Snapshot of what pages were in context at a given turn."""

    turn: int
    page_ids: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MutationLogLite(BaseModel):
    """
    Append-only log of page operations.

    Not full event-sourcing, but enough for:
    - Debugging: "what was in context for turn T?"
    - Replay: Reconstruct state for testing
    - Grounding story: Prove what the model saw
    """

    session_id: str = Field(default="")

    # Append-only mutation list
    _mutations: list[PageMutation] = PrivateAttr(default_factory=list)

    # Index: page_id -> list of mutations
    _by_page: dict[str, list[PageMutation]] = PrivateAttr(default_factory=lambda: defaultdict(list))

    # Context snapshots per turn
    _context_snapshots: dict[int, ContextSnapshot] = PrivateAttr(default_factory=dict)

    def append(self, mutation: PageMutation) -> None:
        """Append a mutation to the log."""
        self._mutations.append(mutation)
        self._by_page[mutation.page_id].append(mutation)

    def record_mutation(
        self,
        page_id: str,
        mutation_type: MutationType,
        tier_after: StorageTier,
        tier_before: StorageTier | None = None,
        actor: Actor = Actor.SYSTEM,
        cause: str | None = None,
        turn: int = 0,
    ) -> PageMutation:
        """Create and append a mutation record."""
        mutation = PageMutation(
            mutation_id=str(uuid.uuid4())[:8],
            page_id=page_id,
            turn=turn,
            mutation_type=mutation_type,
            tier_before=tier_before,
            tier_after=tier_after,
            actor=actor,
            cause=cause,
        )
        self.append(mutation)
        return mutation

    def record_context_at_turn(self, turn: int, page_ids: list[str]) -> None:
        """
        Snapshot the context (L0 pages) at a turn.

        Call this at the start or end of each turn to enable
        "what was in context for turn T?" queries.
        """
        self._context_snapshots[turn] = ContextSnapshot(
            turn=turn,
            page_ids=list(page_ids),
        )

    def get_context_at_turn(self, turn: int) -> list[str]:
        """
        Replay: what page_ids were in L0 at turn T?

        Returns empty list if no snapshot for that turn.
        """
        snapshot = self._context_snapshots.get(turn)
        if snapshot:
            return snapshot.page_ids
        return []

    def get_history(self, page_id: str) -> list[PageMutation]:
        """All mutations for a page."""
        return list(self._by_page.get(page_id, []))

    def get_mutations_by_actor(self, actor: Actor) -> list[PageMutation]:
        """All mutations by a specific actor."""
        return [m for m in self._mutations if m.actor == actor]

    def get_mutations_by_type(self, mutation_type: MutationType) -> list[PageMutation]:
        """All mutations of a specific type."""
        return [m for m in self._mutations if m.mutation_type == mutation_type]

    def get_mutations_in_turn(self, turn: int) -> list[PageMutation]:
        """All mutations in a specific turn."""
        return [m for m in self._mutations if m.turn == turn]

    def get_pages_created_in_turn(self, turn: int) -> list[str]:
        """Get page IDs created in a specific turn."""
        return [m.page_id for m in self._mutations if m.turn == turn and m.mutation_type == MutationType.CREATE]

    def get_pages_faulted_in_turn(self, turn: int) -> list[str]:
        """Get page IDs faulted in during a specific turn."""
        return [m.page_id for m in self._mutations if m.turn == turn and m.mutation_type == MutationType.FAULT_IN]

    def get_pages_evicted_in_turn(self, turn: int) -> list[str]:
        """Get page IDs evicted in a specific turn."""
        return [m.page_id for m in self._mutations if m.turn == turn and m.mutation_type == MutationType.EVICT]

    def get_all_mutations(self) -> list[PageMutation]:
        """Get all mutations in chronological order."""
        return list(self._mutations)

    def mutation_count(self) -> int:
        """Total number of mutations logged."""
        return len(self._mutations)

    def page_count(self) -> int:
        """Number of unique pages with mutations."""
        return len(self._by_page)

    def clear(self) -> None:
        """Clear all mutations (for testing)."""
        self._mutations.clear()
        self._by_page.clear()
        self._context_snapshots.clear()

    def get_summary(self) -> MutationLogSummary:
        """Get summary statistics."""
        by_type: dict[MutationType, int] = defaultdict(int)
        for m in self._mutations:
            by_type[m.mutation_type] += 1

        return MutationLogSummary(
            total_mutations=len(self._mutations),
            unique_pages=len(self._by_page),
            context_snapshots=len(self._context_snapshots),
            creates=by_type.get(MutationType.CREATE, 0),
            faults=by_type.get(MutationType.FAULT_IN, 0),
            evictions=by_type.get(MutationType.EVICT, 0),
            compressions=by_type.get(MutationType.COMPRESS, 0),
            pins=by_type.get(MutationType.PIN, 0),
            unpins=by_type.get(MutationType.UNPIN, 0),
        )
