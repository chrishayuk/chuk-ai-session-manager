# chuk_ai_session_manager/memory/models/stats.py
"""Statistics, token budget, and VM metrics models."""

from pydantic import BaseModel, Field

from chuk_ai_session_manager.memory.models.enums import Modality, StorageTier

# =============================================================================
# Stats Models
# =============================================================================


class TLBStats(BaseModel):
    """Statistics for TLB performance."""

    size: int = Field(default=0, description="Current number of entries")
    max_size: int = Field(default=512, description="Maximum entries")
    utilization: float = Field(default=0.0, description="Current utilization (0-1)")
    hits: int = Field(default=0, description="Total cache hits")
    misses: int = Field(default=0, description="Total cache misses")
    hit_rate: float = Field(default=0.0, description="Hit rate (0-1)")


class WorkingSetStats(BaseModel):
    """Statistics for working set state."""

    l0_pages: int = Field(default=0, description="Pages in L0 (context)")
    l1_pages: int = Field(default=0, description="Pages in L1 (cache)")
    total_pages: int = Field(default=0, description="Total pages in working set")
    tokens_used: int = Field(default=0, description="Tokens currently used")
    tokens_available: int = Field(default=0, description="Tokens available")
    utilization: float = Field(default=0.0, description="Token utilization (0-1)")
    needs_eviction: bool = Field(default=False, description="Whether eviction is needed")
    tokens_by_modality: dict[Modality, int] = Field(default_factory=dict)


class StorageStats(BaseModel):
    """Statistics for storage backend."""

    backend: str = Field(..., description="Backend type name")
    persistent: bool = Field(default=False, description="Whether storage persists")
    session_id: str | None = Field(default=None, description="Associated session")
    pages_stored: int = Field(default=0, description="Number of pages stored")


class CombinedPageTableStats(BaseModel):
    """Combined statistics for PageTable and TLB."""

    page_table: "PageTableStats"
    tlb: TLBStats


class PageTableStats(BaseModel):
    """Statistics about the page table state."""

    total_pages: int
    dirty_pages: int
    pages_by_tier: dict[StorageTier, int]
    pages_by_modality: dict[Modality, int]

    @property
    def working_set_size(self) -> int:
        """Pages in L0 + L1."""
        return self.pages_by_tier.get(StorageTier.L0, 0) + self.pages_by_tier.get(StorageTier.L1, 0)


class FaultMetrics(BaseModel):
    """Metrics for page fault handling."""

    faults_this_turn: int = Field(default=0)
    max_faults_per_turn: int = Field(default=2)
    faults_remaining: int = Field(default=2)
    total_faults: int = Field(default=0)
    tlb_hit_rate: float = Field(default=0.0)


class TokenBudget(BaseModel):
    """
    Token allocation tracking across modalities.

    Helps manage context window usage and decide compression levels.
    """

    total_limit: int = Field(default=128000, description="Total context window size")
    reserved: int = Field(default=4000, description="Reserved for system prompt, tools, etc.")

    # Current usage by modality - stored as dict for Pydantic serialization
    tokens_by_modality: dict[Modality, int] = Field(default_factory=lambda: dict.fromkeys(Modality, 0))

    @property
    def text_tokens(self) -> int:
        return self.tokens_by_modality.get(Modality.TEXT, 0)

    @property
    def image_tokens(self) -> int:
        return self.tokens_by_modality.get(Modality.IMAGE, 0)

    @property
    def audio_tokens(self) -> int:
        return self.tokens_by_modality.get(Modality.AUDIO, 0)

    @property
    def video_tokens(self) -> int:
        return self.tokens_by_modality.get(Modality.VIDEO, 0)

    @property
    def structured_tokens(self) -> int:
        return self.tokens_by_modality.get(Modality.STRUCTURED, 0)

    @property
    def used(self) -> int:
        """Total tokens currently used."""
        return sum(self.tokens_by_modality.values())

    @property
    def available(self) -> int:
        """Tokens available for new content."""
        return max(0, self.total_limit - self.reserved - self.used)

    @property
    def utilization(self) -> float:
        """Current utilization as percentage (0-1)."""
        usable = self.total_limit - self.reserved
        if usable <= 0:
            return 1.0
        return min(1.0, self.used / usable)

    def can_fit(self, tokens: int) -> bool:
        """Check if additional tokens can fit."""
        return tokens <= self.available

    def add(self, tokens: int, modality: Modality) -> bool:
        """
        Add tokens for a modality. Returns True if successful.
        """
        if not self.can_fit(tokens):
            return False

        current = self.tokens_by_modality.get(modality, 0)
        self.tokens_by_modality[modality] = current + tokens
        return True

    def remove(self, tokens: int, modality: Modality) -> None:
        """Remove tokens for a modality."""
        current = self.tokens_by_modality.get(modality, 0)
        self.tokens_by_modality[modality] = max(0, current - tokens)

    def get_tokens(self, modality: Modality) -> int:
        """Get token count for a specific modality."""
        return self.tokens_by_modality.get(modality, 0)

    def set_tokens(self, modality: Modality, tokens: int) -> None:
        """Set token count for a specific modality."""
        self.tokens_by_modality[modality] = max(0, tokens)


class VMMetrics(BaseModel):
    """
    Metrics for monitoring VM health and performance.
    """

    # Fault tracking
    faults_total: int = Field(default=0)
    faults_this_turn: int = Field(default=0)

    # TLB stats
    tlb_hits: int = Field(default=0)
    tlb_misses: int = Field(default=0)

    # Eviction stats
    evictions_total: int = Field(default=0)
    evictions_this_turn: int = Field(default=0)

    # Token tracking
    tokens_in_working_set: int = Field(default=0)
    tokens_available: int = Field(default=0)

    # Page distribution - use Enums as keys
    pages_by_tier: dict[StorageTier, int] = Field(default_factory=lambda: dict.fromkeys(StorageTier, 0))
    pages_by_modality: dict[Modality, int] = Field(default_factory=lambda: dict.fromkeys(Modality, 0))

    @property
    def fault_rate(self) -> float:
        """Faults per turn (if we track turns)."""
        return self.faults_this_turn

    @property
    def tlb_hit_rate(self) -> float:
        """TLB hit rate as percentage."""
        total = self.tlb_hits + self.tlb_misses
        if total == 0:
            return 0.0
        return self.tlb_hits / total

    def record_fault(self) -> None:
        """Record a page fault."""
        self.faults_total += 1
        self.faults_this_turn += 1

    def record_tlb_hit(self) -> None:
        """Record a TLB hit."""
        self.tlb_hits += 1

    def record_tlb_miss(self) -> None:
        """Record a TLB miss."""
        self.tlb_misses += 1

    def record_eviction(self) -> None:
        """Record an eviction."""
        self.evictions_total += 1
        self.evictions_this_turn += 1

    # Compression stats
    compressions_total: int = Field(default=0)
    compressions_this_turn: int = Field(default=0)
    tokens_saved_by_compression: int = Field(default=0)

    def record_compression(self, tokens_saved: int = 0) -> None:
        """Record a compression event."""
        self.compressions_total += 1
        self.compressions_this_turn += 1
        self.tokens_saved_by_compression += tokens_saved

    def new_turn(self) -> None:
        """Reset per-turn counters."""
        self.faults_this_turn = 0
        self.evictions_this_turn = 0
        self.compressions_this_turn = 0
