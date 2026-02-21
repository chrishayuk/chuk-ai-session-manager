# chuk_ai_session_manager/memory/models.py
"""
Core models for the AI Virtual Memory system.

These models represent the fundamental abstractions for OS-style memory management:
- MemoryPage: Atomic unit of content (like an OS page)
- PageTableEntry: Metadata about a page's location and state
- CompressionLevel: Standard compression levels per modality
- TokenBudget: Token allocation tracking

Design principles:
- Pydantic-native: All models are BaseModel subclasses
- No magic strings: Use Enums for all categorical values
- Type-safe: Full type annotations throughout
"""

from datetime import datetime
from enum import Enum, IntEnum
from typing import Any

from pydantic import BaseModel, Field

# =============================================================================
# Enums
# =============================================================================


class CompressionLevel(IntEnum):
    """
    Standard compression levels for page content.

    Lower levels = more detail, more tokens
    Higher levels = less detail, fewer tokens
    """

    FULL = 0  # Complete content (full text, full resolution, full audio)
    REDUCED = 1  # Reduced content (excerpts, thumbnail, transcript)
    ABSTRACT = 2  # Abstract/summary (key points, caption, summary)
    REFERENCE = 3  # Reference only (topic tags, page_id only)


class Modality(str, Enum):
    """Content modality types."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    STRUCTURED = "structured"


class StorageTier(str, Enum):
    """Storage hierarchy tiers (like CPU cache levels)."""

    L0 = "L0"  # Registers - current prompt context window
    L1 = "L1"  # Cache - recent pages, hot data (session state)
    L2 = "L2"  # RAM - session storage (chuk-sessions)
    L3 = "L3"  # Disk - artifact storage (chuk-artifacts filesystem)
    L4 = "L4"  # Cold - archive storage (chuk-artifacts S3)


class Affinity(str, Enum):
    """Locality hints for distributed storage (NUMA awareness)."""

    LOCAL = "local"
    REMOTE = "remote"
    SHARED = "shared"


class VMMode(str, Enum):
    """Virtual memory operation modes."""

    STRICT = "strict"  # No hallucinated memory, citations required
    RELAXED = "relaxed"  # VM-aware but more conversational
    PASSIVE = "passive"  # No tools, runtime handles everything


class MessageRole(str, Enum):
    """Message roles in conversation context."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class PageType(str, Enum):
    """
    Types of memory pages.

    Different page types have different eviction/compression rules:
    - transcript: Raw turns, tool outputs (normal eviction)
    - summary: LLM-generated summaries (low eviction, rebuildable)
    - artifact: Tool-created content (normal eviction)
    - claim: Decisions, facts, conclusions (very low eviction, high-value)
    - procedure: "When calling tool X, we do Y" (low eviction)
    - index: Page metadata for search (very low eviction)
    """

    TRANSCRIPT = "transcript"  # Raw turns, tool outputs
    SUMMARY = "summary"  # LLM-generated summaries
    ARTIFACT = "artifact"  # Tool-created content
    CLAIM = "claim"  # Decisions, facts, conclusions (high-value)
    PROCEDURE = "procedure"  # Learned patterns for tool usage
    INDEX = "index"  # Page metadata for search


class FaultReason(str, Enum):
    """
    Reasons for page faults - enables measuring why faults happen.

    This helps distinguish good faults (user asked) from bad faults (model guessing).
    """

    USER_REQUESTED_RECALL = "user_requested_recall"  # "What did we say about X?"
    RESOLVE_REFERENCE = "resolve_reference"  # Model references page_id
    TOOL_PREREQUISITE = "tool_prereq"  # Tool needs this page
    SPECULATIVE = "speculative"  # Might be relevant


class MutationType(str, Enum):
    """Types of page mutations for the mutation log."""

    CREATE = "create"
    FAULT_IN = "fault_in"
    EVICT = "evict"
    COMPRESS = "compress"
    PIN = "pin"
    UNPIN = "unpin"
    UPDATE = "update"
    DELETE = "delete"


class Actor(str, Enum):
    """Who caused a mutation."""

    USER = "user"
    MODEL = "model"
    TOOL = "tool"
    SYSTEM = "system"


class ContextPrefix(str, Enum):
    """Prefixes for VM:CONTEXT format."""

    USER = "U"
    ASSISTANT = "A"
    TOOL = "T"
    SUMMARY = "S"
    IMAGE = "I"
    AUDIO = "D"  # 'D' for auDio (A is taken)
    VIDEO = "V"
    STRUCTURED = "J"  # JSON/structured
    UNKNOWN = "?"


class ToolType(str, Enum):
    """Tool definition types."""

    FUNCTION = "function"


# =============================================================================
# Constants
# =============================================================================

# MIME types for VM storage
MEMORY_PAGE_MIME_TYPE = "application/x-memory-page"
VM_CHECKPOINT_MIME_TYPE = "application/x-vm-checkpoint"

# All compression levels as a list (for iteration)
ALL_COMPRESSION_LEVELS: list[int] = [level.value for level in CompressionLevel]


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


# =============================================================================
# Content Models (for tool results)
# =============================================================================


class TextContent(BaseModel):
    """Text content representation."""

    text: str = Field(default="")


class ImageContent(BaseModel):
    """Image content representation."""

    caption: str | None = Field(default=None)
    url: str | None = Field(default=None)
    base64: str | None = Field(default=None)
    embedding: list[float] | None = Field(default=None)


class AudioContent(BaseModel):
    """Audio content representation."""

    transcript: str | None = Field(default=None)
    timestamps: list[dict[str, Any]] | None = Field(default=None)
    duration_seconds: float | None = Field(default=None)


class VideoContent(BaseModel):
    """Video content representation."""

    scenes: list[dict[str, Any]] = Field(default_factory=list)
    transcript: str | None = Field(default=None)
    duration_seconds: float | None = Field(default=None)


class StructuredContent(BaseModel):
    """Structured data content representation."""

    data: dict[str, Any] = Field(default_factory=dict)
    schema_name: str | None = Field(default=None)


# Union type for all content types
PageContent = TextContent | ImageContent | AudioContent | VideoContent | StructuredContent


class PageMeta(BaseModel):
    """Metadata for a page in tool results."""

    source_tier: StorageTier | None = Field(default=None)
    mime_type: str | None = Field(default=None)
    size_bytes: int | None = Field(default=None)
    dimensions: list[int] | None = Field(default=None)
    duration_seconds: float | None = Field(default=None)
    latency_ms: float | None = Field(default=None)


class PageData(BaseModel):
    """Page data in tool result envelope."""

    page_id: str
    modality: Modality
    level: CompressionLevel
    tier: StorageTier
    content: PageContent
    meta: PageMeta = Field(default_factory=PageMeta)


class FaultEffects(BaseModel):
    """Side effects of a fault operation."""

    promoted_to_working_set: bool = Field(default=False)
    tokens_est: int = Field(default=0)
    evictions: list[str] | None = Field(default=None)


class SearchResultEntry(BaseModel):
    """Single entry in search results."""

    page_id: str
    modality: Modality
    tier: StorageTier
    levels: list[CompressionLevel] = Field(default_factory=list)
    hint: str = Field(default="")
    relevance: float = Field(default=0.0)


# =============================================================================
# Tool Definition Models
# =============================================================================


class ToolParameter(BaseModel):
    """Single parameter in a tool definition."""

    type: str
    description: str
    enum: list[str] | None = Field(default=None)
    minimum: int | None = Field(default=None)
    maximum: int | None = Field(default=None)
    default: Any | None = Field(default=None)


class ToolParameters(BaseModel):
    """Parameters schema for a tool."""

    type: str = Field(default="object")
    properties: dict[str, ToolParameter] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class ToolFunction(BaseModel):
    """Function definition within a tool."""

    name: str
    description: str
    parameters: ToolParameters


class ToolDefinition(BaseModel):
    """Complete tool definition for Chat Completions API."""

    type: ToolType = Field(default=ToolType.FUNCTION)
    function: ToolFunction


# =============================================================================
# Formatted Output Models
# =============================================================================


class FormattedPage(BaseModel):
    """Result of formatting a page for context."""

    content: str = Field(..., description="Formatted content string")
    tokens_est: int = Field(default=0, description="Estimated token count")


# =============================================================================
# Core Page Models
# =============================================================================


class MemoryPage(BaseModel):
    """
    Atomic unit of content in the virtual memory system.

    A page represents any piece of content (text, image, audio, video, structured)
    with identity, versioning, and multi-resolution representations.

    This is the RIGHT abstraction boundary because it enables:
    - Cross-modal coherence
    - Versioning and dirty tracking
    - Copy-on-write
    - Checkpoint consistency
    """

    # Identity
    page_id: str = Field(..., description="Unique identifier for this page")
    session_id: str | None = Field(default=None, description="Owning session")

    # Content type
    modality: Modality = Field(..., description="Content modality")

    # Page type (critical for eviction/compression decisions)
    page_type: PageType = Field(
        default=PageType.TRANSCRIPT,
        description="Page type determines eviction/compression behavior",
    )

    # Provenance: what pages justify this one (for claims, summaries)
    provenance: list[str] = Field(
        default_factory=list,
        description="page_ids that this page derives from (for claims/summaries)",
    )

    # Representation linking (for compression chain)
    represents: str | None = Field(
        default=None,
        description="page_id this is a compressed version of",
    )
    representation_level: CompressionLevel = Field(
        default=CompressionLevel.FULL,
        description="Compression level this representation is at",
    )

    # Location
    storage_tier: StorageTier = Field(default=StorageTier.L1, description="Current storage tier")
    artifact_id: str | None = Field(default=None, description="Reference to chuk-artifacts storage")

    # Content (when loaded into L0/L1)
    content: Any | None = Field(default=None, description="Actual content when in working set")
    compression_level: CompressionLevel = Field(default=CompressionLevel.FULL, description="Current compression level")

    # Multi-resolution representations
    # Maps compression level -> artifact_id for stored representations
    representations: dict[CompressionLevel, str] = Field(
        default_factory=dict, description="artifact_id for each compression level"
    )

    # Size tracking
    size_bytes: int = Field(default=0, description="Size in bytes")
    size_tokens: int | None = Field(default=None, description="Estimated token count (for text/transcript)")

    # Access tracking (for LRU/eviction)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = Field(default=0, description="Number of times accessed")

    # Importance (affects eviction priority)
    # Claims default to higher importance
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Importance score for eviction decisions",
    )

    # State tracking
    dirty: bool = Field(default=False, description="Has been modified since last flush")
    pinned: bool = Field(default=False, description="Pinned pages are never evicted")

    # Lineage (legacy, use provenance/represents instead)
    parent_page_id: str | None = Field(default=None, description="Parent page if derived (e.g., summary of original)")

    # Modality-specific metadata
    mime_type: str | None = Field(default=None)
    duration_seconds: float | None = Field(default=None, description="Duration for audio/video")
    dimensions: tuple[int, int] | None = Field(default=None, description="Width x height for image/video")
    transcript: str | None = Field(default=None, description="Transcript for audio/video (L1 representation)")
    caption: str | None = Field(default=None, description="Caption for image (L2 representation)")

    # Custom metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    def mark_accessed(self) -> None:
        """Update access tracking."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1

    def mark_dirty(self) -> None:
        """Mark page as modified."""
        self.dirty = True

    def mark_clean(self) -> None:
        """Mark page as flushed/clean."""
        self.dirty = False

    def estimate_tokens(self) -> int:
        """Estimate token count for current content."""
        if self.size_tokens is not None:
            return self.size_tokens

        if self.content is None:
            return 0

        if isinstance(self.content, str):
            # Rough estimate: 4 chars per token
            return len(self.content) // 4
        elif isinstance(self.content, dict):
            import json

            return len(json.dumps(self.content)) // 4
        else:
            return self.size_bytes // 4 if self.size_bytes > 0 else 100


class PageTableEntry(BaseModel):
    """
    Metadata entry for a page in the page table.

    Tracks location, state, and access patterns without holding content.
    This is what the PageTable stores for each page.
    """

    page_id: str

    # Location
    tier: StorageTier
    artifact_id: str | None = None
    compression_level: CompressionLevel = CompressionLevel.FULL

    # Page type (for eviction decisions)
    page_type: PageType = Field(
        default=PageType.TRANSCRIPT,
        description="Page type for eviction/compression decisions",
    )

    # Provenance (for tracing back to source)
    provenance: list[str] = Field(
        default_factory=list,
        description="page_ids this page derives from",
    )

    # State
    dirty: bool = Field(default=False, description="Modified since last flush")
    pinned: bool = Field(default=False, description="Pinned pages are never evicted")
    last_flushed: datetime | None = Field(default=None)

    # Access tracking
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = Field(default=0)

    # Size
    size_tokens: int | None = None

    # Modality (for filtering)
    modality: Modality = Modality.TEXT

    # Locality hints (for NUMA awareness)
    affinity: Affinity = Field(default=Affinity.LOCAL, description="Locality hint for distributed storage")

    def mark_accessed(self) -> None:
        """Update access tracking."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1

    @property
    def eviction_priority(self) -> float:
        """
        Calculate eviction priority based on page type.

        Lower value = less likely to evict.
        """
        type_weights = {
            PageType.CLAIM: 0.1,  # Very low - claims are precious
            PageType.INDEX: 0.2,  # Very low - indexes are needed for search
            PageType.PROCEDURE: 0.3,  # Low - procedures help tool usage
            PageType.SUMMARY: 0.4,  # Low - rebuildable but useful
            PageType.ARTIFACT: 0.6,  # Normal
            PageType.TRANSCRIPT: 0.7,  # Normal
        }
        base = type_weights.get(self.page_type, 0.5)
        # Pinned pages get 0 priority (never evict)
        if self.pinned:
            return 0.0
        return base


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


# =============================================================================
# Fault Policy Models
# =============================================================================


class FaultConfidenceThreshold(str, Enum):
    """Confidence threshold for page faults."""

    EXPLICIT = "explicit"  # Only fault when page_id is directly requested
    REFERENCED = "referenced"  # Fault if page content is referenced/needed
    SPECULATIVE = "speculative"  # Fault on potential relevance (aggressive)


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
