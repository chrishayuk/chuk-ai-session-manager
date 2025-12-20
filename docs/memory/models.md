# Core Models and Enums

The memory system uses Pydantic models and Python enums throughout to ensure type safety and eliminate magic strings.

## Enums

### StorageTier

Defines where a page lives in the storage hierarchy.

```python
from chuk_ai_session_manager.memory import StorageTier

class StorageTier(str, Enum):
    L0 = "L0"  # Context window (active prompt)
    L1 = "L1"  # Hot cache (in-memory)
    L2 = "L2"  # Session storage
    L3 = "L3"  # Disk/artifact storage
    L4 = "L4"  # Cold/archive storage
```

**Usage:**
```python
page = MemoryPage(
    page_id="msg_001",
    modality=Modality.TEXT,
    storage_tier=StorageTier.L0,  # In context window
)

# Move to lower tier
page.storage_tier = StorageTier.L2
```

### Modality

Content type of a page.

```python
from chuk_ai_session_manager.memory import Modality

class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    STRUCTURED = "structured"  # JSON, tool outputs
```

**Usage:**
```python
# Text page
text_page = MemoryPage(page_id="msg_001", modality=Modality.TEXT, content="Hello")

# Image page
image_page = MemoryPage(
    page_id="img_001",
    modality=Modality.IMAGE,
    caption="Architecture diagram",
    dimensions=(1200, 800),
)

# Structured data page
struct_page = MemoryPage(
    page_id="tool_001",
    modality=Modality.STRUCTURED,
    content={"result": 42, "expression": "6 * 7"},
)
```

### CompressionLevel

Resolution level for page content.

```python
from chuk_ai_session_manager.memory import CompressionLevel

class CompressionLevel(int, Enum):
    FULL = 0       # Complete content
    REDUCED = 1    # Summarized/reduced
    ABSTRACT = 2   # Key points/caption only
    REFERENCE = 3  # ID reference only
```

**Compression by modality:**

| Level | Text | Image | Audio | Video |
|-------|------|-------|-------|-------|
| 0 | Full message | Full resolution | Full waveform | All frames |
| 1 | Summary | Thumbnail | Transcript | Keyframes |
| 2 | Key points | Caption | Summary | Scene descriptions |
| 3 | Topic tags | "image:{id}" | Duration | "video:{id}" |

### VMMode

Virtual memory operation mode.

```python
from chuk_ai_session_manager.memory import VMMode

class VMMode(str, Enum):
    STRICT = "strict"    # Model must cite sources, cannot fabricate
    RELAXED = "relaxed"  # Model can infer but prefers faulting
    PASSIVE = "passive"  # Model sees manifest but cannot fault
```

### MessageRole

Role of message content in conversation.

```python
from chuk_ai_session_manager.memory import MessageRole

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"
```

### ContextPrefix

Single-character prefix for VM:CONTEXT format.

```python
from chuk_ai_session_manager.memory import ContextPrefix

class ContextPrefix(str, Enum):
    USER = "U"        # User message
    ASSISTANT = "A"   # Assistant response
    TOOL = "T"        # Tool result
    SUMMARY = "S"     # Summary page
    IMAGE = "I"       # Image content
    AUDIO = "D"       # Audio content (D for "audio Data")
    VIDEO = "V"       # Video content
    STRUCTURED = "J"  # JSON/structured data
    UNKNOWN = "?"     # Unknown type
```

### Affinity

Page locality hint for distributed deployments.

```python
from chuk_ai_session_manager.memory import Affinity

class Affinity(str, Enum):
    LOCAL = "local"    # Page is local to compute
    REMOTE = "remote"  # Page is on remote storage
    SHARED = "shared"  # Page is in shared region
```

### PageType

Classification of page content. Different page types have different eviction/compression rules.

```python
from chuk_ai_session_manager.memory import PageType

class PageType(str, Enum):
    TRANSCRIPT = "transcript"  # Raw turns, tool outputs (normal eviction)
    SUMMARY = "summary"        # LLM-generated summaries (low priority - rebuildable)
    ARTIFACT = "artifact"      # Tool-created content (normal eviction)
    CLAIM = "claim"            # Decisions, facts, conclusions (very low - high value)
    PROCEDURE = "procedure"    # Learned patterns like "when calling X, do Y"
    INDEX = "index"            # Page metadata for search (never compressed)
```

**Eviction Priority by Type:**

| Page Type | Eviction Priority | Why |
|-----------|------------------|-----|
| `transcript` | Normal | Raw content, can be summarized |
| `summary` | Low | Derived, can be rebuilt from L2 |
| `artifact` | Normal | Tool outputs, type-dependent |
| `claim` | Very Low | High-value decisions, referenced constantly |
| `procedure` | Low | Learned patterns, rarely compressed |
| `index` | Very Low | Metadata for search, never compressed |

**Usage:**
```python
from chuk_ai_session_manager.memory import MemoryPage, PageType

# Claim page for a decision
claim = MemoryPage(
    page_id="claim_db_choice",
    page_type=PageType.CLAIM,
    content="Decision: Use PostgreSQL for the database",
    provenance=["msg_042", "msg_043"],  # Messages where decided
    importance=0.95,
)

# Summary page derived from transcripts
summary = MemoryPage(
    page_id="summary_seg_01",
    page_type=PageType.SUMMARY,
    content="Key points from turns 1-10...",
    provenance=["msg_001", "msg_002", "msg_003"],
)
```

### ToolType

Type of tool definition (for VM tools).

```python
from chuk_ai_session_manager.memory import ToolType

class ToolType(str, Enum):
    FUNCTION = "function"
```

## Constants

```python
from chuk_ai_session_manager.memory import (
    MEMORY_PAGE_MIME_TYPE,      # "application/x-memory-page"
    VM_CHECKPOINT_MIME_TYPE,    # "application/x-vm-checkpoint"
    ALL_COMPRESSION_LEVELS,     # [0, 1, 2, 3]
)
```

---

## Core Models

### MemoryPage

The fundamental unit of content in the memory system.

```python
from chuk_ai_session_manager.memory import MemoryPage, Modality, StorageTier

class MemoryPage(BaseModel):
    # Identity
    page_id: str
    modality: Modality = Modality.TEXT
    storage_tier: StorageTier = StorageTier.L0
    compression_level: CompressionLevel = CompressionLevel.FULL

    # Content (one of these based on modality)
    content: Optional[str] = None           # Text content
    caption: Optional[str] = None           # Image/video caption
    transcript: Optional[str] = None        # Audio transcript
    dimensions: Optional[Tuple[int, int]]   # Image/video size
    duration_seconds: Optional[float]       # Audio/video duration

    # Storage reference
    artifact_id: Optional[str] = None       # ID in artifact storage

    # Size tracking
    size_bytes: int = 0
    size_tokens: Optional[int] = None

    # Access patterns
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    importance: float = 0.5                 # 0.0-1.0, affects eviction

    # State
    dirty: bool = False                     # Modified since last flush

    # Metadata
    metadata: Dict[str, Any] = {}
```

**Key methods:**

```python
page = MemoryPage(page_id="msg_001", modality=Modality.TEXT, content="Hello")

# Estimate token count
tokens = page.estimate_tokens()  # Returns int

# Mark as accessed (updates access_count and last_accessed)
page.mark_accessed()

# Mark as modified
page.mark_dirty()

# Check if in working set
in_ws = page.storage_tier in (StorageTier.L0, StorageTier.L1)
```

### PageTableEntry

Metadata entry for tracking a page in the page table.

```python
from chuk_ai_session_manager.memory import PageTableEntry

class PageTableEntry(BaseModel):
    page_id: str
    tier: StorageTier
    artifact_id: Optional[str] = None
    compression_level: CompressionLevel = CompressionLevel.FULL
    dirty: bool = False
    last_flushed: Optional[datetime] = None
    last_accessed: datetime
    access_count: int = 0
    size_tokens: Optional[int] = None
    modality: Modality = Modality.TEXT
    affinity: Affinity = Affinity.LOCAL
```

**Key methods:**

```python
entry = PageTableEntry(page_id="msg_001", tier=StorageTier.L0, modality=Modality.TEXT)

# Update access tracking
entry.mark_accessed()
```

### TokenBudget

Tracks token allocation across modalities.

```python
from chuk_ai_session_manager.memory import TokenBudget

class TokenBudget(BaseModel):
    total_limit: int = 128_000    # Context window size
    reserved: int = 4000          # Reserved for system/tools
    used: int = 0                 # Currently used tokens
    tokens_by_modality: Dict[str, int] = {}
```

**Key properties and methods:**

```python
budget = TokenBudget(total_limit=128_000, reserved=4000)

# Check available space
available = budget.available      # total_limit - reserved - used
utilization = budget.utilization  # used / (total_limit - reserved)

# Check if tokens can fit
can_fit = budget.can_fit(1000)    # Returns bool

# Add/remove tokens
budget.add(500, Modality.TEXT)
budget.remove(200, Modality.TEXT)
```

### VMMetrics

Runtime metrics for the virtual memory system.

```python
from chuk_ai_session_manager.memory import VMMetrics

class VMMetrics(BaseModel):
    # Fault tracking
    faults_total: int = 0
    faults_this_turn: int = 0
    max_faults_per_turn: int = 3

    # TLB stats
    tlb_hits: int = 0
    tlb_misses: int = 0

    # Eviction tracking
    evictions_total: int = 0
    evictions_this_turn: int = 0

    # Token tracking
    tokens_in_working_set: int = 0
    tokens_available: int = 0
```

**Key methods:**

```python
metrics = VMMetrics(max_faults_per_turn=3)

# Record a fault
metrics.record_fault()

# Check if more faults allowed
can_fault = metrics.can_fault()

# Get TLB hit rate
hit_rate = metrics.tlb_hit_rate  # Returns float 0.0-1.0

# Reset per-turn counters
metrics.new_turn()
```

---

## Content Models

Type-safe content representations for different modalities.

### TextContent

```python
from chuk_ai_session_manager.memory import TextContent

class TextContent(BaseModel):
    text: str
```

### ImageContent

```python
from chuk_ai_session_manager.memory import ImageContent

class ImageContent(BaseModel):
    caption: Optional[str] = None
    url: Optional[str] = None
    base64: Optional[str] = None
    dimensions: Optional[Tuple[int, int]] = None
```

### AudioContent

```python
from chuk_ai_session_manager.memory import AudioContent

class AudioContent(BaseModel):
    transcript: Optional[str] = None
    duration_seconds: Optional[float] = None
    timestamps: Optional[List[Dict[str, Any]]] = None
```

### VideoContent

```python
from chuk_ai_session_manager.memory import VideoContent

class VideoContent(BaseModel):
    scenes: Optional[List[Dict[str, Any]]] = None
    transcript: Optional[str] = None
    duration_seconds: Optional[float] = None
    keyframe_urls: Optional[List[str]] = None
```

### StructuredContent

```python
from chuk_ai_session_manager.memory import StructuredContent

class StructuredContent(BaseModel):
    data: Dict[str, Any]
    schema_name: Optional[str] = None
```

### PageContent (Union Type)

```python
PageContent = Union[TextContent, ImageContent, AudioContent, VideoContent, StructuredContent]
```

---

## Stats Models

Models for returning statistics from various components.

### TLBStats

```python
from chuk_ai_session_manager.memory import TLBStats

class TLBStats(BaseModel):
    size: int = 0           # Current entries
    max_size: int = 512     # Maximum capacity
    utilization: float = 0.0
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
```

### WorkingSetStats

```python
from chuk_ai_session_manager.memory import WorkingSetStats

class WorkingSetStats(BaseModel):
    l0_pages: int = 0
    l1_pages: int = 0
    total_pages: int = 0
    tokens_used: int = 0
    tokens_available: int = 0
    utilization: float = 0.0
    needs_eviction: bool = False
    tokens_by_modality: Dict[str, int] = {}
```

### PageTableStats

```python
from chuk_ai_session_manager.memory import PageTableStats

class PageTableStats(BaseModel):
    total_pages: int = 0
    dirty_pages: int = 0
    pages_by_tier: Dict[StorageTier, int] = {}
    pages_by_modality: Dict[Modality, int] = {}

    @property
    def working_set_size(self) -> int:
        return self.pages_by_tier.get(StorageTier.L0, 0) + \
               self.pages_by_tier.get(StorageTier.L1, 0)
```

### StorageStats

```python
from chuk_ai_session_manager.memory import StorageStats

class StorageStats(BaseModel):
    backend: str = "unknown"
    persistent: bool = False
    pages_stored: int = 0
    total_bytes: int = 0
```

### CombinedPageTableStats

```python
from chuk_ai_session_manager.memory import CombinedPageTableStats

class CombinedPageTableStats(BaseModel):
    page_table: PageTableStats
    tlb: TLBStats
```

### FaultMetrics

```python
from chuk_ai_session_manager.memory import FaultMetrics

class FaultMetrics(BaseModel):
    faults_this_turn: int = 0
    max_faults_per_turn: int = 3
    faults_remaining: int = 3
    can_fault: bool = True
```

---

## Tool Definition Models

Models for defining VM tools (page_fault, search_pages).

### ToolParameter

```python
from chuk_ai_session_manager.memory import ToolParameter

class ToolParameter(BaseModel):
    type: str = "string"
    description: Optional[str] = None
    enum: Optional[List[str]] = None
    default: Optional[Any] = None
    minimum: Optional[int] = None
    maximum: Optional[int] = None
```

### ToolParameters

```python
from chuk_ai_session_manager.memory import ToolParameters

class ToolParameters(BaseModel):
    type: str = "object"
    properties: Dict[str, ToolParameter] = {}
    required: List[str] = []
```

### ToolFunction

```python
from chuk_ai_session_manager.memory import ToolFunction

class ToolFunction(BaseModel):
    name: str
    description: str
    parameters: ToolParameters
```

### ToolDefinition

```python
from chuk_ai_session_manager.memory import ToolDefinition, ToolType

class ToolDefinition(BaseModel):
    type: ToolType = ToolType.FUNCTION
    function: ToolFunction
```

---

## Result Models

### PageData

Canonical envelope for page content in tool results.

```python
from chuk_ai_session_manager.memory import PageData, PageMeta

class PageData(BaseModel):
    page_id: str
    modality: str
    level: int
    tier: str
    content: PageContent  # Union of content types
    meta: PageMeta = PageMeta()
```

### PageMeta

```python
from chuk_ai_session_manager.memory import PageMeta

class PageMeta(BaseModel):
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    source_tier: Optional[str] = None
    dimensions: Optional[Tuple[int, int]] = None
    duration_seconds: Optional[float] = None
    word_count: Optional[int] = None
```

### FaultEffects

Side effects of a page fault operation.

```python
from chuk_ai_session_manager.memory import FaultEffects

class FaultEffects(BaseModel):
    promoted_to_working_set: bool = False
    tokens_est: int = 0
    evictions: Optional[List[str]] = None
```

### FormattedPage

Result of packing a page for context.

```python
from chuk_ai_session_manager.memory import FormattedPage

class FormattedPage(BaseModel):
    content: str      # Formatted string
    tokens_est: int   # Estimated tokens
```

### SearchResultEntry

Single result from page search.

```python
from chuk_ai_session_manager.memory import SearchResultEntry

class SearchResultEntry(BaseModel):
    page_id: str
    modality: str
    tier: str
    levels: List[int] = []
    hint: str = ""
    relevance: float = 0.0
```

---

## v0.8 Models

### FaultReason

Explicit intent for page faults - enables measuring *why* faults happen.

```python
from chuk_ai_session_manager.memory import FaultReason

class FaultReason(str, Enum):
    USER_REQUESTED_RECALL = "user_requested_recall"  # "What did we say about X?"
    RESOLVE_REFERENCE = "resolve_reference"          # Model references page_id
    TOOL_PREREQUISITE = "tool_prereq"                # Tool needs this page
    SPECULATIVE = "speculative"                      # Might be relevant
```

**Fault Reason Breakdown (Expected Distribution):**

| Reason | Expected % | Red Flag If |
|--------|------------|-------------|
| `user_requested_recall` | 40-60% | < 20% (model faulting too speculatively) |
| `resolve_reference` | 20-40% | > 60% (poor working set selection) |
| `tool_prereq` | 10-20% | > 40% (tools not getting needed pages) |
| `speculative` | < 10% | > 20% (wasting budget on guesses) |

### FaultPolicy

Guardrails to prevent fault spirals and budget blowouts.

```python
from chuk_ai_session_manager.memory import FaultPolicy, FaultReason

policy = FaultPolicy(
    max_faults_per_turn=3,
    max_fault_tokens_per_turn=8192,
)

# Check if fault is allowed
if policy.can_fault():
    policy.record_fault(
        page_id="claim_001",
        reason=FaultReason.USER_REQUESTED_RECALL,
        tokens=50,
    )

# Start new turn (resets counters)
policy.new_turn()

# Get remaining budget
print(f"Faults remaining: {policy.faults_remaining}")
print(f"Token budget remaining: {policy.tokens_remaining}")
```

### MemoryABI

Application Binary Interface - explicit contract between memory system and models.

```python
from chuk_ai_session_manager.memory import MemoryABI

abi = MemoryABI(
    max_context_tokens=128_000,
    reserved_tokens=2_000,              # System prompt
    tool_schema_tokens_reserved=4_500,  # 15 tools @ ~300 tokens each
)

# Available tokens for content
print(f"Available: {abi.available_tokens}")  # 128000 - 2000 - 4500 = 121500
```

### UserExperienceMetrics

Metrics that correlate with user satisfaction.

```python
from chuk_ai_session_manager.memory import UserExperienceMetrics, FaultReason

metrics = UserExperienceMetrics()

# Record faults
metrics.record_fault(FaultReason.USER_REQUESTED_RECALL)
metrics.record_fault(FaultReason.RESOLVE_REFERENCE)

# Record recall attempts
metrics.record_recall_attempt(success=True)
metrics.record_recall_attempt(success=False)

# Get UX metrics
print(f"Recall success rate: {metrics.recall_success_rate():.2%}")
print(f"Thrash index: {metrics.thrash_index():.2f}")
print(f"Fault breakdown: {metrics.fault_breakdown()}")
```

### PageMutation

Immutable record of a page change for the mutation log.

```python
from chuk_ai_session_manager.memory import PageMutation, MutationType, Actor

mutation = PageMutation(
    page_id="msg_001",
    turn=5,
    mutation_type=MutationType.EVICT,
    tier_before="L0",
    tier_after="L2",
    actor=Actor.SYSTEM,
    cause="eviction_pressure",
)
```
