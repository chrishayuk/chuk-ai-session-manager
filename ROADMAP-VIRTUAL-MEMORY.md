# AI Virtual Memory - Roadmap

> Treating AI context like operating system memory: pages, working sets, faults, and swapping.

## Why This Isn't RAG

| Capability | RAG / Memori | AI Virtual Memory |
|------------|--------------|-------------------|
| Page identity & lifecycle | ❌ Chunks are anonymous | ✅ Pages have IDs, versions, lineage |
| Multi-resolution per page | ❌ Single embedding | ✅ Full → Summary → Caption → Reference |
| Working set semantics | ❌ Top-K retrieval | ✅ Explicit hot/warm/cold tracking |
| Deterministic eviction | ❌ Implicit | ✅ Policy-driven (LRU, importance-weighted) |
| Coherency guarantees | ❌ None | ✅ Dirty tracking, flush, checkpoint consistency |
| Copy-on-write | ❌ Impossible | ✅ Fork memory state efficiently |
| Cross-modal causality | ❌ Separate indexes | ✅ Unified page model, cross-references |
| Checkpoint consistency | ❌ None | ✅ Point-in-time snapshots |
| Tools as memory writers | ❌ External | ✅ First-class page creators/mutators |
| Streaming fault handling | ❌ Block or fail | ✅ Speculative placeholders, async upgrade |
| Distinct content types | ❌ Everything is chunks | ✅ Transcript vs derived vs claims |
| Provenance tracking | ❌ Lost on retrieval | ✅ Claims cite source pages |

**Core difference in one sentence:**
> RAG optimizes *retrieval*. AI Virtual Memory *virtualizes memory*.

---

## Vision

Just as operating systems virtualize physical memory to give processes the illusion of infinite address space, AI Virtual Memory virtualizes context windows to give AI conversations the illusion of infinite memory across all modalities (text, images, audio, video).

```
┌────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                          │
│         AI Agent / Chat / Tool Orchestration                   │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                   VIRTUAL MEMORY LAYER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Page Table   │  │ Working Set  │  │ Fault Handler        │ │
│  │              │  │ Manager      │  │                      │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Compressor   │  │ Eviction     │  │ Prefetch             │ │
│  │ (per-modal)  │  │ Policy       │  │ Manager              │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                    STORAGE HIERARCHY                           │
│                                                                │
│   L0 (Registers)  │  Current prompt context window            │
│   L1 (Cache)      │  Recent pages, hot data (session state)   │
│   L2 (RAM)        │  Session storage (chuk-sessions)          │
│   L3 (Disk)       │  Artifact storage (chuk-artifacts)        │
│   L4 (Cold)       │  Archive/S3 (chuk-artifacts vfs-s3)       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Storage Hierarchy Detail

| Tier | Name | Latency | Capacity | Content | Implementation |
|------|------|---------|----------|---------|----------------|
| L0 | Registers | 0ms | ~128K tokens | Active prompt window | In-memory list |
| L1 | Cache | <10ms | ~1M tokens | **Derived materializations** (summaries, claims, packed context) | Session.state |
| L2 | RAM | <100ms | ~100M tokens | **Authoritative event store** (append-only session events, tool results) | chuk-sessions |
| L3 | Disk | <1s | ~10GB | Artifacts, media, checkpoints | chuk-artifacts (vfs-filesystem) |
| L4 | Cold | 1-10s | Unlimited | Archives, old sessions | chuk-artifacts (vfs-s3) |

### L1/L2 Contract (Critical)

**L2 is source-of-truth.** Append-only event log of everything that happened:
- User messages, assistant responses, tool calls/results
- Never mutated, only appended
- Coherency is guaranteed because it's immutable

**L1 is derived.** Materialized views that can be rebuilt from L2:
- Summaries, extracted claims, packed context
- Dirty tracking is about L1 pages that haven't been persisted yet
- If L1 is lost, rebuild from L2

This makes coherency tractable: L2 is the log, L1 is the cache.

---

## Phase 1: Foundations (v0.8)

### 1.1 Page Type Taxonomy

Not all pages are equal. Different page types have different eviction/compression rules:

| Page Type | Description | Eviction Priority | Compression |
|-----------|-------------|-------------------|-------------|
| `transcript` | Raw turns, tool outputs | Normal | Full → Summary → Reference |
| `summary` | LLM-generated summaries | Low (derived, rebuildable) | Already compressed |
| `artifact` | Tool-created content | Normal | Type-dependent |
| `claim` | Decisions, facts, conclusions | **Very Low** (high-value, referenced constantly) | Rarely compressed |
| `procedure` | "When calling tool X, we do Y" | Low | Rarely compressed |
| `index` | Page metadata for search | Very Low | Never compressed |

**Why `claim` pages matter**: Without first-class claims, high-value decisions get buried in summaries and you pay fault/compression costs repeatedly. Claims are token-cheap, high-importance, and have explicit provenance back to source pages.

### 1.2 MemoryPage Model
Core abstraction representing any piece of content across modalities.

```python
class MemoryPage(BaseModel):
    page_id: str
    modality: Literal["text", "image", "audio", "video", "structured"]
    storage_tier: Literal["L0", "L1", "L2", "L3", "L4"]

    # Page type (critical for eviction/compression decisions)
    page_type: Literal["transcript", "summary", "artifact", "claim", "procedure", "index"]

    # Provenance: what pages justify this one (for claims, summaries)
    provenance: List[str] = []  # page_ids that this page derives from

    # Representation linking (for compression chain)
    represents: Optional[str] = None  # page_id this is a compressed version of
    representation_level: int = 0     # 0=full, 1=reduced, 2=abstract, 3=reference

    # Content or reference
    content: Optional[Any] = None
    artifact_id: Optional[str] = None

    # Multi-resolution representations (artifact_ids for each level)
    representations: Dict[int, str] = {}  # level -> artifact_id

    # Access tracking
    size_bytes: int
    size_tokens: Optional[int] = None
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    importance: float = 0.5

    # Modality-specific metadata
    mime_type: str
    duration_seconds: Optional[float] = None  # audio/video
    dimensions: Optional[Tuple[int, int]] = None  # image/video
```

#### Claim Page Example

```python
# When user says "Let's use PostgreSQL for the database"
claim_page = MemoryPage(
    page_id="claim_db_choice_001",
    page_type="claim",
    modality="text",
    content="Decision: Use PostgreSQL for the database",
    provenance=["msg_042", "msg_043"],  # The messages where this was decided
    importance=0.95,  # High importance = low eviction priority
    size_tokens=12,   # Token-cheap
)
```

#### Procedure Page Example

```python
# Learned pattern: "When calling weather_tool, we usually load location claims first"
procedure_page = MemoryPage(
    page_id="proc_weather_pattern_001",
    page_type="procedure",
    modality="structured",
    content={
        "tool": "weather_tool",
        "prereqs": ["claim_location_*"],
        "typical_pages_read": ["msg_user_location", "claim_timezone"],
    },
    provenance=["tool_trace_087", "tool_trace_092"],
    importance=0.8,
)
```

### 1.3 PageTableEntry (with dirty tracking)
Core metadata for each page - includes dirty bit from day one.

```python
class PageTableEntry(BaseModel):
    page_id: str
    tier: Literal["L0", "L1", "L2", "L3", "L4"]
    artifact_id: Optional[str] = None
    compression_level: int = 0

    # Dirty tracking (critical for coherency)
    dirty: bool = False
    last_flushed: Optional[datetime] = None

    # Access tracking
    last_accessed: datetime
    access_count: int = 0

    # Locality hints (for future NUMA awareness)
    affinity: Literal["local", "remote", "shared"] = "local"
```

### 1.4 PageTable
Maps page IDs to their current location and state.

```python
class PageTable:
    pages: Dict[str, PageTableEntry]

    async def lookup(self, page_id: str) -> PageTableEntry
    async def register(self, page: MemoryPage) -> None
    async def update_location(self, page_id: str, tier: str, artifact_id: str) -> None
    async def mark_accessed(self, page_id: str) -> None
    async def mark_dirty(self, page_id: str) -> None
    async def get_dirty_pages(self) -> List[PageTableEntry]
    async def get_by_tier(self, tier: str) -> List[PageTableEntry]
```

### 1.5 TLB (Translation Lookaside Buffer)
Fast cache for recent page lookups - avoids PageTable + storage hops.

```python
class PageTLB:
    """
    Without this, fault rate metrics lie - you'll be bottlenecked
    on metadata lookups, not content.
    """
    max_entries: int = 512
    entries: Dict[str, PageTableEntry]
    lru: OrderedDict  # For eviction

    def lookup(self, page_id: str) -> Optional[PageTableEntry]:
        """O(1) lookup, updates LRU on hit"""

    def insert(self, entry: PageTableEntry) -> None:
        """Insert with LRU eviction if full"""

    def invalidate(self, page_id: str) -> None:
        """Remove entry (on page mutation/eviction)"""

    def flush(self) -> None:
        """Clear entire TLB (on checkpoint/context switch)"""
```

### 1.6 Compression Levels
Define standard compression levels per modality.

| Level | Text | Image | Audio | Video |
|-------|------|-------|-------|-------|
| 0 (Full) | Complete messages | Full resolution | Full waveform | All frames |
| 1 (Reduced) | Recent + summaries | Thumbnail | Transcript + timestamps | Keyframes + transcript |
| 2 (Abstract) | Key points only | Caption + embedding | Summary + key quotes | Scene descriptions |
| 3 (Reference) | Topic tags | "image:{id}" | Duration + topic | "video:{id}" |

### 1.7 Context Pack Cache

The TLB caches address translations. But in LLM terms, the expensive part is often:
- Selecting pages
- Choosing compression levels
- Rendering into `VM:CONTEXT`

Add a cache for the *packed context output*:

```python
class ContextPackCache:
    """
    Cache packed context to avoid re-packing on small incremental turns.
    Invalidate on working set changes.
    """
    max_entries: int = 32

    def get(
        self,
        session_id: str,
        model_id: str,
        token_budget: int,
        working_set_hash: str,  # Hash of page_ids + versions in working set
    ) -> Optional[PackedContext]:
        """O(1) lookup for cached pack"""

    def put(
        self,
        session_id: str,
        model_id: str,
        token_budget: int,
        working_set_hash: str,
        packed: PackedContext,
    ) -> None:
        """Store packed context, evict LRU if full"""

    def invalidate_session(self, session_id: str) -> None:
        """Invalidate all cached packs for session (on working set change)"""
```

This drastically reduces overhead for "small incremental turns" - user says something short, assistant responds, no pages changed, reuse the pack.

### 1.8 Deliverables
- [ ] `MemoryPage` model with `page_type` and `provenance` in `memory/models.py`
- [ ] `PageTableEntry` with dirty tracking
- [ ] `PageTable` class in `memory/page_table.py`
- [ ] `PageTLB` class in `memory/tlb.py`
- [ ] `ContextPackCache` class in `memory/pack_cache.py`
- [ ] `CompressionLevel` enum and per-modality schemas
- [ ] Unit tests for page lifecycle

---

## Phase 2: Working Set Management (v0.9)

### 2.1 WorkingSetManager
Tracks which pages are "hot" and manages L0/L1 capacity.

```python
class WorkingSetManager:
    max_l0_tokens: int  # Context window limit
    max_l1_pages: int   # Cache size

    async def add_to_working_set(self, page: MemoryPage) -> None
    async def get_working_set(self) -> List[MemoryPage]
    async def get_token_usage(self) -> TokenBudget
    async def needs_eviction(self) -> bool
    async def get_eviction_candidates(self) -> List[MemoryPage]
```

### 2.2 Token Budget
Track token allocation across modalities.

```python
class TokenBudget:
    total_limit: int
    text_tokens: int
    image_tokens: int  # Estimated token cost
    audio_tokens: int  # Transcript tokens
    reserved: int      # System prompt, tools
    available: int
```

### 2.3 Pinned Pages

Some pages should never be evicted. These form the "stable context" the model relies on.

```python
class PinnedSet:
    """
    Pages that are never evicted from working set.
    Pinning prevents thrash on critical context.
    """
    pinned: Set[str]  # page_ids

    # Auto-pinned by default:
    # - System prompt page
    # - Active goal/plan page
    # - User preferences page
    # - Current tool schemas (see ABI)
    # - Last N turns (configurable, typically 2-4)

    def pin(self, page_id: str) -> None
    def unpin(self, page_id: str) -> None
    def is_pinned(self, page_id: str) -> bool
```

Default pinned set (configurable):
- System prompt
- Active plan/goals
- User preferences
- Last 3 turns (user + assistant pairs)
- Current tool schemas

### 2.4 Anti-Thrash Protection

OS working sets fail when you thrash. LLM working sets thrash when user toggles between topics.

```python
class AntiThrashPolicy:
    """
    Prevent evicting pages that were just faulted in.
    """
    # Recently-evicted pages get a "do not evict again" window
    eviction_cooldown_turns: int = 3

    # Recently-faulted pages get temporary protection
    fault_protection_turns: int = 2

    # Track eviction/fault history
    eviction_history: Dict[str, int]  # page_id -> turn_evicted
    fault_history: Dict[str, int]     # page_id -> turn_faulted

    def can_evict(self, page_id: str, current_turn: int) -> bool:
        """Check if page is in cooldown period"""
        if page_id in self.fault_history:
            if current_turn - self.fault_history[page_id] < self.fault_protection_turns:
                return False
        return True

    def get_eviction_penalty(self, page_id: str, current_turn: int) -> float:
        """Higher penalty = less likely to evict"""
        # Recently faulted = high penalty
        # Recently evicted = high penalty (avoid re-evicting)
```

This is the difference between "looks good on paper" and "feels stable in chat".

### 2.5 Access Pattern Tracking
- LRU tracking per page
- Frequency counting (LFU hybrid)
- Importance scoring (user-marked, referenced by tools)
- **Topic affinity**: Pages accessed together get co-eviction bonus

### 2.6 Simple Prefetch Heuristics (v0.9)

Don't wait for ML prediction. Start with dumb heuristics that work:

```python
class SimplePrefetcher:
    """
    Basic prefetch that doesn't need prediction models.
    Good enough for v0.9.
    """
    async def prefetch_on_turn_start(self, session_id: str) -> List[str]:
        pages_to_prefetch = []

        # 1. Last segment summary (almost always needed for "what did we discuss")
        pages_to_prefetch.append(await self.get_last_segment_summary(session_id))

        # 2. Most-referenced claim pages (high access_count claims)
        pages_to_prefetch.extend(await self.get_top_claims(session_id, limit=3))

        # 3. Tool traces for likely tools (based on recent tool usage)
        likely_tools = await self.get_likely_tools(session_id)
        for tool in likely_tools:
            pages_to_prefetch.extend(await self.get_tool_prereq_pages(tool))

        return pages_to_prefetch
```

### 2.7 Lite Mutation Log (v0.9)

Start event-sourcing early (not full time-travel, but enough to debug and replay):

```python
class MutationLogLite:
    """
    Append-only log of page operations.
    Not full event-sourcing, but enough for:
    - Debugging: "what was in context for turn T?"
    - Replay: Reconstruct state for testing
    - Grounding story: Prove what the model saw
    """

    async def append(self, mutation: PageMutation) -> None:
        """Append mutation to log"""

    async def get_context_at_turn(self, session_id: str, turn: int) -> List[str]:
        """Replay: what page_ids were in L0 at turn T?"""

    async def get_history(self, page_id: str) -> List[PageMutation]:
        """All mutations for a page"""

class PageMutation(BaseModel):
    """Immutable record of a page change"""
    mutation_id: str
    page_id: str
    timestamp: datetime
    turn: int
    mutation_type: Literal["create", "fault_in", "evict", "compress", "pin", "unpin"]

    # Context at mutation time
    tier_before: Optional[str]
    tier_after: str

    # Who caused it
    actor: Literal["user", "model", "tool", "system"]
    cause: Optional[str]  # "eviction_pressure", "page_fault", "explicit_request"
```

This is gold for debugging agent weirdness and supports the strict grounding story.

### 2.8 Deliverables
- [ ] `WorkingSetManager` class with pinning support
- [ ] `PinnedSet` management
- [ ] `AntiThrashPolicy` implementation
- [ ] `SimplePrefetcher` with basic heuristics
- [ ] `MutationLogLite` for debugging/replay
- [ ] `TokenBudget` model with modality breakdown
- [ ] Access pattern tracking integration
- [ ] Working set size configuration per model

---

## Phase 3: Page Faults & Loading (v0.10)

### 3.1 PageFaultHandler
Handles requests for pages not in L0/L1.

```python
class PageFaultHandler:
    async def handle_fault(self, page_id: str, target_level: int = 0) -> MemoryPage:
        """
        1. Lookup page location in PageTable
        2. Load from appropriate tier (L2/L3/L4)
        3. Decompress if needed
        4. Add to working set
        5. Trigger eviction if needed
        """
```

### 3.2 Streaming Fault Handling
Handle page faults during streaming responses without blocking.

```python
class DeferredPage:
    """
    Placeholder for pages being loaded mid-stream.
    Like how browsers render HTML while images load.
    """
    page_id: str
    placeholder: str  # "Loading context about X..."
    compression_hint: int  # What level to load at
    future: asyncio.Future[MemoryPage]

    async def await_or_placeholder(self, timeout_ms: int = 100) -> Union[MemoryPage, str]:
        """Return page if ready, placeholder if still loading"""

class StreamingFaultHandler:
    pending: Dict[str, DeferredPage]

    async def request_async(self, page_id: str, urgency: str = "normal") -> DeferredPage:
        """
        Non-blocking fault request.
        Returns immediately with DeferredPage.
        Background task loads actual content.
        """

    async def upgrade_placeholder(self, page_id: str, page: MemoryPage) -> None:
        """
        Called when page loads - can inject into ongoing stream
        if model supports mid-stream context injection.
        """
```

### 3.3 Fault Policies

The real failure mode isn't "can't load a page" - it's the model spiraling into repeated faults because it's unsure.

```python
class FaultPolicy:
    """
    Guardrails to prevent fault spirals and budget blowouts.
    """
    # Existing
    max_faults_per_turn: int = 3

    # NEW: Token budget for fault resolution
    max_fault_tokens_per_turn: int = 8192  # Don't let faults blow the budget

    # NEW: Confidence threshold - only fault if explicitly needed
    fault_confidence_threshold: Literal["explicit", "referenced", "speculative"] = "referenced"
    # explicit: Only fault when page_id is directly requested
    # referenced: Fault if page content is referenced/needed
    # speculative: Fault on potential relevance (aggressive)

    # NEW: Track fault intent for metrics
    def validate_fault(
        self,
        page_id: str,
        reason: FaultReason,
        tokens_used_this_turn: int,
    ) -> bool:
        """Check if fault is allowed under current policy"""
        if tokens_used_this_turn + estimated_tokens > self.max_fault_tokens_per_turn:
            return False
        if self.fault_confidence_threshold == "explicit" and reason != "user_requested":
            return False
        return True

class FaultReason(str, Enum):
    """Make fault intent explicit for measurement"""
    USER_REQUESTED_RECALL = "user_requested_recall"  # "What did we say about X?"
    RESOLVE_REFERENCE = "resolve_reference"          # Model references page_id
    TOOL_PREREQUISITE = "tool_prereq"                # Tool needs this page
    SPECULATIVE = "speculative"                      # Might be relevant
```

#### Fault Request with Intent

```python
# Updated page_fault tool call
page_fault(
    page_id: str,
    target_level: int = 2,
    reason: FaultReason = "resolve_reference"  # NEW: explicit intent
)
```

This enables:
- Measuring *why* faults happen, not just *how many*
- Distinguishing good faults (user asked) from bad faults (model guessing)
- Tuning policies based on fault reason success rates

### 3.4 Lazy Loading Strategies
- **On-demand**: Load when explicitly referenced
- **Predictive**: Prefetch based on access patterns
- **Contextual**: Load related pages (same session segment)
- **Speculative**: Start loading on reference detection, use placeholder if not ready

### 3.5 ArtifactsBridge
Integration layer with chuk-artifacts.

```python
class ArtifactsBridge:
    store: ArtifactStore

    async def store_page(self, page: MemoryPage, tier: str) -> str  # Returns artifact_id
    async def load_page(self, artifact_id: str) -> MemoryPage
    async def stream_page(self, artifact_id: str) -> AsyncIterator[bytes]
    async def checkpoint_pages(self, page_ids: List[str], name: str) -> str
```

### 3.6 Deliverables
- [ ] `PageFaultHandler` class
- [ ] `FaultPolicy` with token budget and confidence threshold
- [ ] `FaultReason` enum for intent tracking
- [ ] `StreamingFaultHandler` with `DeferredPage`
- [ ] `ArtifactsBridge` integration
- [ ] Lazy loading configuration
- [ ] Fault metrics and logging (with reason breakdown)

---

## Phase 4: Eviction & Compression (v0.11)

### 4.1 EvictionPolicy
Decides which pages to move to lower tiers.

```python
class EvictionPolicy(Protocol):
    async def select_victims(
        self,
        pages: List[MemoryPage],
        required_tokens: int
    ) -> List[MemoryPage]

class LRUEvictionPolicy(EvictionPolicy): ...
class ImportanceWeightedLRU(EvictionPolicy): ...
class ModalityAwareLRU(EvictionPolicy): ...  # Keep text, evict media first
```

### 4.2 Modality Compressors
Transform content to lower compression levels.

```python
class Compressor(Protocol):
    modality: str
    async def compress(self, page: MemoryPage, target_level: int) -> MemoryPage

class TextCompressor(Compressor):
    async def compress(self, page, level):
        if level == 1: return await self.summarize(page)
        if level == 2: return await self.extract_key_points(page)
        if level == 3: return self.extract_topics(page)

class ImageCompressor(Compressor):
    async def compress(self, page, level):
        if level == 1: return await self.create_thumbnail(page)
        if level == 2: return await self.generate_caption(page)
        if level == 3: return self.create_reference(page)

class AudioCompressor(Compressor):
    async def compress(self, page, level):
        if level == 1: return await self.transcribe(page)
        if level == 2: return await self.summarize_transcript(page)
        if level == 3: return self.create_reference(page)

class VideoCompressor(Compressor):
    async def compress(self, page, level):
        if level == 1: return await self.extract_keyframes_and_transcript(page)
        if level == 2: return await self.generate_scene_descriptions(page)
        if level == 3: return self.create_reference(page)
```

### 4.3 Compression Triggers
- Token pressure (approaching context limit)
- Time-based (pages older than threshold)
- Explicit (user/system request)

### 4.4 Deliverables
- [ ] `EvictionPolicy` interface and implementations
- [ ] Modality-specific compressors
- [ ] Compression level transitions
- [ ] Eviction metrics and logging

---

## Phase 5: Multi-Modal Integration (v0.12)

### 5.1 Unified Content Addressing
Reference any content uniformly.

```python
# Text reference
"Regarding what you said earlier..."  →  page_fault("msg_abc123")

# Image reference
"In that chart I showed you..."  →  page_fault("img_xyz789")

# Audio reference
"Remember when I mentioned..."  →  page_fault("audio_def456")

# Cross-modal
"The data from the spreadsheet matches the graph"  →  [page_fault("xlsx_..."), page_fault("img_...")]
```

### 5.2 Context Window Packing
Intelligently pack multi-modal content into context.

```python
class ContextPacker:
    async def pack(
        self,
        pages: List[MemoryPage],
        token_budget: int,
        priorities: Dict[str, float]  # modality -> weight
    ) -> PackedContext:
        """
        1. Sort by importance * recency * priority_weight
        2. Select compression level per page to fit budget
        3. Return packed context with manifest
        """
```

### 5.3 Modality Handlers
- **Text**: Direct inclusion or summary
- **Image**: Inline (base64/URL) or caption
- **Audio**: Transcript with timestamps
- **Video**: Keyframes + transcript or description

### 5.4 Memory ABI (Application Binary Interface)
Explicit contract between memory system and models.

```python
class MemoryABI(BaseModel):
    """
    Lets different models negotiate memory requirements.
    Smaller models survive with aggressive compression.
    Tool processors reason about memory cost.
    """
    # What's in context
    pages: List[PageManifestEntry]

    # Capabilities
    faults_allowed: bool = True  # Can model request pages mid-turn?
    upgrade_budget_tokens: int = 2048  # Tokens reserved for fault resolution

    # Constraints
    max_context_tokens: int
    reserved_tokens: int  # System prompt, etc.
    available_tokens: int

    # NEW: Tool schema budget (often the hidden token hog)
    tool_schema_tokens_reserved: int = 0  # Tokens consumed by tool definitions
    active_toolset_hash: Optional[str] = None  # For cache invalidation when tools change

    # Preferences
    modality_weights: Dict[str, float] = {
        "text": 1.0,
        "image": 0.8,
        "audio": 0.6,
        "video": 0.4
    }

class PageManifestEntry(BaseModel):
    page_id: str
    modality: str
    page_type: str  # NEW: transcript, summary, artifact, claim, procedure, index
    compression_level: int
    tokens: int
    importance: float
    provenance: List[str] = []  # NEW: source page_ids (for claims/summaries)
    can_evict: bool = True
    can_compress: bool = True
```

#### Tool Schema as Pinned Pages

In real agent loops, tool schemas are often the hidden token hog. The ABI makes this explicit:

```python
# Tool schemas should be treated as quasi-pinned pages
# They're implicitly in context but consume budget

abi = MemoryABI(
    max_context_tokens=128_000,
    tool_schema_tokens_reserved=4_500,  # 15 tools @ ~300 tokens each
    reserved_tokens=2_000,              # System prompt
    # available_tokens = 128_000 - 4_500 - 2_000 = 121_500 for content
)

# When tools change, invalidate context pack cache
if new_toolset_hash != abi.active_toolset_hash:
    context_pack_cache.invalidate_session(session_id)
```

This enables:
- Models to request specific pages by ID
- Negotiation of compression levels
- Cost-aware tool planning
- Graceful degradation for smaller models

### 5.5 Deliverables
- [ ] Unified page reference syntax
- [ ] `ContextPacker` with multi-modal support
- [ ] `MemoryABI` specification and serialization
- [ ] Modality handlers for LLM context building
- [ ] Cross-modal reference tracking

---

## Phase 6: Prefetch & Prediction (v0.13)

### 6.1 PrefetchManager
Proactively load pages likely to be needed.

```python
class PrefetchManager:
    async def predict_needed_pages(self, context: ConversationContext) -> List[str]
    async def prefetch(self, page_ids: List[str]) -> None
    async def learn_from_access(self, accessed: str, context: ConversationContext) -> None
```

### 6.2 Prediction Strategies
- **Sequential**: Next segment in session chain
- **Similarity**: Pages with similar embeddings to current context
- **Tool-based**: Pages referenced by likely tool calls
- **User pattern**: Based on historical access patterns

### 6.3 Background Loading
- Async prefetch during idle time
- Priority queue for prefetch requests
- Cancel prefetch on context switch

### 6.4 Deliverables
- [ ] `PrefetchManager` class
- [ ] Prediction strategy plugins
- [ ] Background loading infrastructure
- [ ] Prefetch hit rate metrics

---

## Phase 7: Memory Pressure & Health (v0.14)

### 7.1 MemoryPressureMonitor
Track and respond to memory pressure.

```python
class MemoryPressureMonitor:
    async def get_pressure_level(self) -> PressureLevel  # LOW, MEDIUM, HIGH, CRITICAL
    async def get_metrics(self) -> MemoryMetrics
    def on_pressure_change(self, callback: Callable) -> None

class MemoryMetrics:
    l0_utilization: float  # 0-1
    l1_utilization: float
    pages_by_tier: Dict[str, int]
    evictions_per_minute: float
    fault_rate: float
    compression_ratio: float
```

### 7.2 Adaptive Thresholds
- Adjust eviction aggressiveness based on fault rate
- Tune compression levels based on retrieval patterns
- Model-specific context window optimization

### 7.3 Health Checks
- Consistency between PageTable and storage
- Orphaned artifacts cleanup
- Corruption detection

### 7.4 Deliverables
- [ ] `MemoryPressureMonitor` class
- [ ] Adaptive threshold tuning
- [ ] Health check routines
- [ ] Observability dashboard integration

---

## Phase 8: Coherency & Versioning (v0.15)

### 8.1 Page Versioning
Track page mutations over time.

```python
class PageVersion:
    version_id: str
    page_id: str
    timestamp: datetime
    compression_level: int
    artifact_id: str
    parent_version: Optional[str]
```

### 8.2 Coherency Guarantees
- **Read-after-write**: Immediate visibility of writes
- **Checkpoint consistency**: All pages at checkpoint are consistent
- **Cross-tier sync**: Track dirty pages, flush on demand

### 8.3 Conflict Resolution
- Last-write-wins for simple cases
- Version vectors for distributed scenarios
- User-resolvable conflicts for important content

### 8.4 Full Event-Sourced Mutations

Extends `MutationLogLite` from v0.9 with full time-travel capabilities.

```python
class PageMutationFull(PageMutation):
    """Extended mutation with version tracking"""

    # What changed (extends lite version)
    previous_version: Optional[str]
    new_version: str
    delta: Optional[Dict[str, Any]]  # For partial updates

    # Extended actor info
    actor_id: Optional[str]  # tool_name, model_id, etc.

class MutationLog(MutationLogLite):
    """
    Full mutation log with time-travel.
    Extends MutationLogLite from v0.9.
    """

    async def replay_to(self, page_id: str, timestamp: datetime) -> MemoryPage:
        """Reconstruct page state at any point in time"""

    async def get_mutations_by_actor(self, actor_id: str) -> List[PageMutationFull]:
        """All mutations by a specific tool/model"""

    async def get_context_diff(self, turn_a: int, turn_b: int) -> ContextDiff:
        """What changed in context between turns?"""
```

This enables:
- **Time travel**: Reconstruct page state at any point
- **Deterministic replay**: Debug agent behavior exactly
- **Audit trail**: Who changed what and why
- **Undo/redo**: Revert to previous states

### 8.5 Deliverables
- [ ] `PageVersion` model
- [ ] `PageMutation` and `MutationLog`
- [ ] Version history tracking
- [ ] Dirty page tracking and flush
- [ ] Conflict detection and resolution
- [ ] Replay/time-travel utilities

---

## Phase 9: Tools as Memory Writers (v0.16)

Tools are first-class memory citizens - they create, read, and mutate pages.

### 9.1 Tool Memory Interface
Every tool can interact with memory.

```python
class ToolMemoryContext:
    """Passed to tools during execution"""
    memory: MemoryManager

    # Read
    async def get_page(self, page_id: str) -> MemoryPage
    async def search_pages(self, query: str, modality: Optional[str] = None) -> List[MemoryPage]

    # Write
    async def create_page(self, content: Any, modality: str, **metadata) -> MemoryPage
    async def update_page(self, page_id: str, content: Any) -> MemoryPage
    async def annotate_page(self, page_id: str, annotation: str) -> None

    # Reference
    async def link_pages(self, source_id: str, target_id: str, relation: str) -> None

class ToolResult(BaseModel):
    """Extended tool result with memory effects"""
    value: Any
    pages_created: List[str] = []
    pages_read: List[str] = []
    pages_modified: List[str] = []
```

### 9.2 Tool-Created Pages
Tools naturally produce memory artifacts.

```python
# Calculator tool creates structured result page
@tool
async def calculate(expression: str, ctx: ToolMemoryContext) -> ToolResult:
    result = eval_safe(expression)
    page = await ctx.create_page(
        content={"expression": expression, "result": result},
        modality="structured",
        importance=0.8,  # Computed values are important
        tags=["calculation", "numeric"]
    )
    return ToolResult(value=result, pages_created=[page.page_id])

# Image analysis tool creates caption page
@tool
async def analyze_image(image_page_id: str, ctx: ToolMemoryContext) -> ToolResult:
    image = await ctx.get_page(image_page_id)
    caption = await vision_model.caption(image.content)

    # Create L2 representation
    caption_page = await ctx.create_page(
        content=caption,
        modality="text",
        parent_page_id=image_page_id,
        compression_level=2  # This IS a compressed representation
    )

    # Link as representation
    await ctx.link_pages(image_page_id, caption_page.page_id, "has_caption")

    return ToolResult(
        value=caption,
        pages_read=[image_page_id],
        pages_created=[caption_page.page_id]
    )
```

### 9.3 Tool Learning from Memory
Procedural memory learns from tool-created pages.

```python
class ProceduralMemoryIntegration:
    """Connect tool execution to procedural memory"""

    async def on_tool_complete(self, tool_name: str, result: ToolResult):
        # Record what pages the tool touched
        trace = ToolTrace(
            tool_name=tool_name,
            pages_read=result.pages_read,
            pages_created=result.pages_created,
            pages_modified=result.pages_modified,
            outcome="success" if result.value else "failure"
        )
        await self.procedural_memory.record(trace)

    async def suggest_pages_for_tool(self, tool_name: str) -> List[str]:
        """Based on past successful invocations, suggest relevant pages"""
        patterns = await self.procedural_memory.get_patterns(tool_name)
        return patterns.commonly_read_pages
```

### 9.4 Deliverables
- [ ] `ToolMemoryContext` interface
- [ ] Extended `ToolResult` with memory effects
- [ ] Tool decorators for memory-aware tools
- [ ] Procedural memory integration
- [ ] Page lineage tracking (tool → page relationships)

---

## Phase 10: Advanced Features (v1.0+)

### 10.1 Shared Memory
Pages shared across sessions/users.

```python
class SharedMemoryRegion:
    region_id: str
    scope: Literal["session", "user", "sandbox", "global"]
    pages: List[str]
    access_control: AccessPolicy
```

### 10.2 Memory-Mapped Files
Direct artifact access without full loading.

```python
class MemoryMappedArtifact:
    async def read_range(self, offset: int, length: int) -> bytes
    async def get_slice(self, start: int, end: int) -> MemoryPage
```

### 10.3 Copy-on-Write
Efficient forking of memory state.

```python
async def fork_memory_state(source_session: str, target_session: str) -> None:
    """Create new session with COW references to source pages"""
```

### 10.4 Garbage Collection
Automatic cleanup of unreferenced pages.

```python
class GarbageCollector:
    async def collect(self) -> GCStats
    async def mark_reachable(self, root_sessions: List[str]) -> Set[str]
    async def sweep_unreachable(self, reachable: Set[str]) -> int
```

### 10.5 Memory Isolation
Process-like isolation between contexts.

```python
class MemoryNamespace:
    namespace_id: str
    page_table: PageTable
    working_set: WorkingSetManager
    # Isolated view of memory
```

### 10.6 NUMA Awareness (Locality Optimization)
For distributed deployments with multiple storage backends.

```python
class LocalityManager:
    """
    Critical when artifacts live in S3 or across regions.
    Penalize cross-region page faults, bias eviction away from expensive fetches.
    """
    regions: Dict[str, StorageRegion]

    async def get_affinity(self, page_id: str) -> Literal["local", "remote", "shared"]:
        """Determine where page lives relative to current compute"""

    async def estimate_fault_cost(self, page_id: str) -> FaultCost:
        """Latency + bandwidth + monetary cost to fetch"""

    async def suggest_replication(self, page_id: str) -> Optional[str]:
        """Should we replicate this page closer?"""

class FaultCost(BaseModel):
    latency_ms: int
    bandwidth_bytes: int
    monetary_cost_usd: float
    tier: str

class EvictionPolicyWithLocality(EvictionPolicy):
    """Factor in fetch cost when deciding what to evict"""

    async def select_victims(self, pages, required_tokens):
        # Prefer evicting pages that are cheap to re-fetch
        # Keep expensive remote pages longer
        scored = [
            (p, self.eviction_score(p) * (1 / await self.locality.estimate_fault_cost(p.page_id).latency_ms))
            for p in pages
        ]
        return sorted(scored, key=lambda x: x[1])[:required_tokens]
```

This becomes critical when:
- Different MCP servers have different storage backends
- Artifacts live in S3 (high latency)
- Multi-region deployments
- Hybrid cloud/local setups

---

## Chat Completions VM Protocol

The VM layer maps cleanly onto standard Chat Completions API - no custom extensions required.

### Core Insight

| VM Concept | Chat Completions Mapping |
|------------|--------------------------|
| Working set | Messages included in `messages[]` |
| Page table | `<VM:MANIFEST_JSON>` in developer message |
| Page fault | `page_fault` tool call |
| Page discovery | `search_pages` tool call |
| Page load | `role:"tool"` response with `tool_call_id` |
| Context window | What you choose to include |

### Design Principles

1. **VM:MANIFEST_JSON** is machine-readable (strict JSON) for deterministic parsing, diffing, replay
2. **VM:CONTEXT** is human-readable (compact transcript) for model comprehension
3. **Hints are NOT evidence** - only VM:CONTEXT and tool-returned content are evidence
4. **Canonical envelope** for all tool results enables consistent downstream processing

---

### Mode 1: Passive VM (No Tools)

VM as developer metadata only - runtime handles paging transparently.

```json
{
  "model": "gpt-4o",
  "messages": [
    {
      "role": "developer",
      "content": "You are a helpful assistant.\n\n<VM:MANIFEST_JSON>\n{\"session_id\":\"sess_123\",\"working_set\":[{\"page_id\":\"msg_211\",\"modality\":\"text\",\"level\":0,\"tokens_est\":320},{\"page_id\":\"msg_212\",\"modality\":\"text\",\"level\":0,\"tokens_est\":180}],\"available_pages\":[{\"page_id\":\"summary_seg_01\",\"modality\":\"text\",\"tier\":\"L2\",\"hint\":\"early discussion\"}],\"policies\":{\"faults_allowed\":false}}\n</VM:MANIFEST_JSON>\n\n<VM:CONTEXT>\nU (msg_211): \"What about the paging model?\"\nA (msg_212): \"Here's how it works...\"\n</VM:CONTEXT>"
    },
    { "role": "user", "content": "What did we discuss earlier?" },
    { "role": "assistant", "content": "Based on our conversation..." }
  ]
}
```

The model sees the manifest but cannot request pages - your runtime repacks context each turn.

---

### Mode 2: Active VM (Tool-Based Paging)

Model can explicitly request pages via `page_fault` and discover pages via `search_pages`.

#### VM Tool Definitions

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "page_fault",
        "description": "Load a memory page into context at specified compression level. Use when you need content from a known page_id.",
        "parameters": {
          "type": "object",
          "properties": {
            "page_id": {
              "type": "string",
              "description": "ID of the page to load"
            },
            "target_level": {
              "type": "integer",
              "minimum": 0,
              "maximum": 3,
              "default": 2,
              "description": "0=full, 1=reduced, 2=abstract/summary, 3=reference only"
            }
          },
          "required": ["page_id"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "search_pages",
        "description": "Search for pages matching a query. Use when you need to find relevant pages but don't know their IDs.",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "Search query (semantic or keyword)"
            },
            "modality": {
              "type": "string",
              "enum": ["text", "image", "audio", "video", "structured"],
              "description": "Filter by content type"
            },
            "limit": {
              "type": "integer",
              "default": 5,
              "description": "Maximum results to return"
            }
          },
          "required": ["query"]
        }
      }
    }
  ]
}
```

#### Request with Manifest

```json
{
  "model": "gpt-4o",
  "tools": [...],
  "messages": [
    {
      "role": "developer",
      "content": "<VM:RULES>\nYou have access to a virtual memory system.\n- VM:CONTEXT contains your currently mapped memory - this is EVIDENCE\n- VM:MANIFEST_JSON lists available pages - hints are for DISCOVERY ONLY, not evidence\n- To load content, call page_fault(page_id, target_level)\n- To find pages, call search_pages(query)\n- Prefer level=2 (summaries) before level=0 (full) to conserve tokens\n- When citing information, reference the page_id\n- NEVER fabricate content from hints - hints tell you what EXISTS, not what it SAYS\n</VM:RULES>\n\n<VM:MANIFEST_JSON>\n{\n  \"session_id\": \"sess_456\",\n  \"working_set\": [\n    {\"page_id\": \"msg_302\", \"modality\": \"text\", \"level\": 0, \"tokens_est\": 85, \"importance\": 0.7},\n    {\"page_id\": \"summary_seg_03\", \"modality\": \"text\", \"level\": 2, \"tokens_est\": 220, \"importance\": 0.9}\n  ],\n  \"available_pages\": [\n    {\"page_id\": \"summary_seg_02\", \"modality\": \"text\", \"tier\": \"L2\", \"levels\": [2,1,0], \"hint\": \"page fault definitions\"},\n    {\"page_id\": \"msg_132\", \"modality\": \"text\", \"tier\": \"L2\", \"levels\": [0], \"hint\": \"fault handler code\"},\n    {\"page_id\": \"img_045\", \"modality\": \"image\", \"tier\": \"L3\", \"levels\": [2,1], \"hint\": \"architecture diagram\"}\n  ],\n  \"policies\": {\n    \"faults_allowed\": true,\n    \"max_faults_per_turn\": 2,\n    \"upgrade_budget_tokens\": 4096,\n    \"prefer_levels\": [2, 1, 0]\n  }\n}\n</VM:MANIFEST_JSON>\n\n<VM:CONTEXT>\nU (msg_302): \"it should be compatible with openai format\"\nA (msg_303): \"Yes, here's how...\"\nS (summary_seg_03): \"This segment covered: VM protocol design, Chat Completions mapping, tool-based paging.\"\n</VM:CONTEXT>"
    },
    { "role": "user", "content": "Show me that architecture diagram you mentioned" }
  ]
}
```

#### Model Issues Page Fault

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "page_fault",
        "arguments": "{\"page_id\":\"img_045\",\"target_level\":2}"
      }
    }
  ]
}
```

---

### Canonical Tool Result Envelope

All VM tool results use a consistent envelope for downstream processing:

```json
{
  "role": "tool",
  "tool_call_id": "call_abc123",
  "content": "{\"page\":{\"page_id\":\"img_045\",\"modality\":\"image\",\"level\":2,\"tier\":\"L1\",\"content\":{\"caption\":\"Architecture diagram showing L0-L4 storage tiers with bidirectional arrows indicating page promotion and eviction flows.\",\"url\":\"https://artifacts.example.com/img_045.png\"},\"meta\":{\"dimensions\":[1200,800],\"mime_type\":\"image/png\",\"size_bytes\":245000,\"source_tier\":\"L3\"}},\"effects\":{\"promoted_to_working_set\":true,\"tokens_est\":95,\"evictions\":null}}"
}
```

#### Envelope Schema

```typescript
interface VMToolResult {
  page: {
    page_id: string;
    modality: "text" | "image" | "audio" | "video" | "structured";
    level: 0 | 1 | 2 | 3;
    tier: "L0" | "L1" | "L2" | "L3" | "L4";
    content: ModalityContent;  // Varies by modality
    meta: {
      mime_type?: string;
      size_bytes?: number;
      source_tier: string;
      // Modality-specific
      dimensions?: [number, number];      // image/video
      duration_seconds?: number;          // audio/video
      word_count?: number;                // text
    };
  };
  effects: {
    promoted_to_working_set: boolean;
    tokens_est: number;
    evictions?: string[];  // page_ids evicted to make room
  };
}
```

#### Content by Modality

```typescript
// Text (level 0-3)
{ text: string }

// Image
{ caption: string; url?: string; base64?: string; embedding?: number[] }

// Audio
{ transcript: string; timestamps?: Array<{time: number, text: string}>; duration_seconds: number }

// Video
{ scenes: Array<{timestamp: number, description: string, keyframe_url?: string}>; transcript?: string }

// Structured
{ data: object; schema?: string }
```

#### search_pages Result

```json
{
  "role": "tool",
  "tool_call_id": "call_search_xyz",
  "content": "{\"results\":[{\"page_id\":\"msg_087\",\"modality\":\"text\",\"tier\":\"L2\",\"levels\":[0,1,2],\"hint\":\"discussion of compression levels\",\"relevance\":0.92},{\"page_id\":\"summary_seg_01\",\"modality\":\"text\",\"tier\":\"L2\",\"levels\":[2],\"hint\":\"early architecture decisions\",\"relevance\":0.78}],\"total_available\":47}"
}
```

---

### VM:MANIFEST_JSON Schema

```typescript
interface VMManifest {
  session_id: string;
  working_set: Array<{
    page_id: string;
    modality: string;
    level: number;
    tokens_est: number;
    importance?: number;  // 0-1, affects eviction priority
  }>;
  available_pages: Array<{
    page_id: string;
    modality: string;
    tier: "L2" | "L3" | "L4";
    levels: number[];     // Available compression levels
    hint: string;         // Discovery hint - NOT evidence
  }>;
  policies: {
    faults_allowed: boolean;
    max_faults_per_turn: number;
    upgrade_budget_tokens: number;
    prefer_levels: number[];  // Preference order, e.g. [2,1,0]
  };
}
```

---

### VM:CONTEXT Format

Compact, human-readable representation of mapped pages:

```
<VM:CONTEXT>
U (msg_301): "User message text here"
A (msg_302): "Assistant response here"
T (tool_result_045): {"calculator": {"expression": "2+2", "result": 4}}
S (summary_seg_02): "Key points: 1) VM maps to Chat Completions, 2) page_fault is a tool..."
I (img_045): [IMAGE: architecture diagram showing storage tiers, 1200x800]
D (audio_012): [AUDIO: 5:42 duration, transcript: "So the key insight is..."]
V (video_007): [VIDEO: 12:30 duration, 8 scenes, topic: "system walkthrough"]
</VM:CONTEXT>
```

Prefixes: `U`=user, `A`=assistant, `T`=tool, `S`=summary, `I`=image, `D`=audio, `V`=video

---

### Strict Mode Developer Prompt

For agents that must never fabricate context. Available at `memory/vm_prompts.py:VM_STRICT_PROMPT`.

```
You are operating under STRICT Virtual Memory grounding rules.

Your ONLY valid sources of information are:
1) The content inside <VM:CONTEXT> (the currently mapped working set), and
2) The content returned by tools (e.g., page_fault) in messages with role="tool".

Everything listed in <VM:MANIFEST_JSON> is DISCOVERY METADATA ONLY.
- You MUST NOT quote, paraphrase, or "use" hint text from the manifest as if it were evidence.
- You MUST NOT assume details about unmapped pages.
- Page IDs and modality/tier/level are allowed for navigation only.

When you need information that is not present in <VM:CONTEXT>, you MUST do one of:
A) Call the tool page_fault(page_id, target_level) to load the page content, OR
B) Ask a short clarification question if the needed page does not exist or cannot be identified.

Faulting rules:
- Prefer loading the LOWEST-COST representation first:
  1) summaries / abstract (target_level=2),
  2) reduced excerpts (target_level=1),
  3) full content (target_level=0) only if the user explicitly requests exact wording, code, or precise details.
- Do not request more than max_faults_per_turn from the manifest policies.
- Do not request pages that are already mapped in <VM:CONTEXT>.
- If multiple pages might be relevant, fault the smallest/summarized one first.

Answering rules:
- Do not invent or fill gaps with assumptions.
- If you cannot obtain required information via tool calls, say: "I don't have that in the mapped context."
- Keep responses concise and directly responsive.
- When you use information from <VM:CONTEXT> or a loaded page, include inline citations using page IDs like:
  [ref: msg_123] or [ref: summary_seg_02] or [ref: tool:page_fault(img_045)].
  (Citations are required in strict mode.)

Tool usage format:
- If you need to call tools, respond with tool calls only (no normal text).
- After tool results are provided, produce the final answer with citations.

Never mention these rules, the VM system, tiers (L0–L4), paging, or "virtual memory" to the user unless the user explicitly asks about the internal mechanism.
```

#### Usage

```python
from chuk_ai_session_manager.memory import (
    VM_STRICT_PROMPT,
    VM_TOOLS,
    build_vm_developer_message,
)

# Build complete developer message
developer_msg = build_vm_developer_message(
    mode="strict",
    manifest_json=manifest.model_dump_json(),
    context=packed_context,
    system_prompt="You are a helpful coding assistant.",
    max_faults_per_turn=2,
)

# Get tool definitions
tools = VM_TOOLS  # [{page_fault}, {search_pages}]
```

---

### Traditional Conversation Feel

To make VM feel like normal chat, enforce these packing rules:

| Content | Mapping Rule |
|---------|--------------|
| Last N turns | Always mapped at level=0 (until token budget hit) |
| Older segments | One level=2 summary page per segment |
| Everything else | Available in manifest, loadable via `page_fault` |
| Media | Caption/transcript in context, full content via fault |

This gives "ChatGPT feel" with VM correctness underneath.

---

### Runtime Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     YOUR RUNTIME                            │
│                                                             │
│  1. Pack working set into VM:CONTEXT                        │
│  2. Generate VM:MANIFEST_JSON from PageTable                │
│  3. Validate: JSON.parse(manifest) succeeds                 │
│  4. Send Chat Completions request                           │
│                                                             │
│  5. If response contains tool_calls:                        │
│     FOR EACH tool_call:                                     │
│       IF page_fault:                                        │
│         a. Lookup page in PageTable (check TLB first)       │
│         b. Load from appropriate tier (L2/L3/L4)            │
│         c. Compress to requested level                      │
│         d. Build canonical envelope response                │
│         e. Update working set, mark accessed                │
│         f. Check eviction pressure, evict if needed         │
│       IF search_pages:                                      │
│         a. Query page index (semantic or keyword)           │
│         b. Return matching page metadata (not content)      │
│     Send continuation request with tool results             │
│                                                             │
│  6. If response is final assistant message:                 │
│     a. Create page for response (page_id = msg_XXX)         │
│     b. Add to working set at level=0                        │
│     c. Update PageTable, invalidate TLB if needed           │
│     d. Log: fault_count, tokens_used, evictions             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### Compatibility

This protocol works with:
- OpenAI Chat Completions API
- Anthropic Messages API (with tool_use)
- Google Gemini API
- Any Chat Completions-compatible endpoint
- Local models via OpenAI-compatible servers (vLLM, Ollama, etc.)

No vendor lock-in, no custom extensions.

---

## Integration Points

### With Existing Session Manager

```python
class SessionManager:
    memory_manager: MemoryManager  # NEW

    async def user_says(self, content: Union[str, bytes, MemoryPage], ...):
        # Create page for content
        page = await self.memory_manager.create_page(content, modality="text")
        # Add to working set
        await self.memory_manager.add_to_working_set(page)
        # Existing event tracking
        await self._add_message_event(...)

    async def attach_media(self, data: bytes, mime: str, ...):
        # Create media page
        page = await self.memory_manager.create_page(data, modality=detect_modality(mime))
        # Compress for context
        compressed = await self.memory_manager.compress(page, level=1)
        # Store full version
        await self.memory_manager.store(page, tier="L3")
```

### With Infinite Conversation

```python
class InfiniteConversationManager:
    async def _create_new_segment(self):
        # Existing segment creation
        new_session = await self._create_child_session()

        # NEW: Page out old segment
        old_pages = await self.memory_manager.get_pages_for_session(self.session_id)
        for page in old_pages:
            await self.memory_manager.evict(page, target_tier="L2")

        # NEW: Create summary page
        summary_page = await self.memory_manager.compress_session(self.session_id, level=2)
        await self.memory_manager.add_to_working_set(summary_page)
```

### With chuk-artifacts

```python
# Configuration
ARTIFACT_PROVIDER=vfs-s3  # For L4 cold storage
ARTIFACT_MEMORY_PROVIDER=vfs-memory  # For L3 hot artifacts

# Page storage uses artifacts
page_artifact_id = await artifacts_store.store(
    data=page.serialize(),
    mime="application/x-memory-page",
    scope=StorageScope.SESSION,
    metadata={
        "page_id": page.page_id,
        "modality": page.modality,
        "compression_level": page.compression_level,
    }
)
```

---

## Configuration

```python
class MemoryConfig:
    # Tier sizes
    l0_max_tokens: int = 128_000  # Context window
    l1_max_pages: int = 100       # Hot cache
    l2_max_bytes: int = 100_000_000  # 100MB session storage

    # Eviction
    eviction_policy: str = "importance_weighted_lru"
    eviction_threshold: float = 0.8  # Trigger at 80% capacity

    # Compression
    auto_compress_after_seconds: int = 300  # 5 minutes
    default_compression_level: int = 1

    # Prefetch
    prefetch_enabled: bool = True
    prefetch_depth: int = 2  # Segments ahead

    # Storage
    artifacts_provider: str = "vfs-filesystem"
    cold_storage_provider: str = "vfs-s3"

    # Per-modality settings
    image_max_inline_size: int = 1_000_000  # 1MB
    audio_auto_transcribe: bool = True
    video_extract_keyframes: bool = True
```

---

## Success Metrics

### Core Performance Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Fault rate | < 5% | Percentage of accesses requiring tier promotion |
| Compression ratio | > 10:1 | Average reduction from L0 to L2 |
| Context utilization | > 90% | How well we fill the context window |
| Prefetch hit rate | > 60% | Prefetched pages actually used |
| P95 fault latency | < 500ms | Time to resolve page fault |
| Memory overhead | < 10% | Metadata vs. content size |

### User Experience Metrics (Most Important)

These tell you whether the system "feels good" to users:

| Metric | Target | Description |
|--------|--------|-------------|
| **Recall success rate** | > 95% | When user asks "what did we say about X?", answer correctly without correction |
| **Thrash index** | < 0.5 | Faults per turn *after the first fault* in same topic window |
| **Effective tokens delivered** | > 60% | Tokens in context that are actually referenced in the answer |

```python
class UserExperienceMetrics:
    """
    Metrics that correlate with user satisfaction.
    """

    def recall_success_rate(self, session_id: str) -> float:
        """
        Track: user asks recall question → assistant answers → user confirms/corrects
        Success = no correction needed
        """
        recall_attempts = self.get_recall_attempts(session_id)
        successes = sum(1 for r in recall_attempts if not r.user_corrected)
        return successes / len(recall_attempts) if recall_attempts else 1.0

    def thrash_index(self, session_id: str, window_turns: int = 5) -> float:
        """
        Faults after first fault in a topic window.
        Low = stable working set. High = constantly missing what we need.

        Thrash index = (total_faults - unique_first_faults) / turns
        """
        faults = self.get_faults_in_window(session_id, window_turns)
        first_faults = set()
        thrash_faults = 0
        for fault in faults:
            topic = self.get_topic(fault.page_id)
            if topic not in first_faults:
                first_faults.add(topic)
            else:
                thrash_faults += 1
        return thrash_faults / window_turns

    def effective_tokens_ratio(self, turn_id: str) -> float:
        """
        What fraction of context tokens actually contributed to the answer?
        Track by seeing which page_ids are cited or clearly used.
        """
        context_tokens = self.get_context_tokens(turn_id)
        referenced_tokens = self.get_referenced_tokens(turn_id)
        return referenced_tokens / context_tokens if context_tokens else 0.0
```

### Fault Reason Breakdown

Track *why* faults happen to tune policies:

| Reason | Expected % | Red Flag If |
|--------|------------|-------------|
| `user_requested_recall` | 40-60% | < 20% (model faulting too speculatively) |
| `resolve_reference` | 20-40% | > 60% (poor working set selection) |
| `tool_prereq` | 10-20% | > 40% (tools not getting needed pages) |
| `speculative` | < 10% | > 20% (wasting budget on guesses) |

---

## Timeline Overview

| Phase | Version | Focus |
|-------|---------|-------|
| 1 | v0.8 | Foundations - MemoryPage, PageTable, TLB |
| 2 | v0.9 | Working Set Management |
| 3 | v0.10 | Page Faults & Loading (including streaming) |
| 4 | v0.11 | Eviction & Compression |
| 5 | v0.12 | Multi-Modal Integration + Memory ABI |
| 6 | v0.13 | Prefetch & Prediction |
| 7 | v0.14 | Memory Pressure & Health |
| 8 | v0.15 | Coherency, Versioning & Event Sourcing |
| 9 | v0.16 | Tools as Memory Writers |
| 10 | v1.0+ | Advanced Features (Shared Memory, COW, GC, NUMA) |

---

## Minimum Viable v0.8 Scope

What to ship first to prove VM works. Text-only, single-session, synchronous faults.

### Ship First (v0.8)

```
memory/
├── models.py          # MemoryPage, PageTableEntry, CompressionLevel
├── page_table.py      # PageTable with dirty tracking
├── tlb.py             # PageTLB (512 entries, LRU)
├── working_set.py     # WorkingSetManager + TokenBudget
├── context_packer.py  # Pack working set → VM:CONTEXT (text only)
├── manifest.py        # Generate VM:MANIFEST_JSON from PageTable
├── fault_handler.py   # Synchronous PageFaultHandler
└── artifacts_bridge.py # Store/load pages via chuk-artifacts
```

#### Core Models (v0.8)

```python
class MemoryPage(BaseModel):
    page_id: str
    modality: Literal["text"] = "text"  # v0.8: text only
    storage_tier: Literal["L0", "L1", "L2", "L3"]
    compression_level: int = 0

    # Page type from day one (critical for eviction/compression)
    page_type: Literal["transcript", "summary", "artifact", "claim"] = "transcript"

    # Provenance from day one (linked representations)
    provenance: List[str] = []  # page_ids this derives from
    represents: Optional[str] = None  # page_id this is a compressed version of

    content: Optional[str] = None
    artifact_id: Optional[str] = None
    size_bytes: int
    tokens_est: int
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    importance: float = 0.5
    dirty: bool = False

class PageTableEntry(BaseModel):
    page_id: str
    tier: str
    page_type: str  # NEW: for eviction decisions
    artifact_id: Optional[str]
    compression_level: int
    dirty: bool = False
    last_accessed: datetime
    access_count: int = 0
```

#### Tools (v0.8)

```python
# Just these two tools
page_fault(page_id: str, target_level: int = 2) -> VMToolResult
search_pages(query: str, limit: int = 5) -> SearchResult
```

#### Metrics to Track (v0.8)

```python
class VMMetrics:
    faults_total: int
    faults_per_turn: float
    tlb_hits: int
    tlb_misses: int
    evictions_total: int
    tokens_in_working_set: int
    tokens_available: int
    pages_by_tier: Dict[str, int]
```

#### Integration (v0.8)

```python
class SessionManager:
    # Add these
    memory: MemoryManager

    async def user_says(self, content: str, ...):
        page = self.memory.create_page(content, modality="text")
        await self.memory.add_to_working_set(page)
        # ... existing logic

    def get_vm_developer_message(self) -> str:
        """Generate the developer message with VM:RULES, VM:MANIFEST_JSON, VM:CONTEXT"""
```

### Defer to Later

| Feature | Reason to Defer |
|---------|-----------------|
| StreamingFaultHandler | Hard, model-dependent, not needed to prove concept |
| Multi-modal (image/audio/video) | Start with text, add modalities incrementally |
| Shared memory regions | Complex access control, not needed for single-session |
| NUMA / locality | Only matters at scale with distributed storage |
| Copy-on-write | Optimization, not core functionality |
| Garbage collection | Can manually clean up initially |
| Full event sourcing | Lite version in v0.9, full time-travel in v0.15 |
| ML-based prefetch | Simple heuristics in v0.9, ML prediction in v0.13 |

### v0.8 Success Criteria

| Metric | Target |
|--------|--------|
| Tests passing | 100% |
| Fault handling works | Model can page_fault, get content |
| TLB hit rate | > 80% for repeated accesses |
| Manifest generation | Valid JSON, parseable |
| Context packing | Stays under token budget |
| Integration | Works with existing SessionManager |

### v0.8 Non-Goals

- Perfect eviction policy (LRU is fine)
- Production-ready compression (stub summarizer OK)
- Multi-modal anything
- Distributed/shared state
- Sub-100ms fault latency

---

## Open Questions

### Resolved
- ~~**Real-time streaming**: How to page in/out during streaming responses?~~ → `DeferredPage` with speculative placeholders (Phase 3)
- ~~**Cross-session sharing**: How to handle shared pages across sessions?~~ → `SharedMemoryRegion` with scope-based access control (Phase 10)

### Active
1. **Embedding storage**: Store vectors in artifacts or separate vector DB?
   - Option A: Artifacts (unified, simpler)
   - Option B: Dedicated vector DB (faster similarity search)
   - Option C: Hybrid (embeddings in vector DB, linked to artifact pages)

2. **Compression callbacks**: Use LLM for summarization or local models?
   - LLM: Higher quality, higher latency, cost
   - Local: Fast, cheap, lower quality
   - Hybrid: Local for L1, LLM for L2+

3. **Cost optimization**: Balance storage vs. compute (recompression)?
   - Store all compression levels (storage heavy)
   - Recompute on demand (compute heavy)
   - Adaptive based on access patterns

4. **Multi-model memory sharing**: How do different models share memory?
   - Same Memory ABI, different token budgets?
   - Model-specific compression preferences?
   - Handoff protocol between models?

5. **Memory as capability**: Should tools declare memory requirements?
   - "This tool needs 10K tokens of context"
   - "This tool creates large pages"
   - Memory budgeting in tool selection

---

## North Star Acceptance Test

One direct acceptance test that proves the whole system works:

> **A 200+ turn conversation where the user repeatedly says "as we decided earlier…" across 5 topics, and the assistant reliably recalls the *right* decision with page_id citations, while keeping L0 under budget and faulting <2 pages per turn.**

This test instantly reveals whether working-set logic, pinning, anti-thrash, and claim extraction are doing their job.

### Test Scenario

```python
async def test_north_star_vm_acceptance():
    """
    200+ turn conversation across 5 topics with reliable recall.
    """
    sm = SessionManager(
        system_prompt="You are a project planning assistant.",
        vm_config=VMConfig(
            max_l0_tokens=32_000,  # Tight budget to force paging
            faults_allowed=True,
            max_faults_per_turn=2,
        )
    )

    topics = [
        "database_choice",      # "Let's use PostgreSQL"
        "api_framework",        # "We'll go with FastAPI"
        "frontend_stack",       # "React with TypeScript"
        "deployment_strategy",  # "Kubernetes on GCP"
        "testing_approach",     # "Pytest with 80% coverage"
    ]

    # Phase 1: Make decisions (turns 1-50)
    for i, topic in enumerate(topics):
        await sm.user_says(f"For {topic}, what should we use?")
        await sm.ai_responds(f"I recommend X for {topic}...")
        await sm.user_says(f"Agreed, let's go with X for {topic}")
        # This should create a claim page

    # Phase 2: Extended discussion (turns 51-150)
    # Talk about unrelated things to push decisions out of L0
    for i in range(100):
        await sm.user_says(f"Tell me about implementation detail {i}")
        await sm.ai_responds(f"Here's detail {i}...")

    # Phase 3: Recall challenges (turns 151-200)
    recall_successes = 0
    total_faults = 0

    for topic in topics:
        await sm.user_says(f"As we decided earlier, what's our {topic}?")
        response = await sm.get_last_response()

        # Check: correct recall with citation
        assert f"[ref: claim_{topic}" in response.content or correct_decision in response.content
        recall_successes += 1

        # Check: fault count for this turn
        turn_faults = sm.memory.get_faults_this_turn()
        assert turn_faults <= 2, f"Too many faults: {turn_faults}"
        total_faults += turn_faults

    # Assertions
    assert recall_successes == 5, "Should recall all 5 decisions"
    assert total_faults <= 10, f"Should fault <= 2 per recall turn, got {total_faults}"
    assert sm.memory.get_l0_tokens() <= 32_000, "L0 budget exceeded"

    # Thrash index should be low
    thrash = sm.memory.metrics.thrash_index()
    assert thrash < 0.5, f"Thrash index too high: {thrash}"

    # Recall success rate should be high
    recall_rate = sm.memory.metrics.recall_success_rate()
    assert recall_rate >= 0.95, f"Recall rate too low: {recall_rate}"
```

### What This Tests

| Capability | How It's Tested |
|------------|-----------------|
| Claim extraction | Decisions become token-cheap claim pages |
| Pinning | Claims stay accessible despite 100 turns of other content |
| Anti-thrash | Same topics don't repeatedly fault |
| Working set | L0 stays under budget across 200 turns |
| Fault budget | Never exceeds 2 faults per turn |
| Provenance | Citations link back to source pages |
| UX metrics | Recall success and thrash index within targets |

If this test passes, the VM system "feels good" in production.

---

## References

- OS Virtual Memory: Page tables, TLB, working sets, page replacement algorithms
- chuk-artifacts: Blob storage with VFS abstraction
- chuk-sessions: Session state management
- chuk-ai-session-manager: Conversation tracking, infinite context
