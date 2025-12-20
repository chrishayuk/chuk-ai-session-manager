# Fault Handling

When a model needs content that isn't in the current context (L0), it issues a "page fault" to load the content. The fault handler retrieves the page, optionally compresses it, and returns it in a canonical format.

## PageFaultHandler

Handles `page_fault` tool calls from models.

### Basic Setup

```python
from chuk_ai_session_manager.memory import (
    PageFaultHandler, PageTable, MemoryPage,
    Modality, StorageTier, CompressionLevel,
)

# Create page table with some pages
page_table = PageTable()

# Register pages at various tiers
pages = {
    "msg_001": MemoryPage(
        page_id="msg_001",
        modality=Modality.TEXT,
        content="Full content of the message...",
        storage_tier=StorageTier.L2,
    ),
    "img_001": MemoryPage(
        page_id="img_001",
        modality=Modality.IMAGE,
        caption="Architecture diagram",
        dimensions=(1200, 800),
        storage_tier=StorageTier.L3,
    ),
}

for page in pages.values():
    page_table.register(page)

# Create page resolver (returns content by ID)
async def get_page(page_id: str) -> MemoryPage:
    return pages.get(page_id)

# Create fault handler
handler = PageFaultHandler(
    page_table=page_table,
    get_page=get_page,
    max_faults_per_turn=3,
)
```

### Handling Faults

```python
import asyncio

async def handle_model_fault():
    # Model requests a page
    result = await handler.handle_fault(
        page_id="msg_001",
        target_level=2,  # Request abstract/summary level
    )

    if result.success:
        print(f"Loaded page from {result.source_tier}")
        print(f"Latency: {result.latency_ms:.2f}ms")
        print(f"Content: {result.page.content[:100]}...")
    else:
        print(f"Fault failed: {result.error}")

asyncio.run(handle_model_fault())
```

### FaultResult

```python
from chuk_ai_session_manager.memory import FaultResult

class FaultResult(BaseModel):
    success: bool
    page: Optional[MemoryPage] = None
    source_tier: Optional[str] = None
    latency_ms: float = 0.0
    error: Optional[str] = None
```

### Building Tool Results

After handling a fault, build the canonical tool result envelope:

```python
async def process_tool_call(tool_call):
    # Parse arguments
    args = json.loads(tool_call.function.arguments)
    page_id = args["page_id"]
    target_level = args.get("target_level", 2)

    # Handle the fault
    result = await handler.handle_fault(page_id, target_level)

    if result.success:
        # Build canonical envelope
        tool_result = handler.build_tool_result(result.page)
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result.model_dump_json(),
        }
    else:
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps({"error": result.error}),
        }
```

### VMToolResult Envelope

```python
from chuk_ai_session_manager.memory import VMToolResult, PageData, FaultEffects

class VMToolResult(BaseModel):
    page: PageData        # The loaded page content
    effects: FaultEffects  # Side effects of the fault
```

**PageData structure:**
```python
class PageData(BaseModel):
    page_id: str
    modality: str     # "text", "image", etc.
    level: int        # Compression level
    tier: str         # Current tier
    content: PageContent  # Modality-specific content
    meta: PageMeta    # Additional metadata
```

**FaultEffects structure:**
```python
class FaultEffects(BaseModel):
    promoted_to_working_set: bool = False
    tokens_est: int = 0
    evictions: Optional[List[str]] = None  # Pages evicted to make room
```

### Fault Limiting

```python
# Check if more faults are allowed
if handler.can_fault():
    result = await handler.handle_fault("msg_001")
else:
    print("Fault limit reached for this turn")

# Get current metrics
metrics = handler.get_metrics()
print(f"Faults this turn: {metrics.faults_this_turn}")
print(f"Faults remaining: {metrics.faults_remaining}")

# Reset for new turn
handler.new_turn()
```

### Content Formatting by Modality

The handler automatically formats content based on modality:

```python
# Text → TextContent
{"text": "Full message content..."}

# Image → ImageContent
{
    "caption": "Architecture diagram",
    "dimensions": [1200, 800],
    "url": "https://..."  # If available
}

# Audio → AudioContent
{
    "transcript": "The key insight is...",
    "duration_seconds": 342
}

# Video → VideoContent
{
    "scenes": [...],
    "transcript": "...",
    "duration_seconds": 720
}

# Structured → StructuredContent
{
    "data": {"result": 42},
    "schema_name": "calculator_result"
}
```

---

## PageSearchHandler

Handles `search_pages` tool calls to discover relevant pages.

### Basic Setup

```python
from chuk_ai_session_manager.memory import PageSearchHandler, AvailablePageEntry

# Create list of available pages
available_pages = [
    AvailablePageEntry(
        page_id="msg_001",
        modality="text",
        tier="L2",
        levels=[0, 1, 2],
        hint="discussion about authentication",
    ),
    AvailablePageEntry(
        page_id="msg_002",
        modality="text",
        tier="L2",
        levels=[0, 1, 2],
        hint="JWT token implementation",
    ),
    AvailablePageEntry(
        page_id="img_001",
        modality="image",
        tier="L3",
        levels=[1, 2],
        hint="architecture diagram",
    ),
]

# Create search handler
search_handler = PageSearchHandler(available_pages=available_pages)
```

### Searching

```python
# Search by query
result = search_handler.search(
    query="authentication",
    modality=None,  # All modalities
    limit=5,
)

print(f"Total available: {result.total_available}")
print(f"Matches found: {len(result.results)}")

for entry in result.results:
    print(f"  {entry.page_id}: {entry.modality}, relevance={entry.relevance:.2f}")
```

### SearchResult

```python
from chuk_ai_session_manager.memory import SearchResult, SearchResultEntry

class SearchResult(BaseModel):
    results: List[SearchResultEntry]
    total_available: int

class SearchResultEntry(BaseModel):
    page_id: str
    modality: str
    tier: str
    levels: List[int]
    hint: str
    relevance: float  # 0.0-1.0
```

### Filtering by Modality

```python
# Search only images
result = search_handler.search(
    query="diagram",
    modality="image",
    limit=3,
)
```

### Building Tool Results

```python
async def process_search_call(tool_call):
    args = json.loads(tool_call.function.arguments)
    query = args["query"]
    modality = args.get("modality")
    limit = args.get("limit", 5)

    result = search_handler.search(query, modality, limit)

    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": result.model_dump_json(),
    }
```

### Custom Search Implementation

You can provide a custom search function for semantic search:

```python
async def semantic_search(
    query: str,
    pages: List[AvailablePageEntry],
    limit: int,
) -> List[SearchResultEntry]:
    # Your embedding-based search logic
    embeddings = await get_embeddings([query] + [p.hint for p in pages])
    # ... compute similarities
    return sorted_results[:limit]

# Create handler with custom search
search_handler = PageSearchHandler(
    available_pages=available_pages,
    search_fn=semantic_search,  # Custom implementation
)
```

---

## Complete Fault Flow Example

```python
import json
from chuk_ai_session_manager.memory import (
    PageFaultHandler, PageSearchHandler, PageTable,
    MemoryPage, AvailablePageEntry,
    Modality, StorageTier,
)

class VMToolProcessor:
    def __init__(self, page_table, pages, available):
        self.fault_handler = PageFaultHandler(
            page_table=page_table,
            get_page=lambda pid: pages.get(pid),
            max_faults_per_turn=3,
        )
        self.search_handler = PageSearchHandler(
            available_pages=available,
        )

    async def process_tool_call(self, tool_call):
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        if name == "page_fault":
            return await self._handle_fault(tool_call.id, args)
        elif name == "search_pages":
            return await self._handle_search(tool_call.id, args)
        else:
            return self._error_response(tool_call.id, f"Unknown tool: {name}")

    async def _handle_fault(self, call_id, args):
        page_id = args["page_id"]
        target_level = args.get("target_level", 2)

        if not self.fault_handler.can_fault():
            return self._error_response(call_id, "Fault limit reached")

        result = await self.fault_handler.handle_fault(page_id, target_level)

        if result.success:
            tool_result = self.fault_handler.build_tool_result(result.page)
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "content": tool_result.model_dump_json(),
            }
        else:
            return self._error_response(call_id, result.error)

    async def _handle_search(self, call_id, args):
        query = args["query"]
        modality = args.get("modality")
        limit = args.get("limit", 5)

        result = self.search_handler.search(query, modality, limit)
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": result.model_dump_json(),
        }

    def _error_response(self, call_id, error):
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps({"error": error}),
        }

    def new_turn(self):
        """Reset per-turn counters."""
        self.fault_handler.new_turn()
```

---

## Error Handling

### Page Not Found

```python
result = await handler.handle_fault("nonexistent_page")
if not result.success:
    print(f"Error: {result.error}")  # "Page not found: nonexistent_page"
```

### Fault Limit Exceeded

```python
if not handler.can_fault():
    # Return error to model
    return VMToolError(
        error="fault_limit_exceeded",
        message=f"Maximum {handler.max_faults_per_turn} faults per turn",
    )
```

### VMToolError

```python
from chuk_ai_session_manager.memory import VMToolError

class VMToolError(BaseModel):
    error: str
    message: str
    page_id: Optional[str] = None
```

---

## Metrics and Monitoring

```python
# Get fault metrics
metrics = handler.get_metrics()
print(f"Total faults: {metrics.faults_this_turn}")
print(f"Max allowed: {metrics.max_faults_per_turn}")
print(f"Remaining: {metrics.faults_remaining}")
print(f"Can fault: {metrics.can_fault}")

# Track over time
class FaultMonitor:
    def __init__(self):
        self.total_faults = 0
        self.total_turns = 0

    def on_turn_complete(self, handler):
        metrics = handler.get_metrics()
        self.total_faults += metrics.faults_this_turn
        self.total_turns += 1
        handler.new_turn()

    @property
    def avg_faults_per_turn(self):
        return self.total_faults / self.total_turns if self.total_turns else 0
```
