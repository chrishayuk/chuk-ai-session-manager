# Data Structures

The memory system uses three core data structures for tracking pages:

1. **PageTable** - Authoritative registry of all pages and their locations
2. **PageTLB** - Fast cache for recently accessed page entries
3. **WorkingSetManager** - Tracks what's currently in L0/L1

## PageTable

The PageTable maps page IDs to their metadata (location, state, access patterns). Like an OS page table, it doesn't hold content - just metadata about where content lives.

### Basic Usage

```python
from chuk_ai_session_manager.memory import PageTable, MemoryPage, StorageTier, Modality

# Create page table
table = PageTable()

# Register a page
page = MemoryPage(
    page_id="msg_001",
    modality=Modality.TEXT,
    content="Hello world",
    storage_tier=StorageTier.L0,
)
entry = table.register(page)

# Lookup by ID
entry = table.lookup("msg_001")
if entry:
    print(f"Page {entry.page_id} is in tier {entry.tier}")

# Check if page exists
if "msg_001" in table:
    print("Page exists")

# Get page count
print(f"Total pages: {len(table)}")
```

### Tier Management

```python
from chuk_ai_session_manager.memory import StorageTier, CompressionLevel

# Get pages by tier
l0_pages = table.get_by_tier(StorageTier.L0)
l2_pages = table.get_by_tier(StorageTier.L2)

# Move a page to a different tier
table.update_location(
    page_id="msg_001",
    tier=StorageTier.L2,
    artifact_id="artifact_xyz",          # Optional: storage reference
    compression_level=CompressionLevel.REDUCED,  # Optional: new level
)

# Get working set (L0 + L1)
working_set = table.get_working_set()
```

### Dirty Tracking

Pages can be marked as "dirty" when modified, indicating they need to be flushed to storage.

```python
# Mark page as modified
table.mark_dirty("msg_001")

# Get all dirty pages
dirty_pages = table.get_dirty_pages()

# Mark as flushed/clean
table.mark_clean("msg_001")
```

### Access Tracking

```python
# Mark page as accessed (updates LRU tracking)
table.mark_accessed("msg_001")

# Get eviction candidates (oldest accessed first)
candidates = table.get_eviction_candidates(
    tier=StorageTier.L1,
    limit=10,
)
```

### Statistics

```python
stats = table.get_stats()
print(f"Total pages: {stats.total_pages}")
print(f"Dirty pages: {stats.dirty_pages}")
print(f"Working set size: {stats.working_set_size}")
print(f"Pages by tier: {stats.pages_by_tier}")
print(f"Pages by modality: {stats.pages_by_modality}")

# Get total tokens across tiers
total_tokens = table.get_total_tokens()
l0_l1_tokens = table.get_total_tokens(tiers=[StorageTier.L0, StorageTier.L1])
```

### Modality Queries

```python
from chuk_ai_session_manager.memory import Modality

# Get all text pages
text_pages = table.get_by_modality(Modality.TEXT)

# Get all image pages
image_pages = table.get_by_modality(Modality.IMAGE)
```

### Removal

```python
# Remove a page
removed_entry = table.remove("msg_001")
if removed_entry:
    print(f"Removed page from tier {removed_entry.tier}")
```

---

## PageTLB

The Translation Lookaside Buffer (TLB) is a fast cache for page table entries. It avoids the overhead of full PageTable lookups for frequently accessed pages.

### Basic Usage

```python
from chuk_ai_session_manager.memory import PageTLB, PageTableEntry, StorageTier, Modality

# Create TLB with custom size
tlb = PageTLB(max_entries=512)

# Create an entry
entry = PageTableEntry(
    page_id="msg_001",
    tier=StorageTier.L0,
    modality=Modality.TEXT,
    size_tokens=100,
)

# Insert into TLB
tlb.insert(entry)

# Lookup (returns None on miss)
result = tlb.lookup("msg_001")
if result:
    print(f"TLB hit: {result.page_id}")
else:
    print("TLB miss")
```

### Hit/Miss Tracking

```python
# Check stats
print(f"Hits: {tlb.hits}")
print(f"Misses: {tlb.misses}")
print(f"Hit rate: {tlb.hit_rate:.2%}")

# Get full stats as model
stats = tlb.get_stats()
print(f"Size: {stats.size}/{stats.max_size}")
print(f"Utilization: {stats.utilization:.2%}")

# Reset counters
tlb.reset_stats()
```

### Invalidation

When pages are modified or evicted, their TLB entries must be invalidated.

```python
# Invalidate single entry
was_present = tlb.invalidate("msg_001")

# Invalidate all entries in a tier
count = tlb.invalidate_tier(StorageTier.L1)
print(f"Invalidated {count} entries")

# Flush entire TLB
count = tlb.flush()
print(f"Flushed {count} entries")
```

### LRU Eviction

The TLB automatically evicts the least recently used entry when full.

```python
# TLB with small size for demonstration
tlb = PageTLB(max_entries=3)

# Insert 4 entries - first one will be evicted
for i in range(4):
    entry = PageTableEntry(
        page_id=f"msg_{i:03d}",
        tier=StorageTier.L0,
        modality=Modality.TEXT,
    )
    tlb.insert(entry)

# msg_000 was evicted (LRU)
assert tlb.lookup("msg_000") is None  # Miss
assert tlb.lookup("msg_003") is not None  # Hit
```

### Inspection

```python
# Get all cached entries
all_entries = tlb.get_all()

# Check if page is cached
if "msg_001" in tlb:
    print("Page is in TLB")
```

---

## TLBWithPageTable

A convenience wrapper that combines TLB with PageTable for unified lookups. It automatically checks TLB first, falls back to PageTable, and keeps TLB updated.

### Basic Usage

```python
from chuk_ai_session_manager.memory import (
    PageTable, PageTLB, TLBWithPageTable, MemoryPage, StorageTier, Modality
)

# Create components
table = PageTable()
tlb = PageTLB(max_entries=100)

# Create wrapper
combined = TLBWithPageTable(table, tlb)

# Register a page (adds to both)
page = MemoryPage(
    page_id="msg_001",
    modality=Modality.TEXT,
    content="Hello",
    storage_tier=StorageTier.L0,
)
entry = combined.register(page)

# Lookup checks TLB first, then PageTable
# Automatically populates TLB on miss
entry = combined.lookup("msg_001")
```

### Automatic TLB Population

```python
# First lookup - TLB miss, PageTable hit, TLB populated
entry = combined.lookup("msg_001")
print(f"TLB misses: {tlb.misses}")  # 1

# Second lookup - TLB hit
entry = combined.lookup("msg_001")
print(f"TLB hits: {tlb.hits}")  # 1
```

### Location Updates

```python
# Update location invalidates stale TLB entry
combined.update_location(
    page_id="msg_001",
    tier=StorageTier.L2,
)
# TLB entry invalidated, will be refreshed on next lookup
```

### Dirty Tracking

```python
# Mark dirty invalidates TLB (entry is stale)
combined.mark_dirty("msg_001")
```

### Removal

```python
# Remove from both
entry = combined.remove("msg_001")
```

### Combined Statistics

```python
stats = combined.get_stats()
print(f"PageTable total: {stats.page_table.total_pages}")
print(f"TLB size: {stats.tlb.size}")
print(f"TLB hit rate: {stats.tlb.hit_rate:.2%}")
```

---

## WorkingSetManager

Manages the working set - pages that are currently "hot" (in L0 or L1). It tracks token budgets and handles capacity constraints.

### Configuration

```python
from chuk_ai_session_manager.memory import (
    WorkingSetManager, WorkingSetConfig, TokenBudget
)

# Create with custom config
config = WorkingSetConfig(
    max_l0_tokens=128_000,      # Context window size
    max_l1_pages=100,           # Max pages in L1 cache
    eviction_threshold=0.85,    # Trigger eviction at 85%
    target_utilization=0.70,    # Target after eviction
    reserved_tokens=4000,       # Reserved for system/tools
)

# Create budget
budget = TokenBudget(total_limit=128_000, reserved=4000)

# Create manager
manager = WorkingSetManager(config=config, budget=budget)
```

### Adding Pages to L0

```python
from chuk_ai_session_manager.memory import MemoryPage, Modality, StorageTier

page = MemoryPage(
    page_id="msg_001",
    modality=Modality.TEXT,
    content="Hello world",
    storage_tier=StorageTier.L0,
    size_tokens=10,
)

# Add to L0 (context window)
success = manager.add_to_l0(page)
if success:
    print(f"Added to L0, tokens used: {manager.tokens_used}")
else:
    print("Insufficient space in L0")
```

### Adding Pages to L1

```python
# Add to L1 (hot cache, not in context)
image_page = MemoryPage(
    page_id="img_001",
    modality=Modality.IMAGE,
    caption="Diagram",
    storage_tier=StorageTier.L1,
)

success = manager.add_to_l1(image_page)
if success:
    print(f"Added to L1, L1 count: {manager.l1_count}")
```

### Promotion and Demotion

```python
# Promote from L1 to L0
success = manager.promote_to_l0(image_page)

# Demote from L0 to L1
success = manager.demote_to_l1(page)
```

### Token Budget Tracking

```python
# Check current usage
print(f"Tokens used: {manager.tokens_used}")
print(f"Tokens available: {manager.tokens_available}")
print(f"Utilization: {manager.utilization:.2%}")

# Check if more tokens can fit
can_fit = manager.can_fit(1000)

# Check if eviction is needed
if manager.needs_eviction():
    print("Context window pressure - eviction needed")
```

### Eviction

```python
# Get eviction candidates (sorted by priority, lowest first)
candidates = manager.get_eviction_candidates(
    tokens_needed=5000,
    from_tier=StorageTier.L0,
)

for page_id, score in candidates:
    print(f"  {page_id}: score={score:.2f}")

# Calculate how many tokens to free
tokens_to_free = manager.calculate_eviction_target(tokens_needed=5000)
print(f"Need to free {tokens_to_free} tokens")
```

### Importance Overrides

You can manually adjust page importance to influence eviction decisions.

```python
# Set high importance (0.0-1.0)
manager.set_importance("msg_001", importance=0.9)

# Clear override
manager.clear_importance("msg_001")
```

### Checking Page Location

```python
# Check if page is in L0
if manager.is_in_l0("msg_001"):
    print("Page is in context window")

# Check if page is in L1
if manager.is_in_l1("img_001"):
    print("Page is in cache")

# Check if in working set (L0 or L1)
if manager.is_in_working_set("msg_001"):
    print("Page is hot")
```

### Removal

```python
# Remove from L0 with budget update
success = manager.remove_from_l0("msg_001", page)

# Remove from entire working set
removed_page = manager.remove("msg_001")
```

### Accessing Pages

```python
# Get page from L1 cache (L0 pages are ID-only)
page = manager.get_page("img_001")

# Get all L0 page IDs (in order)
l0_ids = manager.get_l0_page_ids()

# Get all L1 pages
l1_pages = manager.get_l1_pages()
```

### Statistics

```python
stats = manager.get_stats()
print(f"L0 pages: {stats.l0_pages}")
print(f"L1 pages: {stats.l1_pages}")
print(f"Total pages: {stats.total_pages}")
print(f"Tokens used: {stats.tokens_used}")
print(f"Tokens available: {stats.tokens_available}")
print(f"Utilization: {stats.utilization:.2%}")
print(f"Needs eviction: {stats.needs_eviction}")
print(f"Tokens by modality: {stats.tokens_by_modality}")
```

### Clearing

```python
# Clear entire working set
manager.clear()
```

---

## Eviction Scoring

The WorkingSetManager uses a composite score for eviction decisions:

### L0 Eviction (by position)

```python
# Score = position_score * 0.5 + importance * 0.5
# Earlier position = lower score = evict first
```

### L1 Eviction (by access patterns)

```python
# recency_score = 1.0 / (1.0 + age_seconds / 3600)  # Decay over hours
# frequency_score = log1p(access_count) / 10.0       # Log scale
# importance = user override or page.importance

# Score = recency * 0.4 + frequency * 0.2 + importance * 0.4
# Lower score = evict first
```

---

## Usage Pattern: Combined Flow

```python
from chuk_ai_session_manager.memory import (
    PageTable, PageTLB, TLBWithPageTable,
    WorkingSetManager, WorkingSetConfig, TokenBudget,
    MemoryPage, Modality, StorageTier,
)

# Initialize data structures
table = PageTable()
tlb = PageTLB(max_entries=512)
combined = TLBWithPageTable(table, tlb)

config = WorkingSetConfig(max_l0_tokens=128_000)
budget = TokenBudget(total_limit=128_000, reserved=4000)
working_set = WorkingSetManager(config=config, budget=budget)

# Create and register a page
page = MemoryPage(
    page_id="msg_001",
    modality=Modality.TEXT,
    content="User message",
    storage_tier=StorageTier.L0,
)

# Register in page table (with TLB)
combined.register(page)

# Add to working set
working_set.add_to_l0(page)

# Later: lookup with TLB acceleration
entry = combined.lookup("msg_001")

# Check working set status
if working_set.needs_eviction():
    candidates = working_set.get_eviction_candidates()
    for page_id, score in candidates[:3]:
        # Evict page
        working_set.remove(page_id)
        combined.update_location(page_id, tier=StorageTier.L2)
```
