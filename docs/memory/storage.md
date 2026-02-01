# Storage and Persistence

The ArtifactsBridge provides persistent storage for memory pages, enabling pages to survive beyond the current session and be restored later.

## ArtifactsBridge

Bridges the memory system to persistent storage backends.

### Basic Usage

```python
from chuk_ai_session_manager.memory import (
    ArtifactsBridge, InMemoryBackend,
    MemoryPage, Modality, StorageTier,
)

# Create in-memory backend (for testing/development)
backend = InMemoryBackend()

# Create bridge
bridge = ArtifactsBridge(backend=backend)

# Create a page
page = MemoryPage(
    page_id="msg_001",
    modality=Modality.TEXT,
    content="Important message content",
    storage_tier=StorageTier.L3,
)

# Store page
artifact_id = await bridge.store_page(page)
print(f"Stored with artifact_id: {artifact_id}")

# Load page
loaded = await bridge.load_page(artifact_id)
print(f"Loaded: {loaded.page_id}, content: {loaded.content}")
```

### Storage Backends

#### InMemoryBackend

For testing and development:

```python
from chuk_ai_session_manager.memory import InMemoryBackend

backend = InMemoryBackend()
bridge = ArtifactsBridge(backend=backend)

# Check stats
stats = bridge.get_stats()
print(f"Backend: {stats.backend}")      # "in-memory"
print(f"Persistent: {stats.persistent}") # False
print(f"Pages stored: {stats.pages_stored}")
```

#### Custom Backend

Implement your own storage backend:

```python
from typing import Optional, Dict, Any

class CustomBackend:
    """Interface for storage backends."""

    async def store(
        self,
        page_id: str,
        data: bytes,
        metadata: Dict[str, Any],
    ) -> str:
        """Store page data, return artifact_id."""
        ...

    async def load(self, artifact_id: str) -> Optional[bytes]:
        """Load page data by artifact_id."""
        ...

    async def delete(self, artifact_id: str) -> bool:
        """Delete stored page."""
        ...

    async def exists(self, artifact_id: str) -> bool:
        """Check if artifact exists."""
        ...

    async def list_pages(self) -> List[str]:
        """List all stored artifact IDs."""
        ...

# Use custom backend
backend = CustomBackend()
bridge = ArtifactsBridge(backend=backend)
```

#### S3 Backend Example

```python
import boto3
import json

class S3Backend:
    def __init__(self, bucket: str, prefix: str = "memory/"):
        self.s3 = boto3.client('s3')
        self.bucket = bucket
        self.prefix = prefix

    async def store(self, page_id: str, data: bytes, metadata: dict) -> str:
        artifact_id = f"{self.prefix}{page_id}"
        self.s3.put_object(
            Bucket=self.bucket,
            Key=artifact_id,
            Body=data,
            Metadata={k: str(v) for k, v in metadata.items()},
        )
        return artifact_id

    async def load(self, artifact_id: str) -> Optional[bytes]:
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=artifact_id)
            return response['Body'].read()
        except self.s3.exceptions.NoSuchKey:
            return None

    async def delete(self, artifact_id: str) -> bool:
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=artifact_id)
            return True
        except:
            return False
```

### Storing Pages

```python
# Store with automatic artifact_id generation
artifact_id = await bridge.store_page(page)

# The artifact_id follows pattern: mem_{tier}_{sequence}
# Example: "mem_L3_1", "mem_L2_42"
```

### Loading Pages

```python
# Load by artifact_id
page = await bridge.load_page("mem_L3_1")
if page:
    print(f"Loaded: {page.page_id}")
    print(f"Content: {page.content}")
else:
    print("Page not found")
```

### Deleting Pages

```python
success = await bridge.delete_page("mem_L3_1")
if success:
    print("Page deleted")
```

### Listing Pages

```python
# List all stored artifact IDs
artifact_ids = await bridge.list_pages()
for aid in artifact_ids:
    print(f"Stored: {aid}")
```

---

## Checkpointing

Save and restore complete memory state at a point in time.

### Creating Checkpoints

```python
from chuk_ai_session_manager.memory import (
    ArtifactsBridge, CheckpointManifest, CheckpointEntry,
)

# Pages to checkpoint
pages_to_save = [
    MemoryPage(page_id="msg_001", modality=Modality.TEXT, content="..."),
    MemoryPage(page_id="msg_002", modality=Modality.TEXT, content="..."),
    MemoryPage(page_id="img_001", modality=Modality.IMAGE, caption="..."),
]

# Create checkpoint
checkpoint_id = await bridge.checkpoint(
    pages=pages_to_save,
    name="session_checkpoint_v1",
)
print(f"Checkpoint created: {checkpoint_id}")
```

### Restoring from Checkpoints

```python
# Restore all pages from checkpoint
restored_pages = await bridge.restore_checkpoint(checkpoint_id)
print(f"Restored {len(restored_pages)} pages")

for page in restored_pages:
    print(f"  {page.page_id}: {page.modality}")
```

### Checkpoint Manifest

```python
from chuk_ai_session_manager.memory import CheckpointManifest, CheckpointEntry

class CheckpointManifest(BaseModel):
    name: str
    created_at: datetime
    page_count: int
    pages: List[CheckpointEntry]

class CheckpointEntry(BaseModel):
    page_id: str
    artifact_id: str
    modality: str
    tier: str
```

### Listing Checkpoints

```python
# Get all checkpoints (stored with special artifact IDs)
checkpoints = await bridge.list_checkpoints()
for cp_id in checkpoints:
    manifest = await bridge.get_checkpoint_manifest(cp_id)
    print(f"{cp_id}: {manifest.name}, {manifest.page_count} pages")
```

---

## Page Metadata

When storing pages, metadata is preserved:

```python
from chuk_ai_session_manager.memory import PageMetadata

class PageMetadata(BaseModel):
    page_id: str
    modality: str
    tier: str
    compression_level: int
    size_bytes: int
    created_at: datetime
    checksum: Optional[str] = None
```

### Accessing Metadata

```python
# Get metadata without loading full page
metadata = await bridge.get_metadata(artifact_id)
if metadata:
    print(f"Page: {metadata.page_id}")
    print(f"Size: {metadata.size_bytes} bytes")
    print(f"Created: {metadata.created_at}")
```

---

## Storage Statistics

```python
from chuk_ai_session_manager.memory import StorageStats

stats = bridge.get_stats()
print(f"Backend: {stats.backend}")
print(f"Persistent: {stats.persistent}")
print(f"Pages stored: {stats.pages_stored}")
print(f"Total bytes: {stats.total_bytes}")
```

---

## Integration with Page Table

Track stored pages in the page table:

```python
from chuk_ai_session_manager.memory import PageTable, StorageTier

async def evict_to_storage(page_table, bridge, page):
    """Evict a page to persistent storage."""
    # Store in L3
    artifact_id = await bridge.store_page(page)

    # Update page table entry
    page_table.update_location(
        page_id=page.page_id,
        tier=StorageTier.L3,
        artifact_id=artifact_id,
    )

    # Mark as clean (just flushed)
    page_table.mark_clean(page.page_id)

    return artifact_id


async def fault_from_storage(page_table, bridge, page_id):
    """Load a page from storage on fault."""
    # Look up artifact_id
    entry = page_table.lookup(page_id)
    if not entry or not entry.artifact_id:
        return None

    # Load from storage
    page = await bridge.load_page(entry.artifact_id)

    if page:
        # Update access tracking
        page_table.mark_accessed(page_id)

    return page
```

---

## Complete Example

```python
import asyncio
from chuk_ai_session_manager.memory import (
    ArtifactsBridge, InMemoryBackend,
    PageTable, WorkingSetManager,
    MemoryPage, Modality, StorageTier,
)

async def main():
    # Initialize storage
    backend = InMemoryBackend()
    bridge = ArtifactsBridge(backend=backend)
    page_table = PageTable()

    # Create some pages
    pages = [
        MemoryPage(
            page_id=f"msg_{i:03d}",
            modality=Modality.TEXT,
            content=f"Message {i} content...",
            storage_tier=StorageTier.L0,
        )
        for i in range(10)
    ]

    # Register in page table
    for page in pages:
        page_table.register(page)

    # Evict older pages to storage
    for page in pages[:5]:
        artifact_id = await bridge.store_page(page)
        page_table.update_location(
            page.page_id,
            tier=StorageTier.L3,
            artifact_id=artifact_id,
        )
        print(f"Evicted {page.page_id} -> {artifact_id}")

    # Create checkpoint
    checkpoint_id = await bridge.checkpoint(
        pages=pages,
        name="session_snapshot",
    )
    print(f"\nCheckpoint: {checkpoint_id}")

    # Simulate restart - restore from checkpoint
    restored = await bridge.restore_checkpoint(checkpoint_id)
    print(f"\nRestored {len(restored)} pages")

    # Stats
    stats = bridge.get_stats()
    print(f"\nStorage: {stats.pages_stored} pages stored")

asyncio.run(main())
```

---

## Best Practices

### 1. Use Appropriate Tiers

```python
# L2: Session storage (fast, limited)
# L3: Disk storage (persistent, larger)
# L4: Cold storage (archive, slow)

if page.access_count < 2 and page.age_hours > 24:
    tier = StorageTier.L4  # Archive rarely accessed
elif page.age_hours > 1:
    tier = StorageTier.L3  # Persist older pages
else:
    tier = StorageTier.L2  # Keep recent in session
```

### 2. Checkpoint Regularly

```python
# Checkpoint at segment boundaries
async def on_segment_complete(pages):
    checkpoint_id = await bridge.checkpoint(
        pages=pages,
        name=f"segment_{datetime.utcnow().isoformat()}",
    )
    return checkpoint_id
```

### 3. Clean Up Old Checkpoints

```python
async def cleanup_old_checkpoints(bridge, max_age_days=7):
    checkpoints = await bridge.list_checkpoints()
    cutoff = datetime.utcnow() - timedelta(days=max_age_days)

    for cp_id in checkpoints:
        manifest = await bridge.get_checkpoint_manifest(cp_id)
        if manifest.created_at < cutoff:
            await bridge.delete_checkpoint(cp_id)
```

### 4. Handle Storage Failures

```python
async def safe_store(bridge, page, retries=3):
    for attempt in range(retries):
        try:
            return await bridge.store_page(page)
        except Exception as e:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
```
