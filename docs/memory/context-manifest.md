# Context Packing and Manifest Generation

The memory system produces two key outputs for model integration:

1. **VM:CONTEXT** - Human-readable representation of mapped pages
2. **VM:MANIFEST_JSON** - Machine-readable metadata for page discovery

## ContextPacker

Packs pages into the compact VM:CONTEXT format that models can understand.

### Basic Usage

```python
from chuk_ai_session_manager.memory import (
    ContextPacker, ContextPackerConfig,
    MemoryPage, Modality, StorageTier,
)

# Create packer with default config
packer = ContextPacker()

# Create pages
pages = [
    MemoryPage(
        page_id="msg_001",
        modality=Modality.TEXT,
        content="What's the weather?",
        storage_tier=StorageTier.L0,
        metadata={"role": "user"},
    ),
    MemoryPage(
        page_id="msg_002",
        modality=Modality.TEXT,
        content="The weather is sunny.",
        storage_tier=StorageTier.L0,
        metadata={"role": "assistant"},
    ),
]

# Pack into context
packed = packer.pack(pages)
print(packed.content)
```

**Output:**
```
U (msg_001): "What's the weather?"
A (msg_002): "The weather is sunny."
```

### Configuration

```python
from chuk_ai_session_manager.memory import ContextPackerConfig

config = ContextPackerConfig(
    max_line_length=500,      # Truncate long content
    include_page_ids=True,    # Include page IDs in output
    quote_text=True,          # Wrap text in quotes
    max_tokens=None,          # Token limit (optional)
)

packer = ContextPacker(config=config)
```

### VM:CONTEXT Format

The format uses single-character prefixes for compact representation:

| Prefix | Meaning | Example |
|--------|---------|---------|
| `U` | User message | `U (msg_001): "Hello"` |
| `A` | Assistant response | `A (msg_002): "Hi there"` |
| `T` | Tool result | `T (tool_001): {"result": 42}` |
| `S` | Summary | `S (summary_01): "Key points..."` |
| `I` | Image | `I (img_001): [IMAGE: caption, WxH]` |
| `D` | Audio | `D (audio_001): [AUDIO: 5:30, transcript]` |
| `V` | Video | `V (video_001): [VIDEO: 12:00, 5 scenes]` |
| `J` | Structured data | `J (data_001): {...}` |

### Multi-Modal Packing

```python
# Image page
image_page = MemoryPage(
    page_id="img_001",
    modality=Modality.IMAGE,
    caption="Architecture diagram",
    dimensions=(1200, 800),
    storage_tier=StorageTier.L0,
)

# Audio page
audio_page = MemoryPage(
    page_id="audio_001",
    modality=Modality.AUDIO,
    transcript="The key insight is...",
    duration_seconds=342,  # 5:42
    storage_tier=StorageTier.L0,
)

# Pack together
packed = packer.pack([image_page, audio_page])
print(packed.content)
```

**Output:**
```
I (img_001): [IMAGE: Architecture diagram, 1200x800]
D (audio_001): [AUDIO: 5:42 duration, transcript: "The key insight is..."]
```

### With XML Wrapper

```python
# Get content wrapped in <VM:CONTEXT> tags
wrapped = packer.pack_with_wrapper(pages)
print(wrapped)
```

**Output:**
```xml
<VM:CONTEXT>
U (msg_001): "What's the weather?"
A (msg_002): "The weather is sunny."
</VM:CONTEXT>
```

### Token Estimation

```python
packed = packer.pack(pages)
print(f"Estimated tokens: {packed.tokens_est}")
```

### Role Detection

The packer automatically detects message roles from page metadata:

```python
# Role from metadata
page = MemoryPage(
    page_id="msg_001",
    modality=Modality.TEXT,
    content="Hello",
    metadata={"role": "user"},  # Will use "U" prefix
)

# Summary pages use "S" prefix
summary = MemoryPage(
    page_id="summary_001",
    modality=Modality.TEXT,
    content="Key points from discussion...",
    metadata={"type": "summary"},  # Will use "S" prefix
)
```

---

## ManifestBuilder

Generates VM:MANIFEST_JSON - the machine-readable metadata that tells models what pages are available.

### Basic Usage

```python
from chuk_ai_session_manager.memory import (
    ManifestBuilder, ManifestPolicies,
    MemoryPage, Modality, StorageTier,
)

# Create builder
builder = ManifestBuilder(
    session_id="session_abc123",
    policies=ManifestPolicies(
        max_faults_per_turn=3,
        upgrade_budget_tokens=4096,
    ),
)

# Add working set pages (in context)
working_page = MemoryPage(
    page_id="msg_001",
    modality=Modality.TEXT,
    content="Hello",
    storage_tier=StorageTier.L0,
    size_tokens=10,
)
builder.add_working_set_page(working_page)

# Add available pages (not in context, can be faulted)
available_page = MemoryPage(
    page_id="msg_older",
    modality=Modality.TEXT,
    storage_tier=StorageTier.L2,
    size_tokens=500,
)
builder.add_available_page(
    page=available_page,
    hint="Earlier discussion about architecture",
)

# Build manifest
manifest = builder.build()
```

### Manifest Structure

```python
from chuk_ai_session_manager.memory import VMManifest

class VMManifest(BaseModel):
    session_id: str
    working_set: List[WorkingSetEntry]      # Pages in context
    available_pages: List[AvailablePageEntry]  # Pages that can be faulted
    policies: ManifestPolicies
```

### Working Set Entries

```python
from chuk_ai_session_manager.memory import WorkingSetEntry

class WorkingSetEntry(BaseModel):
    page_id: str
    modality: str
    level: int           # Compression level (0=full, 3=reference)
    tokens_est: int
    importance: float    # 0.0-1.0, affects eviction priority
```

### Available Page Entries

```python
from chuk_ai_session_manager.memory import AvailablePageEntry

class AvailablePageEntry(BaseModel):
    page_id: str
    modality: str
    tier: str            # L2, L3, or L4
    levels: List[int]    # Available compression levels
    hint: str            # Discovery hint (NOT content)
```

### Policies

```python
from chuk_ai_session_manager.memory import ManifestPolicies

policies = ManifestPolicies(
    faults_allowed=True,          # Can model request pages?
    max_faults_per_turn=3,        # Limit per turn
    upgrade_budget_tokens=4096,   # Token budget for faults
    prefer_levels=[2, 1, 0],      # Compression preference
)
```

### Generating Hints

Hints help models discover relevant pages without revealing content. They are for **navigation only**, not evidence.

```python
from chuk_ai_session_manager.memory import generate_simple_hint, HintType

# Auto-generate hint from page
hint = generate_simple_hint(page)

# Or use hint type
hint = generate_simple_hint(page, hint_type=HintType.TOPIC)
```

**Hint types:**

| Type | Example |
|------|---------|
| `TOPIC` | "discussion about authentication" |
| `SUMMARY` | "key points from session" |
| `MODALITY` | "image from L3" |

### JSON Output

```python
# Get manifest as JSON string
manifest_json = manifest.model_dump_json()

# With wrapping tags
wrapped = builder.to_wrapped_json()
print(wrapped)
```

**Output:**
```json
<VM:MANIFEST_JSON>
{
  "session_id": "session_abc123",
  "working_set": [
    {"page_id": "msg_001", "modality": "text", "level": 0, "tokens_est": 10, "importance": 0.5}
  ],
  "available_pages": [
    {"page_id": "msg_older", "modality": "text", "tier": "L2", "levels": [0, 1, 2], "hint": "Earlier discussion about architecture"}
  ],
  "policies": {
    "faults_allowed": true,
    "max_faults_per_turn": 3,
    "upgrade_budget_tokens": 4096,
    "prefer_levels": [2, 1, 0]
  }
}
</VM:MANIFEST_JSON>
```

### Adding Multiple Pages

```python
# Add from page table entries
for entry in page_table.get_by_tier(StorageTier.L0):
    page = get_page_content(entry.page_id)
    builder.add_working_set_page(page)

# Add available pages from lower tiers
for entry in page_table.get_by_tier(StorageTier.L2):
    builder.add_available_page_entry(
        AvailablePageEntry(
            page_id=entry.page_id,
            modality=entry.modality.value,
            tier=entry.tier.value,
            levels=[0, 1, 2],
            hint=f"{entry.modality.value} from {entry.tier.value}",
        )
    )
```

---

## Building Developer Messages

Combine rules, manifest, and context into a complete developer message.

### Using build_vm_developer_message

```python
from chuk_ai_session_manager.memory import (
    VMMode, build_vm_developer_message,
    ContextPacker, ManifestBuilder,
)

# Pack context
packer = ContextPacker()
packed = packer.pack(working_set_pages)

# Build manifest
builder = ManifestBuilder(session_id="session_123")
for page in working_set_pages:
    builder.add_working_set_page(page)
for page in available_pages:
    builder.add_available_page(page, hint="...")
manifest = builder.build()

# Build complete developer message
developer_message = build_vm_developer_message(
    mode=VMMode.STRICT,
    manifest_json=manifest.model_dump_json(),
    context=packed.content,
    system_prompt="You are a helpful assistant.",  # Optional
)

print(developer_message)
```

**Output:**
```
You are a helpful assistant.

<VM:RULES>
You are operating under STRICT Virtual Memory grounding rules.
...
</VM:RULES>

<VM:MANIFEST_JSON>
{"session_id":"session_123","working_set":[...],"available_pages":[...],"policies":{...}}
</VM:MANIFEST_JSON>

<VM:CONTEXT>
U (msg_001): "Hello"
A (msg_002): "Hi there"
</VM:CONTEXT>
```

### VM Modes

```python
from chuk_ai_session_manager.memory import VMMode

# STRICT: Model must cite sources, cannot fabricate
developer_msg = build_vm_developer_message(mode=VMMode.STRICT, ...)

# RELAXED: Model can infer but prefers faulting
developer_msg = build_vm_developer_message(mode=VMMode.RELAXED, ...)

# PASSIVE: Model sees manifest but cannot fault
developer_msg = build_vm_developer_message(mode=VMMode.PASSIVE, ...)
```

### Mode Prompts

Access mode-specific prompts directly:

```python
from chuk_ai_session_manager.memory import (
    VM_STRICT_PROMPT,
    VM_RELAXED_PROMPT,
    VM_PASSIVE_PROMPT,
    get_prompt_for_mode,
    VMMode,
)

# Get prompt for specific mode
prompt = get_prompt_for_mode(VMMode.STRICT)
```

---

## Getting VM Tools

Get tool definitions for Chat Completions API:

```python
from chuk_ai_session_manager.memory import get_vm_tools, get_vm_tools_as_dicts

# Get as Pydantic models
tools = get_vm_tools()
for tool in tools:
    print(f"Tool: {tool.function.name}")
    print(f"Description: {tool.function.description}")

# Get as dicts (for API calls)
tools_dict = get_vm_tools_as_dicts()
```

**Available tools:**

| Tool | Purpose |
|------|---------|
| `page_fault` | Load a specific page into context |
| `search_pages` | Search for pages matching a query |

### Tool Definitions

```python
from chuk_ai_session_manager.memory import PAGE_FAULT_TOOL, SEARCH_PAGES_TOOL

# page_fault tool
print(PAGE_FAULT_TOOL.function.name)  # "page_fault"
print(PAGE_FAULT_TOOL.function.parameters.properties.keys())
# dict_keys(['page_id', 'target_level'])

# search_pages tool
print(SEARCH_PAGES_TOOL.function.name)  # "search_pages"
print(SEARCH_PAGES_TOOL.function.parameters.properties.keys())
# dict_keys(['query', 'modality', 'limit'])
```

---

## Complete Example

```python
from chuk_ai_session_manager.memory import (
    # Models
    MemoryPage, Modality, StorageTier, VMMode,
    # Context building
    ContextPacker, ManifestBuilder, ManifestPolicies,
    # Tools
    build_vm_developer_message, get_vm_tools_as_dicts,
)

# Simulate conversation pages
conversation = [
    MemoryPage(
        page_id="msg_001",
        modality=Modality.TEXT,
        content="How do I implement authentication?",
        storage_tier=StorageTier.L0,
        metadata={"role": "user"},
        size_tokens=10,
    ),
    MemoryPage(
        page_id="msg_002",
        modality=Modality.TEXT,
        content="You can use JWT tokens. Here's how...",
        storage_tier=StorageTier.L0,
        metadata={"role": "assistant"},
        size_tokens=50,
    ),
]

# Older pages available for faulting
available = [
    MemoryPage(
        page_id="summary_seg_01",
        modality=Modality.TEXT,
        storage_tier=StorageTier.L2,
    ),
]

# Pack context
packer = ContextPacker()
packed = packer.pack(conversation)

# Build manifest
builder = ManifestBuilder(
    session_id="session_abc",
    policies=ManifestPolicies(max_faults_per_turn=2),
)
for page in conversation:
    builder.add_working_set_page(page)
for page in available:
    builder.add_available_page(page, hint="earlier discussion")
manifest = builder.build()

# Build developer message
developer_msg = build_vm_developer_message(
    mode=VMMode.STRICT,
    manifest_json=manifest.model_dump_json(),
    context=packed.content,
)

# Get tools for API call
tools = get_vm_tools_as_dicts()

# Ready for Chat Completions API
request = {
    "model": "gpt-4o",
    "messages": [
        {"role": "developer", "content": developer_msg},
        {"role": "user", "content": "What did we discuss about auth?"},
    ],
    "tools": tools,
}
```
