# CHUK AI Session Manager

**A powerful session management system for AI applications**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Automatic conversation tracking, token usage monitoring, tool call logging, infinite context support with automatic summarization, and hierarchical session relationships. Perfect for AI applications that need reliable session management.

## 🚀 Quick Start

### Installation Options

```bash
# Basic installation (memory storage only)
pip install chuk-ai-session-manager

# With Redis support for persistent storage
pip install chuk-ai-session-manager[redis]

# With enhanced token counting
pip install chuk-ai-session-manager[tiktoken]

# Full installation with all optional features
pip install chuk-ai-session-manager[all]

# Development installation
pip install chuk-ai-session-manager[dev]
```

### Quick Example

```python
from chuk_ai_session_manager import track_conversation

# Track any conversation automatically
session_id = await track_conversation(
    user_message="What's the weather like?",
    ai_response="I don't have access to real-time weather data.",
    model="gpt-3.5-turbo",
    provider="openai"
)

print(f"Conversation tracked in session: {session_id}")
```

That's it! Zero configuration required.

## ⚡ Major Features

### 🧠 **AI Virtual Memory**
OS-style memory management for AI context windows. Pages, working sets, faults, eviction, and compression — giving conversations the illusion of infinite memory.

```python
from chuk_ai_session_manager import SessionManager
from chuk_ai_session_manager.memory import (
    MemoryManager, CompressorRegistry, ImportanceWeightedLRU,
    PageType, VMMode, WorkingSetConfig,
)

# Zero-config: just enable VM on SessionManager
sm = SessionManager(enable_vm=True, vm_mode=VMMode.STRICT)

# Or fully customize eviction and compression
vm = MemoryManager(
    session_id="my_session",
    config=WorkingSetConfig(max_l0_tokens=32_000),
    eviction_policy=ImportanceWeightedLRU(),       # Swappable strategy
    compressor_registry=CompressorRegistry.default(), # Per-modality compression
)

# Create pages, add to working set
page = vm.create_page("Decision: Use JWT for auth", page_type=PageType.CLAIM)
await vm.add_to_working_set(page)

# Build context for LLM call
ctx = vm.build_context(system_prompt="You are helpful.")
# ctx["developer_message"] has VM:RULES + VM:MANIFEST_JSON + VM:CONTEXT
```

**Eviction policies**: `ImportanceWeightedLRU` (default), `LRUEvictionPolicy`, `ModalityAwareLRU` — or implement the `EvictionPolicy` protocol for custom strategies.

**Compression**: Pages compress through FULL → REDUCED → ABSTRACT → REFERENCE before eviction, saving tokens without losing context. Text, image, and passthrough compressors included; plug in custom `summarize_fn` for LLM-based compression.

See [AI Virtual Memory docs](docs/memory/README.md) for full documentation.

### 🎯 **Zero-Configuration Tracking**
```python
from chuk_ai_session_manager import SessionManager

# Just start using it
sm = SessionManager()
await sm.user_says("Hello!")
await sm.ai_responds("Hi there!", model="gpt-4")

# Get stats instantly
stats = await sm.get_stats()
print(f"Tokens: {stats['total_tokens']}, Cost: ${stats['estimated_cost']:.4f}")
```

### 🔄 **Infinite Context**
```python
# Automatically handles conversations longer than token limits
sm = SessionManager(infinite_context=True, token_threshold=4000)
await sm.user_says("Tell me about the history of computing...")
await sm.ai_responds("Computing history begins with...", model="gpt-4")
# Session will auto-segment when limits are reached
```

### ⚙️ **Storage Backends**

| Installation | Storage | Use Case | Performance |
|-------------|---------|----------|-------------|
| `pip install chuk-ai-session-manager` | Memory | Development, testing | 1.8M ops/sec |
| `pip install chuk-ai-session-manager[redis]` | Redis | Persistent, distributed | 20K ops/sec |

### 🛡️ **Conversation Guards and Tool State**
Runtime guardrails that prevent runaway tool loops, track value bindings, and enforce grounded tool calls.

```python
from chuk_ai_session_manager.guards import get_tool_state, ToolStateManager

# Get the singleton tool state manager
tool_state = get_tool_state()

# Track tool calls and bind results as $v1, $v2, ...
binding = tool_state.bind_value("sqrt", {"x": 16}, 4.0)
# LLM can now reference $v1 in subsequent calls

# Check for runaway tool loops
status = tool_state.check_runaway()

# Detect ungrounded calls (missing $vN references)
check = tool_state.check_ungrounded_call("normal_cdf", {"mean": 0, "std": 1, "x": 1.5})

# Reset state for a new prompt
tool_state.reset_for_new_prompt()
```

**Guard components:**
- `ToolStateManager` - Coordinator for all guards, bindings, and cache
- `BindingManager` - `$vN` reference system for tracking tool results
- `ResultCache` - Tool result caching for deduplication
- `UngroundedGuard` - Detects calls with missing computed-value references
- Runtime guards (budget, runaway, per-tool limits) from `chuk-tool-processor`

### 🧩 **Procedural Memory**
Learn from tool call history to improve future tool use.

```python
from chuk_ai_session_manager import ToolMemoryManager, ProceduralContextFormatter

# Record tool outcomes
memory = ToolMemoryManager()
await memory.record("calculator", {"op": "add", "a": 5, "b": 3}, result=8, success=True)

# Format learned patterns for the model's context
formatter = ProceduralContextFormatter()
context = formatter.format(memory.get_patterns())
```

### 🛠️ **Tool Integration**
```python
# Automatic tool call tracking
await sm.tool_used(
    tool_name="calculator",
    arguments={"operation": "add", "a": 5, "b": 3},
    result={"result": 8}
)
```

## 💡 Common Use Cases

### Web App Conversation Tracking
```python
from chuk_ai_session_manager import track_conversation

# In your chat endpoint
session_id = await track_conversation(
    user_message=request.message,
    ai_response=ai_response,
    model="gpt-4",
    provider="openai",
    session_id=request.session_id  # Continue existing conversation
)
```

### LLM Wrapper with Automatic Tracking
```python
from chuk_ai_session_manager import track_llm_call
import openai

async def my_openai_call(prompt):
    response = await openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Automatically tracked
response, session_id = await track_llm_call(
    user_input="Explain machine learning",
    llm_function=my_openai_call,
    model="gpt-3.5-turbo",
    provider="openai"
)
```

### Long Conversations with Auto-Segmentation
```python
from chuk_ai_session_manager import track_infinite_conversation

# Start a conversation
session_id = await track_infinite_conversation(
    user_message="Tell me about the history of computing",
    ai_response="Computing history begins with ancient calculating devices...",
    model="gpt-4",
    token_threshold=4000  # Auto-segment after 4000 tokens
)

# Continue the conversation - will auto-segment if needed
session_id = await track_infinite_conversation(
    user_message="What about quantum computers?",
    ai_response="Quantum computing represents a fundamental shift...",
    session_id=session_id,
    model="gpt-4"
)
```

## 🔧 Configuration

### Storage Configuration

```bash
# Memory provider (default) - fast, no persistence
export SESSION_PROVIDER=memory

# Redis provider - persistent, distributed (requires redis extra)
export SESSION_PROVIDER=redis
export SESSION_REDIS_URL=redis://localhost:6379/0
```

### Installation Matrix

| Command | Memory | Redis | Token Counting | Use Case |
|---------|--------|-------|----------------|----------|
| `pip install chuk-ai-session-manager` | ✅ | ❌ | Basic | Development |
| `pip install chuk-ai-session-manager[redis]` | ✅ | ✅ | Basic | Persistent |
| `pip install chuk-ai-session-manager[tiktoken]` | ✅ | ❌ | Enhanced | Better accuracy |
| `pip install chuk-ai-session-manager[all]` | ✅ | ✅ | Enhanced | Full features |

## 📊 Monitoring & Analytics

```python
# Get comprehensive session analytics
stats = await sm.get_stats(include_all_segments=True)

print(f"""
🚀 Session Analytics Dashboard
============================
Session ID: {stats['session_id']}
Total Messages: {stats['total_messages']}
User Messages: {stats['user_messages']}
AI Messages: {stats['ai_messages']}
Tool Calls: {stats['tool_calls']}
Total Tokens: {stats['total_tokens']}
Total Cost: ${stats['estimated_cost']:.6f}
Session Segments: {stats.get('session_segments', 1)}
""")
```

## 🏗️ Why CHUK AI Session Manager?

- **Zero Configuration**: Start tracking conversations in 3 lines of code
- **Infinite Context**: Never worry about token limits again
- **Universal**: Works with any LLM provider (OpenAI, Anthropic, etc.)
- **Robust**: Built-in persistence, monitoring, and error handling
- **Token Aware**: Automatic cost tracking across all providers
- **Tool Friendly**: Seamless tool call logging and retry mechanisms
- **Guardrails**: Runtime guards prevent runaway tool loops and ungrounded calls
- **Procedural Memory**: Learn from tool call history to improve future use

## 🛡️ Error Handling

```python
from chuk_ai_session_manager import (
    SessionManagerError,
    SessionNotFound,
    TokenLimitExceeded
)

try:
    session_id = await track_conversation("Hello", "Hi there")
except SessionNotFound as e:
    print(f"Session not found: {e}")
except TokenLimitExceeded as e:
    print(f"Token limit exceeded: {e}")
except SessionManagerError as e:
    print(f"General session error: {e}")
```

## 🔄 Dependencies

- **Required**: `chuk-sessions` (session storage), `pydantic` (data models), `chuk-tool-processor` (tool integration)
- **Optional**: `redis` (Redis storage), `tiktoken` (accurate token counting)

## 📄 License

Apache 2.0 - build amazing AI applications with confidence!

---

**Ready to build better AI applications?**

```bash
pip install chuk-ai-session-manager
```

**Start tracking conversations in 30 seconds!**