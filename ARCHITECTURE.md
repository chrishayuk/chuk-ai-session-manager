# CHUK AI Session Manager — Architecture Principles

> These principles govern all code in chuk-ai-session-manager.
> Every PR should be evaluated against them.

---

## 1. Async Native

Every public API is `async def`. No blocking calls anywhere in the call path.

**Rules:**
- All session operations, storage access, and memory management use `async`/`await`
- Store operations (`save`, `get`, `list`) are always awaited
- Use `asyncio.Lock` (not `threading.Lock`) for shared state — e.g., `SessionManager._lock`
- Synchronous helpers (pure computation, token counting, model validation) are acceptable but must not block the event loop
- Callbacks are async: `SummarizeFn = Callable[[str, int], Awaitable[str]]`, `LLMCallbackAsync`
- Initialization is lazy — `_ensure_initialized()` is awaited on first message, not in `__init__`

**Why:** Session management is inherently concurrent — multiple conversations, tool calls, and memory operations run in parallel. A single blocking call stalls every session sharing that event loop.

---

## 2. Pydantic Native

Structured data flows through Pydantic v2 models, not raw dicts.

**Rules:**
- Inputs and outputs of public APIs are Pydantic `BaseModel` instances
- Configuration objects are Pydantic models — `WorkingSetConfig`, `UngroundedGuardConfig`, `FormatterConfig`
- Use `Field(...)` with defaults and descriptions for all model attributes
- Use `model_validator`, `field_validator` for cross-field constraints
- Use `ConfigDict(arbitrary_types_allowed=True)` when models hold non-Pydantic types
- Use `model_post_init()` for lazy initialization (e.g., wiring up guards in `ToolStateManager`)
- Generic models (`Session[MessageT]`) for type-safe flexibility
- Serialization goes through `.model_dump()` / `.model_validate()`

**Why:** Pydantic gives validation at construction time, clear field documentation, and serialization for free. Raw dicts defer errors to runtime and make refactoring dangerous.

---

## 3. No Dictionary Goop

Never pass `dict[str, Any]` through public interfaces when a model will do.

**Rules:**
- If a dict has a known shape, define a model or `TypedDict`
- If a function returns `dict[str, Any]`, ask: should this be a model?
- Accessing nested dicts with `.get("key")` chains is a code smell — model it
- Internal dict usage for caches, indexes, and transient lookups is fine
- `SessionMetadata.properties: dict[str, Any]` is the one acceptable flexible dict — it's user-defined metadata
- Stats and metrics are returned as typed models (`WorkingSetStats`, `VMMetrics`, `TokenSummary`)

**Why:** `data["events"][0]["token_usage"]["model"]` is unreadable, unrefactorable, and produces `KeyError` at runtime instead of a validation error at construction.

---

## 4. No Magic Strings

Use enums, constants, or Pydantic `Literal` types — never bare string comparisons.

**Rules:**
- Message roles → `MessageRole` enum (`user`, `assistant`, `tool`, `system`)
- Event types → `EventType` enum (`message`, `summary`, `tool_call`, `tool_trace`, etc.)
- Event sources → `EventSource` enum (`user`, `ai`, `tool`, `system`)
- Storage tiers → `StorageTier` enum (`L0` through `L4`)
- Compression levels → `CompressionLevel` enum (`FULL`, `REDUCED`, `ABSTRACT`, `REFERENCE`)
- Page types → `PageType` enum (`transcript`, `summary`, `artifact`, `claim`, etc.)
- VM modes → `VMMode` enum (`STRICT`, `RELAXED`, `PASSIVE`)
- Tool outcomes → `ToolOutcome` enum (`success`, `failure`, `partial`, `timeout`, `cancelled`)
- Run statuses → `RunStatus` enum (`in_progress`, `completed`, `failed`, `pending`)
- Module-level constants for numeric values: `VM_IMPORTANCE_USER = 0.6`, `VM_IMPORTANCE_SUMMARY = 0.9`
- If you find yourself writing `if x == "some_string"`, define a constant or enum first

**Why:** Magic strings are invisible to refactoring tools, produce silent bugs when misspelled, and can't be auto-completed by IDEs.

---

## 5. Protocol Over ABC

Use `Protocol` (structural subtyping) for pluggable strategies — never ABCs.

**Rules:**
- Eviction strategies implement the `EvictionPolicy` protocol — no base class required
- Compressors implement the `Compressor` protocol — registered per `Modality`
- Storage backends follow implicit protocols via duck typing
- Mark protocols `@runtime_checkable` when `isinstance` checks are needed
- Implementations are standalone classes — they compose, they don't inherit
- New strategies require zero changes to existing implementations

**Current protocols:**
- `EvictionPolicy` → `ImportanceWeightedLRU`, `LRUEvictionPolicy`, `ModalityAwareLRU`
- `Compressor` → `TextCompressor`, `ImageCompressor`, `PassthroughCompressor`

**Why:** ABCs force inheritance hierarchies and tight coupling. Protocols let any class participate by implementing the right methods — no registration, no base class, no diamond problems.

---

## 6. Layered Architecture

SessionManager is a facade. Subsystems are independent and composable.

```
SessionManager (facade)
    ├── InfiniteConversationManager  (session chaining, summarization)
    ├── MemoryManager                (virtual memory, context building)
    ├── ToolStateManager             (guards, bindings, result cache)
    ├── ToolMemoryManager            (procedural memory, patterns)
    └── SessionAwareToolProcessor    (tool execution with session logging)
        │
        ▼
    ChukSessionsStore (persistence)
```

**Rules:**
- SessionManager coordinates; subsystems do the work
- Subsystems don't import each other — they communicate through SessionManager
- Each subsystem is independently testable with its own mocks
- VM is optional — `enable_vm=False` disables the entire memory subsystem gracefully
- Storage is injected — `SessionManager(store=...)`, never created deep in call stacks
- Configuration flows down — eviction policies, compressor registries, and callbacks are passed at construction

**Why:** A monolithic session manager would be untestable and unextendable. Layering lets users opt into only the features they need.

---

## 7. OS-Inspired Virtual Memory

The memory subsystem follows operating system memory management principles.

**Storage hierarchy:**
| Tier | Analogy | Purpose | Default Budget |
|------|---------|---------|----------------|
| L0 | Registers | Current LLM context window | 32K tokens |
| L1 | L1 Cache | Working set (hot pages) | Configurable |
| L2 | Main Memory | Session store (ChukSessions) | Unlimited |
| L3 | Disk | Artifact storage (filesystem) | Unlimited |
| L4 | Tape | Cold archive (S3) | Unlimited |

**Rules:**
- All content is stored as `MemoryPage` objects with metadata (importance, modality, timestamps)
- Pages move between tiers via eviction (L0→L1→L2) and fault-in (L2→L1→L0)
- `PageTable` tracks every page's location; `TLB` caches hot lookups
- Eviction prefers compress-before-evict — reduce tokens before removing from working set
- Pinned pages (`is_pinned=True`) are never evicted — use for critical system context
- Anti-thrash protection prevents recently faulted pages from immediate re-eviction
- `DemandPagingPrePass` detects page faults before LLM calls — not after
- Context is packed into `VM:CONTEXT` blocks via `ContextPacker`

**Why:** LLM context windows are finite and expensive. OS memory management — paging, eviction, working sets — solves exactly this class of problem. The analogy gives us a proven vocabulary and proven algorithms.

---

## 8. Clean Code

Small functions. Clear names. Single responsibility. Minimal coupling.

**Rules:**
- Functions do one thing; if a function needs a comment explaining what it does, extract sub-functions
- Modules have a single area of responsibility
- Prefer composition and protocols over deep inheritance
- No dead code, no commented-out blocks, no `# TODO: maybe later` without a tracking issue
- Limit module size — if a file exceeds ~500 lines of logic, consider splitting
- Use `from __future__ import annotations` for forward references, not string literals
- Modern Python typing: `str | None` not `Optional[str]`, `collections.abc` not `typing`

**Why:** Session management spans many concerns (storage, memory, tools, summarization). Clarity in each piece makes the whole system debuggable.

---

## 9. Custom Exception Hierarchy

Errors are typed, contextual, and never swallowed silently.

```
SessionManagerError (base)
├── SessionNotFound          (session_id)
├── SessionAlreadyExists     (session_id)
├── InvalidSessionOperation  (session_id, operation)
├── TokenLimitExceeded       (limit, actual)
├── StorageError             (operation, backend)
└── ToolProcessingError      (tool_name, details)
```

**Rules:**
- Every custom exception carries structured context (session_id, limits, etc.)
- Never `except Exception: pass` in production paths
- Storage failures degrade gracefully — log and continue where possible
- Import errors for optional dependencies (`tiktoken`, `redis`) use try/except at module level
- Validation errors surface at construction time via Pydantic, not at runtime via KeyError

**Why:** `SessionNotFound("abc123")` tells you exactly what went wrong. `KeyError: 'session'` tells you nothing.

---

## 10. Test Coverage ≥ 90% Per File

Every source file must have ≥ 90% line coverage individually.

**Rules:**
- Each `src/.../foo.py` has corresponding test coverage (often across multiple test files)
- Coverage is measured per-file, not just as a project aggregate
- Test both happy paths and error/edge cases
- Async tests use `pytest-asyncio` with auto mode
- Mock external dependencies (storage backends, LLM callbacks) — never hit real services in unit tests
- Use `pytest.mark.unit`, `pytest.mark.integration`, `pytest.mark.slow` markers

**Current status:** 99% aggregate, 1703 tests passing. All files ≥ 91%.

---

## 11. Separation of Concerns

The session manager manages **state**. LLMs decide **content**. Tools execute **actions**.

**Rules:**
- SessionManager never generates content — it tracks, stores, and retrieves
- Summarization delegates to an injected LLM callback — the manager doesn't pick the model
- Tool execution delegates to `SessionAwareToolProcessor` — the manager logs the result
- Guards can block or warn, but never rewrite the caller's intent silently
- Bindings (`$v0`, `$v1`) track tool outputs — they don't interpret them
- Virtual memory decides what fits in context — not what's important to the conversation
- The `UngroundedGuard` detects ungrounded tool calls — it doesn't fix them

**Why:** Mixing state management with content decisions creates untestable, unpredictable systems. Clear boundaries let each component be replaced independently.

---

## 12. Circular Import Avoidance

The codebase uses four patterns to prevent circular imports. Use them consistently.

**Pattern 1: `TYPE_CHECKING` guard**
```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chuk_ai_session_manager.models.session import Session
```

**Pattern 2: Late import in methods**
```python
async def async_init(self) -> None:
    from chuk_ai_session_manager.session_storage import ChukSessionsStore
```

**Pattern 3: Try/except import guard** (for __init__.py re-exports)
```python
try:
    from chuk_ai_session_manager.models.event_source import EventSource
except ImportError:
    pass
```

**Pattern 4: Raw types at function boundaries**
```python
def event_to_page(message: str, role: str, ...) -> MemoryPage:
    """Accept raw strings, not SessionEvent — avoids importing the model."""
```

**Rules:**
- Prefer `TYPE_CHECKING` for type hints — it's the cleanest solution
- Use late imports only when runtime access is needed and the import would be circular at module level
- Never add a direct import that creates a cycle — check before committing

---

## 13. Observable by Default

Every subsystem exposes structured metrics without opt-in.

**Rules:**
- Module-level loggers: `logger = logging.getLogger(__name__)` in every module
- VM exposes `get_metrics()` → `VMMetrics` (working set stats, page table stats, TLB stats, storage stats)
- Sessions expose `get_stats()` → `TokenSummary` (token counts, costs)
- Conversations expose `get_history()` → structured message list
- Mutation log records every page lifecycle event with timestamps
- Observability must not throw — if metrics fail, execution still succeeds
- Structured log messages include session_id and context where available

**Why:** When an LLM conversation goes wrong at 2 AM, you need to know: what was in context, what was evicted, what was faulted in, and why.

---

## Checklist for PRs

- [ ] All new public APIs are `async def`
- [ ] New data structures use Pydantic models (not raw dicts)
- [ ] No new magic string comparisons (use enums/constants)
- [ ] New file has corresponding test coverage with ≥ 90% line coverage
- [ ] No blocking I/O in async code paths
- [ ] Errors use custom exception hierarchy with structured context
- [ ] Pluggable strategies use `Protocol`, not ABC
- [ ] No circular imports — use `TYPE_CHECKING`, late imports, or raw types
- [ ] New subsystem features are optional and degrade gracefully when disabled
- [ ] Observability: new code paths have logging and metrics models where appropriate
