# chuk_ai_session_manager/memory/models/enums.py
"""Enums and constants for the AI Virtual Memory system."""

from enum import Enum, IntEnum

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


class FaultConfidenceThreshold(str, Enum):
    """Confidence threshold for page faults."""

    EXPLICIT = "explicit"  # Only fault when page_id is directly requested
    REFERENCED = "referenced"  # Fault if page content is referenced/needed
    SPECULATIVE = "speculative"  # Fault on potential relevance (aggressive)


# =============================================================================
# Constants
# =============================================================================

# MIME types for VM storage
MEMORY_PAGE_MIME_TYPE = "application/x-memory-page"
VM_CHECKPOINT_MIME_TYPE = "application/x-vm-checkpoint"

# All compression levels as a list (for iteration)
ALL_COMPRESSION_LEVELS: list[int] = [level.value for level in CompressionLevel]
