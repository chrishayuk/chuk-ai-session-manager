#!/usr/bin/env python3
"""
AI Virtual Memory Demo (v0.8)

Demonstrates the VM protocol with a real LLM. Shows:
- VM:MANIFEST_JSON generation with page types
- VM:CONTEXT packing
- page_fault tool handling with fault reasons
- search_pages tool handling
- Strict mode grounding
- PinnedSet for critical pages
- AntiThrashPolicy for eviction protection
- MutationLogLite for debugging
- SimplePrefetcher for heuristic prefetch
- UserExperienceMetrics for UX tracking

Usage:
    # With OpenAI
    export OPENAI_API_KEY=your-key
    python examples/vm_demo.py

    # With Anthropic
    export ANTHROPIC_API_KEY=your-key
    python examples/vm_demo.py --provider anthropic

    # With local model (Ollama)
    python examples/vm_demo.py --provider ollama --model llama3.2

    # Show VM internals
    python examples/vm_demo.py --show-internals

    # Or create a .env file with your API keys
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Optional

# Import v0.8 VM components
from chuk_ai_session_manager.memory import (
    # Enums
    Actor,
    FaultReason,
    MutationType,
    PageType,
    StorageTier,
    # Core components
    AntiThrashPolicy,
    FaultPolicy,
    MutationLogLite,
    PinnedSet,
    SimplePrefetcher,
    UserExperienceMetrics,
)


# Load .env file if it exists
def load_env():
    """Load environment variables from .env file."""
    env_paths = [
        Path.cwd() / ".env",
        Path(__file__).parent.parent / ".env",
        Path.home() / ".env",
    ]

    for env_path in env_paths:
        if env_path.exists():
            print(f"Loading environment from {env_path}")
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key not in os.environ:  # Don't override existing
                            os.environ[key] = value
            return True
    return False


load_env()

# Simulated page storage with v0.8 page types and provenance
PAGE_STORE = {
    # === TRANSCRIPT PAGES (raw turns) ===
    "msg_001": {
        "modality": "text",
        "page_type": PageType.TRANSCRIPT,
        "levels": {
            0: "The user asked about implementing a REST API with authentication. They mentioned wanting to use JWT tokens and needed advice on refresh token strategies.",
            2: "User asked about REST API + JWT auth implementation.",
        },
        "tier": "L2",
        "provenance": [],
        "meta": {"word_count": 28, "turn": 1},
    },
    "msg_002": {
        "modality": "text",
        "page_type": PageType.TRANSCRIPT,
        "levels": {
            0: "I explained that JWT tokens should have short expiration times (15-30 minutes) and that refresh tokens should be stored securely, rotated on use, and have longer expiration (7-30 days). I recommended using HTTP-only cookies for refresh tokens.",
            2: "Explained JWT best practices: short-lived access tokens, secure refresh token storage, rotation.",
        },
        "tier": "L2",
        "provenance": [],
        "meta": {"word_count": 42, "turn": 2},
    },
    "msg_003": {
        "modality": "text",
        "page_type": PageType.TRANSCRIPT,
        "levels": {
            0: "The user then asked about rate limiting strategies for their API endpoints.",
            2: "User asked about API rate limiting.",
        },
        "tier": "L2",
        "provenance": [],
        "meta": {"word_count": 12, "turn": 3},
    },
    # === SUMMARY PAGES (LLM-generated) ===
    "summary_seg_01": {
        "modality": "text",
        "page_type": PageType.SUMMARY,
        "levels": {
            2: "Segment 1 covered: REST API design discussion, JWT authentication strategy, refresh token handling, rate limiting considerations.",
        },
        "tier": "L2",
        "provenance": ["msg_001", "msg_002", "msg_003"],  # Derived from these
        "meta": {},
    },
    # === CLAIM PAGES (high-value decisions/facts) ===
    "claim_jwt_expiry": {
        "modality": "text",
        "page_type": PageType.CLAIM,
        "levels": {
            0: "DECISION: JWT access tokens should expire in 15-30 minutes. Refresh tokens should expire in 7-30 days and be rotated on each use.",
        },
        "tier": "L1",  # Claims stay hot
        "provenance": ["msg_002"],  # Derived from this message
        "meta": {"importance": 0.9},
    },
    "claim_refresh_storage": {
        "modality": "text",
        "page_type": PageType.CLAIM,
        "levels": {
            0: "DECISION: Refresh tokens should be stored in HTTP-only cookies for security.",
        },
        "tier": "L1",
        "provenance": ["msg_002"],
        "meta": {"importance": 0.85},
    },
    # === ARTIFACT PAGES (tool-created) ===
    "code_snippet_001": {
        "modality": "text",
        "page_type": PageType.ARTIFACT,
        "levels": {
            0: """```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
import jwt

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
```""",
            2: "FastAPI JWT verification code snippet with OAuth2 bearer scheme.",
        },
        "tier": "L3",
        "provenance": [],
        "meta": {"language": "python", "lines": 14},
    },
    # === PROCEDURE PAGES (learned tool patterns) ===
    "procedure_auth_flow": {
        "modality": "text",
        "page_type": PageType.PROCEDURE,
        "levels": {
            0: "PROCEDURE: When implementing auth, always: 1) Define token expiry policy, 2) Set up refresh token rotation, 3) Configure HTTP-only cookies, 4) Add rate limiting.",
        },
        "tier": "L2",
        "provenance": ["msg_001", "msg_002", "claim_jwt_expiry"],
        "meta": {},
    },
    # === IMAGE PAGES ===
    "img_architecture": {
        "modality": "image",
        "page_type": PageType.ARTIFACT,
        "levels": {
            0: "https://example.com/images/jwt_auth_architecture_full.png",
            1: "https://example.com/images/jwt_auth_architecture_thumb.png",
            2: "Architecture diagram showing JWT authentication flow: Client → API Gateway → Auth Service → Token Validation → Protected Resource.",
        },
        "tier": "L3",
        "provenance": [],
        "meta": {
            "dimensions": [1200, 800],
            "mime_type": "image/png",
            "size_bytes": 245000,
        },
    },
    # === AUDIO PAGES ===
    "audio_meeting_001": {
        "modality": "audio",
        "page_type": PageType.TRANSCRIPT,
        "levels": {
            0: "https://example.com/audio/team_meeting_2024_01_15.mp3",
            1: {
                "transcript": """[00:00] Alex: Let's discuss the authentication implementation.
[00:15] Jordan: I've been looking at JWT vs session-based auth.
[01:30] Alex: What are the trade-offs you've found?
[02:00] Jordan: JWT is stateless, great for microservices, but token revocation is tricky.
[03:45] Alex: We could use short-lived tokens with refresh rotation.""",
            },
            2: "Team meeting (6 min): Alex and Jordan discussed JWT vs session auth. Decided on short-lived JWT + refresh tokens.",
        },
        "tier": "L4",
        "provenance": [],
        "meta": {
            "duration_seconds": 375,
            "speakers": ["Alex", "Jordan"],
        },
    },
    # === STRUCTURED DATA PAGES ===
    "data_api_metrics": {
        "modality": "structured",
        "page_type": PageType.ARTIFACT,
        "levels": {
            0: {
                "period": "2024-01-15 to 2024-01-21",
                "endpoints": {
                    "/auth/login": {
                        "requests": 45230,
                        "p50_ms": 120,
                        "error_rate": 0.02,
                    },
                    "/auth/refresh": {
                        "requests": 12450,
                        "p50_ms": 45,
                        "error_rate": 0.001,
                    },
                },
                "total_requests": 303800,
                "availability": 99.94,
            },
            2: "API metrics (Jan 15-21): 303K requests, 99.94% availability. Login p50: 120ms.",
        },
        "tier": "L3",
        "provenance": [],
        "meta": {"schema": "api_metrics_v1"},
    },
}


# =============================================================================
# v0.8 VM Components
# =============================================================================


class VMRuntime:
    """
    Virtual Memory Runtime with v0.8 features.

    Manages:
    - PinnedSet: Pages that should never be evicted
    - AntiThrashPolicy: Prevent evicting recently faulted pages
    - MutationLogLite: Track all page operations
    - SimplePrefetcher: Predict pages to prefetch
    - FaultPolicy: Limit faults per turn
    - UserExperienceMetrics: Track UX quality
    """

    def __init__(self, session_id: str = "demo_session_001"):
        self.session_id = session_id
        self.current_turn = 0

        # Pinning
        self.pinned_set = PinnedSet(
            auto_pin_last_n_turns=2,
            auto_pin_system_prompt=True,
            auto_pin_claims=True,
        )

        # Anti-thrash
        self.anti_thrash = AntiThrashPolicy(
            eviction_cooldown_turns=3,
            fault_protection_turns=2,
        )

        # Mutation log
        self.mutation_log = MutationLogLite(session_id=session_id)

        # Prefetcher
        self.prefetcher = SimplePrefetcher(
            max_claims_to_prefetch=3,
            max_recent_tools=3,
        )

        # Fault policy
        self.fault_policy = FaultPolicy(
            max_faults_per_turn=3,
            max_fault_tokens_per_turn=8192,
        )

        # UX metrics
        self.ux_metrics = UserExperienceMetrics()

        # Working set tracking
        self.working_set: list[str] = []
        self.l1_cache: set[str] = set()

    def new_turn(self):
        """Start a new turn."""
        self.current_turn += 1
        self.fault_policy.new_turn()

        # Record context snapshot for debugging
        self.mutation_log.record_context_at_turn(self.current_turn, self.working_set)

    def pin_page(self, page_id: str, reason: str = "explicit"):
        """Pin a page to prevent eviction."""
        self.pinned_set.pin(page_id)
        self.mutation_log.record_mutation(
            page_id=page_id,
            mutation_type=MutationType.PIN,
            tier_after=StorageTier.L0,
            actor=Actor.SYSTEM,
            cause=reason,
            turn=self.current_turn,
        )

    def auto_pin_claims(self):
        """Auto-pin all claim pages."""
        for page_id, page in PAGE_STORE.items():
            if page.get("page_type") == PageType.CLAIM:
                self.pinned_set.auto_pin(page_id)

    def can_fault(self, page_id: str, estimated_tokens: int) -> tuple[bool, str]:
        """Check if a fault is allowed."""
        if not self.fault_policy.can_fault(estimated_tokens):
            return False, "Fault limit reached for this turn"
        if page_id in self.working_set:
            return False, "Page already in working set"
        return True, ""

    def record_fault(
        self,
        page_id: str,
        reason: FaultReason,
        tokens_loaded: int,
        latency_ms: float = 0.0,
    ):
        """Record a page fault."""
        # Update fault policy
        self.fault_policy.record_fault(tokens_loaded)

        # Update anti-thrash
        self.anti_thrash.record_fault(page_id, self.current_turn)

        # Update mutation log
        self.mutation_log.record_mutation(
            page_id=page_id,
            mutation_type=MutationType.FAULT_IN,
            tier_after=StorageTier.L0,
            tier_before=StorageTier.L2,
            actor=Actor.MODEL,
            cause=reason.value,
            turn=self.current_turn,
        )

        # Update UX metrics
        self.ux_metrics.record_fault(
            page_id=page_id,
            reason=reason,
            turn=self.current_turn,
            tokens_loaded=tokens_loaded,
            latency_ms=latency_ms,
        )

        # Update prefetcher
        self.prefetcher.record_page_access(page_id)

        # Add to working set
        if page_id not in self.working_set:
            self.working_set.append(page_id)

    def get_eviction_candidates(self) -> list[str]:
        """Get pages that can be evicted (respects pinning and anti-thrash)."""
        candidates = []
        for page_id in self.working_set:
            # Skip pinned pages
            if self.pinned_set.is_pinned(page_id):
                continue
            # Skip recently faulted pages
            if not self.anti_thrash.can_evict(page_id, self.current_turn):
                continue
            candidates.append(page_id)
        return candidates

    def evict_page(self, page_id: str):
        """Evict a page from working set."""
        if page_id in self.working_set:
            self.working_set.remove(page_id)
            self.anti_thrash.record_eviction(page_id, self.current_turn)
            self.mutation_log.record_mutation(
                page_id=page_id,
                mutation_type=MutationType.EVICT,
                tier_after=StorageTier.L2,
                tier_before=StorageTier.L0,
                actor=Actor.SYSTEM,
                cause="eviction_pressure",
                turn=self.current_turn,
            )

    async def prefetch_for_turn(self) -> list[str]:
        """Get pages to prefetch for this turn."""
        return await self.prefetcher.prefetch_on_turn_start(self.session_id)

    def get_stats(self) -> dict:
        """Get runtime statistics."""
        return {
            "session_id": self.session_id,
            "current_turn": self.current_turn,
            "working_set_size": len(self.working_set),
            "pinned_count": self.pinned_set.count(),
            "mutations_logged": self.mutation_log.mutation_count(),
            "fault_policy": {
                "faults_this_turn": self.fault_policy.faults_this_turn,
                "tokens_used_this_turn": self.fault_policy.tokens_used_this_turn,
            },
            "ux_metrics": {
                "recall_success_rate": self.ux_metrics.recall_success_rate(),
                "thrash_index": self.ux_metrics.thrash_index(),
            },
            "prefetcher_stats": self.prefetcher.get_stats(),
        }


# Global runtime instance
vm_runtime = VMRuntime()


def get_level_hints(modality: str) -> dict:
    """Return hints about what each compression level provides."""
    return {
        "text": {
            0: "full text",
            1: "key excerpts",
            2: "summary",
            3: "topic tags only",
        },
        "image": {
            0: "full resolution URL",
            1: "thumbnail URL",
            2: "caption/description",
            3: "reference only",
        },
        "audio": {
            0: "audio URL + full transcript",
            1: "transcript with timestamps",
            2: "summary",
            3: "duration + topic only",
        },
        "video": {
            0: "video URL + full content",
            1: "scene list with timestamps + transcript",
            2: "summary",
            3: "duration + topic only",
        },
        "structured": {
            0: "full data",
            1: "key fields",
            2: "summary",
            3: "schema only",
        },
    }.get(modality, {})


def build_manifest(working_set_ids: list[str], all_page_ids: list[str]) -> dict:
    """Build VM:MANIFEST_JSON from page store with v0.8 page types."""
    working_set = []
    for page_id in working_set_ids:
        if page_id in PAGE_STORE:
            page = PAGE_STORE[page_id]
            content = page["levels"].get(0, page["levels"].get(2, ""))
            if isinstance(content, str):
                tokens_est = len(content) // 4
            else:
                tokens_est = len(str(content)) // 4

            working_set.append(
                {
                    "page_id": page_id,
                    "modality": page["modality"],
                    "page_type": page.get("page_type", PageType.TRANSCRIPT).value,
                    "level": 0 if 0 in page["levels"] else 2,
                    "tokens_est": tokens_est,
                    "pinned": vm_runtime.pinned_set.is_pinned(page_id),
                    "provenance": page.get("provenance", []),
                }
            )

    available_pages = []
    for page_id in all_page_ids:
        if page_id not in working_set_ids and page_id in PAGE_STORE:
            page = PAGE_STORE[page_id]
            modality = page["modality"]
            page_type = page.get("page_type", PageType.TRANSCRIPT)
            meta = page.get("meta", {})

            hint_content = page["levels"].get(2, page["levels"].get(0, ""))
            if isinstance(hint_content, str):
                hint = (
                    hint_content[:80] + "..."
                    if len(hint_content) > 80
                    else hint_content
                )
            else:
                hint = str(hint_content)[:80]

            level_hints = get_level_hints(modality)
            levels_available = sorted(page["levels"].keys())
            level_info = {
                lvl: level_hints.get(lvl, f"level {lvl}") for lvl in levels_available
            }

            entry = {
                "page_id": page_id,
                "modality": modality,
                "page_type": page_type.value,
                "tier": page["tier"],
                "levels": level_info,
                "hint": hint,
                "provenance": page.get("provenance", []),
            }

            # Add modality-specific metadata
            if modality == "audio":
                duration = meta.get("duration_seconds", 0)
                mins, secs = divmod(duration, 60)
                entry["duration"] = f"{int(mins)}:{int(secs):02d}"
                entry["speakers"] = meta.get("speakers", [])
            elif modality == "image":
                if "dimensions" in meta:
                    entry["dimensions"] = meta["dimensions"]

            available_pages.append(entry)

    return {
        "session_id": vm_runtime.session_id,
        "turn": vm_runtime.current_turn,
        "working_set": working_set,
        "available_pages": available_pages,
        "policies": {
            "faults_allowed": True,
            "max_faults_per_turn": vm_runtime.fault_policy.max_faults_per_turn,
            "faults_remaining": vm_runtime.fault_policy.max_faults_per_turn
            - vm_runtime.fault_policy.faults_this_turn,
            "upgrade_budget_tokens": vm_runtime.fault_policy.max_fault_tokens_per_turn
            - vm_runtime.fault_policy.tokens_used_this_turn,
            "prefer_levels": [2, 1, 0],
        },
        "page_types": {
            "transcript": "Raw conversation turns (normal eviction)",
            "summary": "LLM-generated summaries (low eviction, rebuildable)",
            "claim": "Decisions/facts (very low eviction, high-value)",
            "procedure": "Learned patterns (low eviction)",
            "artifact": "Tool-created content (normal eviction)",
            "index": "Page metadata for search (very low eviction)",
        },
    }


def build_context(working_set_ids: list[str]) -> str:
    """Build VM:CONTEXT from working set."""
    lines = []
    for page_id in working_set_ids:
        if page_id in PAGE_STORE:
            page = PAGE_STORE[page_id]
            modality = page["modality"]
            page_type = page.get("page_type", PageType.TRANSCRIPT)
            meta = page.get("meta", {})

            # Determine prefix based on page type and modality
            if page_type == PageType.CLAIM:
                prefix = "!"  # Claims are important
            elif page_type == PageType.SUMMARY:
                prefix = "S"
            elif page_type == PageType.PROCEDURE:
                prefix = "P"
            elif modality == "image":
                prefix = "I"
            elif modality == "audio":
                prefix = "D"
            elif modality == "video":
                prefix = "V"
            elif modality == "structured":
                prefix = "X"
            elif page_id.startswith("msg_"):
                idx = int(page_id.split("_")[1])
                prefix = "U" if idx % 2 == 1 else "A"
            else:
                prefix = "T"

            # Pinned indicator
            pin_marker = "📌" if vm_runtime.pinned_set.is_pinned(page_id) else ""

            # Build display based on modality
            if modality == "text":
                content = page["levels"].get(0, page["levels"].get(2, ""))
                if isinstance(content, str):
                    display = content[:200] + "..." if len(content) > 200 else content
                else:
                    display = str(content)[:200]
                lines.append(f'{prefix} ({page_id}){pin_marker}: "{display}"')

            elif modality == "image":
                caption = page["levels"].get(2, "No caption")
                dims = meta.get("dimensions", [0, 0])
                lines.append(
                    f'{prefix} ({page_id}){pin_marker}: [IMAGE: {dims[0]}x{dims[1]}, "{caption[:100]}"]'
                )

            elif modality == "audio":
                summary = page["levels"].get(2, "No summary")
                duration = meta.get("duration_seconds", 0)
                speakers = meta.get("speakers", [])
                mins = duration // 60
                secs = duration % 60
                lines.append(
                    f'{prefix} ({page_id}){pin_marker}: [AUDIO: {mins}:{secs:02d}, speakers: {", ".join(speakers)}, "{summary[:80]}"]'
                )

            elif modality == "structured":
                summary = page["levels"].get(2, "No summary")
                schema = meta.get("schema", "unknown")
                lines.append(
                    f'{prefix} ({page_id}){pin_marker}: [DATA: schema={schema}, "{summary[:100]}"]'
                )

    return "\n".join(lines)


def handle_page_fault(
    page_id: str, target_level: int, reason: str = "resolve_reference"
) -> dict:
    """Handle a page_fault tool call with v0.8 features."""
    if page_id not in PAGE_STORE:
        return {
            "error": f"Page '{page_id}' not found",
            "effects": {"promoted_to_working_set": False, "tokens_est": 0},
        }

    page = PAGE_STORE[page_id]
    modality = page["modality"]
    page_type = page.get("page_type", PageType.TRANSCRIPT)
    meta = page.get("meta", {})

    # Find best available level
    available_levels = sorted(page["levels"].keys())
    actual_level = target_level
    if target_level not in available_levels:
        actual_level = min(available_levels, key=lambda x: abs(x - target_level))

    content = page["levels"][actual_level]

    # Estimate tokens
    if isinstance(content, str):
        tokens_est = len(content) // 4
    elif isinstance(content, dict):
        tokens_est = len(json.dumps(content)) // 4
    else:
        tokens_est = 100

    # Check fault policy
    can_fault, deny_reason = vm_runtime.can_fault(page_id, tokens_est)
    if not can_fault:
        return {
            "error": deny_reason,
            "effects": {"promoted_to_working_set": False, "tokens_est": 0},
        }

    # Map reason string to enum
    fault_reason = FaultReason.RESOLVE_REFERENCE
    if reason == "user_requested":
        fault_reason = FaultReason.USER_REQUESTED_RECALL
    elif reason == "tool_prereq":
        fault_reason = FaultReason.TOOL_PREREQUISITE
    elif reason == "speculative":
        fault_reason = FaultReason.SPECULATIVE

    # Record the fault
    vm_runtime.record_fault(
        page_id=page_id,
        reason=fault_reason,
        tokens_loaded=tokens_est,
        latency_ms=15.0,  # Simulated
    )

    # Build content based on modality
    if modality == "text":
        content_obj = {"text": content}
    elif modality == "image":
        if actual_level <= 1:
            content_obj = {"url": content, "dimensions": meta.get("dimensions")}
        else:
            content_obj = {"caption": content, "dimensions": meta.get("dimensions")}
    elif modality == "audio":
        if actual_level == 0:
            content_obj = {
                "url": content,
                "duration_seconds": meta.get("duration_seconds"),
            }
        elif actual_level == 1 and isinstance(content, dict):
            content_obj = {
                "transcript": content.get("transcript", ""),
                "duration_seconds": meta.get("duration_seconds"),
            }
        else:
            content_obj = {
                "summary": content,
                "duration_seconds": meta.get("duration_seconds"),
            }
    elif modality == "structured":
        if actual_level == 0:
            content_obj = {"data": content, "schema": meta.get("schema")}
        else:
            content_obj = {"summary": content, "schema": meta.get("schema")}
    else:
        content_obj = {"data": content}

    return {
        "page": {
            "page_id": page_id,
            "modality": modality,
            "page_type": page_type.value,
            "level": actual_level,
            "tier": "L0",  # Now in working set
            "content": content_obj,
            "provenance": page.get("provenance", []),
            "meta": {
                "source_tier": page["tier"],
                "available_levels": available_levels,
                "pinned": vm_runtime.pinned_set.is_pinned(page_id),
            },
        },
        "effects": {
            "promoted_to_working_set": True,
            "tokens_est": tokens_est,
            "fault_reason": fault_reason.value,
            "evictions": None,
        },
    }


def handle_search_pages(
    query: str,
    modality: Optional[str] = None,
    page_type: Optional[str] = None,
    limit: int = 5,
) -> dict:
    """Handle a search_pages tool call with v0.8 page type filtering."""
    results = []
    query_lower = query.lower()

    for page_id, page in PAGE_STORE.items():
        if modality and page["modality"] != modality:
            continue
        if page_type:
            pt = page.get("page_type", PageType.TRANSCRIPT)
            if pt.value != page_type:
                continue

        content = page["levels"].get(2, page["levels"].get(0, ""))
        if isinstance(content, str) and (
            query_lower in content.lower() or query_lower in page_id.lower()
        ):
            hint = content[:50] + "..." if len(content) > 50 else content
            results.append(
                {
                    "page_id": page_id,
                    "modality": page["modality"],
                    "page_type": page.get("page_type", PageType.TRANSCRIPT).value,
                    "tier": page["tier"],
                    "levels": list(page["levels"].keys()),
                    "hint": hint,
                    "provenance": page.get("provenance", []),
                    "relevance": 0.8,
                }
            )

    results = sorted(results, key=lambda x: x["relevance"], reverse=True)[:limit]

    return {
        "results": results,
        "total_available": len(PAGE_STORE),
    }


def process_tool_calls(tool_calls: list) -> list[dict]:
    """Process tool calls and return results."""
    results = []
    for tool_call in tool_calls:
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        if func_name == "page_fault":
            result = handle_page_fault(
                args["page_id"],
                args.get("target_level", 2),
                args.get("reason", "resolve_reference"),
            )
        elif func_name == "search_pages":
            result = handle_search_pages(
                args["query"],
                args.get("modality"),
                args.get("page_type"),
                args.get("limit", 5),
            )
        else:
            result = {"error": f"Unknown tool: {func_name}"}

        results.append(
            {
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            }
        )

    return results


# VM Prompts with v0.8 page type awareness
VM_STRICT_PROMPT = """You are operating under STRICT Virtual Memory grounding rules.

Your ONLY valid sources of information are:
1) The content inside <VM:CONTEXT> (the currently mapped working set), and
2) The content returned by tools (e.g., page_fault) in messages with role="tool".

PAGE TYPES:
- transcript: Raw conversation turns
- summary: LLM-generated summaries (derived from transcripts)
- claim: Decisions, facts, conclusions (HIGH VALUE - prefer these!)
- procedure: Learned patterns for tool usage
- artifact: Tool-created content (code, diagrams, etc.)
- index: Page metadata for search

When you need information:
1) FIRST check for claim pages - they contain key decisions
2) Then check summaries for overview
3) Only load transcripts if you need exact wording

Faulting rules:
- Prefer claim and summary pages over raw transcripts
- Use reason="user_requested" when user explicitly asks
- Use reason="resolve_reference" when following a reference
- Do not exceed max_faults_per_turn from the manifest

Answering rules:
- Do not invent or fill gaps with assumptions
- If you cannot obtain required information, say: "I don't have that in the mapped context."
- When you use information, include inline citations like [ref: page_id]
- Pinned pages (marked with 📌) contain critical context

Never mention VM internals to the user unless explicitly asked."""

VM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "page_fault",
            "description": "Load a memory page into context at specified compression level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "ID of the page to load",
                    },
                    "target_level": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 3,
                        "default": 2,
                        "description": "0=full, 1=reduced, 2=abstract/summary, 3=reference only",
                    },
                    "reason": {
                        "type": "string",
                        "enum": [
                            "user_requested",
                            "resolve_reference",
                            "tool_prereq",
                            "speculative",
                        ],
                        "default": "resolve_reference",
                        "description": "Why this page is being loaded",
                    },
                },
                "required": ["page_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_pages",
            "description": "Search for pages matching a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "modality": {
                        "type": "string",
                        "enum": ["text", "image", "audio", "video", "structured"],
                    },
                    "page_type": {
                        "type": "string",
                        "enum": [
                            "transcript",
                            "summary",
                            "claim",
                            "procedure",
                            "artifact",
                            "index",
                        ],
                        "description": "Filter by page type (prefer 'claim' for decisions)",
                    },
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
]


async def run_openai_demo(
    user_query: str, model: str = "gpt-4o", show_internals: bool = False
):
    """Run demo with OpenAI."""
    try:
        from openai import OpenAI
    except ImportError:
        print("OpenAI not installed. Run: pip install openai")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set.")
        return

    client = OpenAI(api_key=api_key)

    # Initialize VM runtime
    vm_runtime.new_turn()
    vm_runtime.auto_pin_claims()

    # Build initial context with claim pages pinned
    working_set = ["msg_003", "claim_jwt_expiry"]  # Recent message + key claim
    vm_runtime.working_set = working_set.copy()
    all_pages = list(PAGE_STORE.keys())

    manifest = build_manifest(working_set, all_pages)
    context = build_context(working_set)

    developer_content = f"""<VM:RULES>
{VM_STRICT_PROMPT}
</VM:RULES>

<VM:MANIFEST_JSON>
{json.dumps(manifest, indent=2)}
</VM:MANIFEST_JSON>

<VM:CONTEXT>
{context}
</VM:CONTEXT>"""

    messages = [
        {"role": "developer", "content": developer_content},
        {"role": "user", "content": user_query},
    ]

    print(f"\n{'=' * 60}")
    print(f"USER: {user_query}")
    print(f"{'=' * 60}")
    print(f"\nWorking set: {working_set}")
    print(f"Pinned pages: {list(vm_runtime.pinned_set.get_all_pinned())}")

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=VM_TOOLS,
    )

    message = response.choices[0].message
    turn = 1

    while message.tool_calls:
        print(
            f"\n--- Turn {turn}: Model requested {len(message.tool_calls)} page fault(s) ---"
        )

        for tc in message.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"  → {tc.function.name}({args})")

        tool_results = process_tool_calls(message.tool_calls)

        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
        )

        for result in tool_results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": result["content"],
                }
            )
            loaded = json.loads(result["content"])
            if "page" in loaded:
                pg = loaded["page"]
                print(
                    f"  ← Loaded {pg['page_id']} (type={pg['page_type']}, level={pg['level']})"
                )

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=VM_TOOLS,
        )

        message = response.choices[0].message
        turn += 1
        vm_runtime.new_turn()

        if turn > 5:
            print("Max turns reached")
            break

    print(f"\n{'=' * 60}")
    print(f"ASSISTANT:\n{message.content}")
    print(f"{'=' * 60}")

    # Show v0.8 metrics
    if show_internals:
        print("\n--- VM Runtime Stats ---")
        stats = vm_runtime.get_stats()
        print(json.dumps(stats, indent=2))

        print("\n--- Mutation Log Summary ---")
        print(json.dumps(vm_runtime.mutation_log.get_summary(), indent=2))


async def run_anthropic_demo(
    user_query: str,
    model: str = "claude-sonnet-4-20250514",
    show_internals: bool = False,
):
    """Run demo with Anthropic."""
    try:
        import anthropic
    except ImportError:
        print("Anthropic not installed. Run: pip install anthropic")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set.")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # Initialize VM runtime
    vm_runtime.new_turn()
    vm_runtime.auto_pin_claims()

    working_set = ["msg_003", "claim_jwt_expiry"]
    vm_runtime.working_set = working_set.copy()
    all_pages = list(PAGE_STORE.keys())

    manifest = build_manifest(working_set, all_pages)
    context = build_context(working_set)

    system_content = f"""<VM:RULES>
{VM_STRICT_PROMPT}
</VM:RULES>

<VM:MANIFEST_JSON>
{json.dumps(manifest, indent=2)}
</VM:MANIFEST_JSON>

<VM:CONTEXT>
{context}
</VM:CONTEXT>"""

    anthropic_tools = [
        {
            "name": t["function"]["name"],
            "description": t["function"]["description"],
            "input_schema": t["function"]["parameters"],
        }
        for t in VM_TOOLS
    ]

    messages = [{"role": "user", "content": user_query}]

    print(f"\n{'=' * 60}")
    print(f"USER: {user_query}")
    print(f"{'=' * 60}")
    print(f"\nWorking set: {working_set}")
    print(f"Pinned pages: {list(vm_runtime.pinned_set.get_all_pinned())}")

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_content,
        tools=anthropic_tools,
        messages=messages,
    )

    turn = 1

    while response.stop_reason == "tool_use":
        print(f"\n--- Turn {turn}: Processing tool calls ---")

        tool_uses = [block for block in response.content if block.type == "tool_use"]
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tu in tool_uses:
            print(f"  → {tu.name}({tu.input})")

            if tu.name == "page_fault":
                result = handle_page_fault(
                    tu.input["page_id"],
                    tu.input.get("target_level", 2),
                    tu.input.get("reason", "resolve_reference"),
                )
            elif tu.name == "search_pages":
                result = handle_search_pages(
                    tu.input["query"],
                    tu.input.get("modality"),
                    tu.input.get("page_type"),
                    tu.input.get("limit", 5),
                )
            else:
                result = {"error": f"Unknown tool: {tu.name}"}

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps(result),
                }
            )

            if "page" in result:
                pg = result["page"]
                print(
                    f"  ← Loaded {pg['page_id']} (type={pg['page_type']}, level={pg['level']})"
                )

        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_content,
            tools=anthropic_tools,
            messages=messages,
        )

        turn += 1
        vm_runtime.new_turn()
        if turn > 5:
            break

    final_text = "".join(
        block.text for block in response.content if hasattr(block, "text")
    )

    print(f"\n{'=' * 60}")
    print(f"ASSISTANT:\n{final_text}")
    print(f"{'=' * 60}")

    if show_internals:
        print("\n--- VM Runtime Stats ---")
        print(json.dumps(vm_runtime.get_stats(), indent=2))


async def run_ollama_demo(
    user_query: str, model: str = "llama3.2", show_internals: bool = False
):
    """Run demo with Ollama (local)."""
    try:
        from openai import OpenAI
    except ImportError:
        print("OpenAI not installed. Run: pip install openai")
        return

    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

    vm_runtime.new_turn()
    vm_runtime.auto_pin_claims()

    working_set = ["msg_003", "claim_jwt_expiry"]
    vm_runtime.working_set = working_set.copy()
    all_pages = list(PAGE_STORE.keys())

    manifest = build_manifest(working_set, all_pages)
    context = build_context(working_set)

    system_content = f"""<VM:RULES>
{VM_STRICT_PROMPT}
</VM:RULES>

<VM:MANIFEST_JSON>
{json.dumps(manifest, indent=2)}
</VM:MANIFEST_JSON>

<VM:CONTEXT>
{context}
</VM:CONTEXT>"""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_query},
    ]

    print(f"\n{'=' * 60}")
    print(f"USER: {user_query}")
    print(f"{'=' * 60}")
    print(f"\nNote: Ollama tool support varies by model. Using {model}.")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=VM_TOOLS,
        )

        message = response.choices[0].message
        print(f"\n{'=' * 60}")
        print(f"ASSISTANT:\n{message.content}")
        print(f"{'=' * 60}")

        if show_internals:
            print("\n--- VM Runtime Stats ---")
            print(json.dumps(vm_runtime.get_stats(), indent=2))

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running: ollama serve")


def main():
    parser = argparse.ArgumentParser(description="AI Virtual Memory Demo (v0.8)")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "ollama"],
        default="openai",
        help="LLM provider to use",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to use (default varies by provider)",
    )
    parser.add_argument(
        "--query",
        default="What did we decide about JWT tokens? I need the specific recommendations.",
        help="User query to test",
    )
    parser.add_argument(
        "--show-internals",
        action="store_true",
        help="Show VM runtime internals after demo",
    )

    args = parser.parse_args()

    if args.model is None:
        args.model = {
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-20250514",
            "ollama": "llama3.2",
        }[args.provider]

    print(f"Running VM demo v0.8 with {args.provider} ({args.model})")
    print(f"Query: {args.query}")

    if args.provider == "openai":
        asyncio.run(run_openai_demo(args.query, args.model, args.show_internals))
    elif args.provider == "anthropic":
        asyncio.run(run_anthropic_demo(args.query, args.model, args.show_internals))
    elif args.provider == "ollama":
        asyncio.run(run_ollama_demo(args.query, args.model, args.show_internals))


if __name__ == "__main__":
    main()
