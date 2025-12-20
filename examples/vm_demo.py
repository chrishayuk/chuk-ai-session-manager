#!/usr/bin/env python3
"""
AI Virtual Memory Demo

Demonstrates the VM protocol with a real LLM. Shows:
- VM:MANIFEST_JSON generation
- VM:CONTEXT packing
- page_fault tool handling
- search_pages tool handling
- Strict mode grounding

Usage:
    # With OpenAI
    export OPENAI_API_KEY=your-key
    python examples/vm_demo.py

    # With Anthropic
    export ANTHROPIC_API_KEY=your-key
    python examples/vm_demo.py --provider anthropic

    # With local model (Ollama)
    python examples/vm_demo.py --provider ollama --model llama3.2

    # Or create a .env file with your API keys
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

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

# Simulated page storage (in real implementation, this comes from chuk-artifacts)
PAGE_STORE = {
    # === TEXT PAGES ===
    "msg_001": {
        "modality": "text",
        "levels": {
            0: "The user asked about implementing a REST API with authentication. They mentioned wanting to use JWT tokens and needed advice on refresh token strategies.",
            2: "User asked about REST API + JWT auth implementation.",
        },
        "tier": "L2",
        "meta": {"word_count": 28},
    },
    "msg_002": {
        "modality": "text",
        "levels": {
            0: "I explained that JWT tokens should have short expiration times (15-30 minutes) and that refresh tokens should be stored securely, rotated on use, and have longer expiration (7-30 days). I recommended using HTTP-only cookies for refresh tokens.",
            2: "Explained JWT best practices: short-lived access tokens, secure refresh token storage, rotation.",
        },
        "tier": "L2",
        "meta": {"word_count": 42},
    },
    "msg_003": {
        "modality": "text",
        "levels": {
            0: "The user then asked about rate limiting strategies for their API endpoints.",
            2: "User asked about API rate limiting.",
        },
        "tier": "L2",
        "meta": {"word_count": 12},
    },
    "summary_seg_01": {
        "modality": "text",
        "levels": {
            2: "Segment 1 covered: REST API design discussion, JWT authentication strategy, refresh token handling, rate limiting considerations.",
        },
        "tier": "L2",
        "meta": {},
    },
    "code_snippet_001": {
        "modality": "text",
        "levels": {
            0: '''```python
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
```''',
            2: "FastAPI JWT verification code snippet with OAuth2 bearer scheme.",
        },
        "tier": "L3",
        "meta": {"language": "python", "lines": 14},
    },

    # === IMAGE PAGES ===
    "img_architecture": {
        "modality": "image",
        "levels": {
            0: "https://example.com/images/jwt_auth_architecture_full.png",  # Full resolution URL
            1: "https://example.com/images/jwt_auth_architecture_thumb.png",  # Thumbnail
            2: "Architecture diagram showing JWT authentication flow: Client → API Gateway → Auth Service → Token Validation → Protected Resource. Shows request/response arrows with token payloads labeled.",
        },
        "tier": "L3",
        "meta": {
            "dimensions": [1200, 800],
            "mime_type": "image/png",
            "size_bytes": 245000,
            "alt_text": "JWT authentication flow diagram",
        },
    },
    "img_database_schema": {
        "modality": "image",
        "levels": {
            0: "https://example.com/images/user_db_schema_full.png",
            1: "https://example.com/images/user_db_schema_thumb.png",
            2: "Database schema diagram showing Users table (id, email, password_hash, created_at), RefreshTokens table (id, user_id, token_hash, expires_at, revoked), and Sessions table (id, user_id, device_info, last_active).",
        },
        "tier": "L3",
        "meta": {
            "dimensions": [1000, 600],
            "mime_type": "image/png",
            "size_bytes": 180000,
            "alt_text": "User database schema with auth tables",
        },
    },
    "img_performance_chart": {
        "modality": "image",
        "levels": {
            0: "https://example.com/images/api_latency_chart_full.png",
            1: "https://example.com/images/api_latency_chart_thumb.png",
            2: "Line chart showing API response times over 24 hours. P50 latency: 45ms, P95: 120ms, P99: 350ms. Notable spike at 14:00 UTC reaching 800ms during deployment.",
        },
        "tier": "L3",
        "meta": {
            "dimensions": [800, 400],
            "mime_type": "image/png",
            "size_bytes": 95000,
            "chart_type": "line",
            "data_points": 288,
        },
    },

    # === AUDIO PAGES ===
    "audio_meeting_001": {
        "modality": "audio",
        "levels": {
            0: "https://example.com/audio/team_meeting_2024_01_15.mp3",  # Full audio
            1: {  # Transcript with timestamps
                "transcript": """[00:00] Alex: Let's discuss the authentication implementation.
[00:15] Jordan: I've been looking at JWT vs session-based auth.
[01:30] Alex: What are the trade-offs you've found?
[02:00] Jordan: JWT is stateless, great for microservices, but token revocation is tricky.
[03:45] Alex: We could use short-lived tokens with refresh rotation.
[04:30] Jordan: That's what I was thinking. Redis for the refresh token blacklist?
[05:15] Alex: Yes, with a TTL matching the refresh token expiry.
[06:00] Jordan: I'll draft the implementation. Meeting adjourned.""",
                "timestamps": [
                    {"time": 0, "speaker": "Alex", "text": "Let's discuss the authentication implementation."},
                    {"time": 15, "speaker": "Jordan", "text": "I've been looking at JWT vs session-based auth."},
                    {"time": 90, "speaker": "Alex", "text": "What are the trade-offs you've found?"},
                    {"time": 120, "speaker": "Jordan", "text": "JWT is stateless, great for microservices, but token revocation is tricky."},
                    {"time": 225, "speaker": "Alex", "text": "We could use short-lived tokens with refresh rotation."},
                    {"time": 270, "speaker": "Jordan", "text": "That's what I was thinking. Redis for the refresh token blacklist?"},
                    {"time": 315, "speaker": "Alex", "text": "Yes, with a TTL matching the refresh token expiry."},
                    {"time": 360, "speaker": "Jordan", "text": "I'll draft the implementation. Meeting adjourned."},
                ],
            },
            2: "Team meeting (6 min): Alex and Jordan discussed JWT vs session auth. Decided on short-lived JWT + refresh tokens with Redis blacklist for revocation.",
        },
        "tier": "L4",
        "meta": {
            "duration_seconds": 375,
            "speakers": ["Alex", "Jordan"],
            "mime_type": "audio/mpeg",
            "size_bytes": 5600000,
            "recorded_at": "2024-01-15T10:00:00Z",
        },
    },
    "audio_standup_002": {
        "modality": "audio",
        "levels": {
            0: "https://example.com/audio/standup_2024_01_16.mp3",
            1: {
                "transcript": """[00:00] Jordan: Quick update - JWT implementation is 80% done.
[00:20] Jordan: Blocked on the Redis connection pooling config.
[00:45] Alex: I can help with that after this call.
[01:00] Jordan: Great, should be done by EOD then.""",
                "timestamps": [
                    {"time": 0, "speaker": "Jordan", "text": "Quick update - JWT implementation is 80% done."},
                    {"time": 20, "speaker": "Jordan", "text": "Blocked on the Redis connection pooling config."},
                    {"time": 45, "speaker": "Alex", "text": "I can help with that after this call."},
                    {"time": 60, "speaker": "Jordan", "text": "Great, should be done by EOD then."},
                ],
            },
            2: "Standup (1 min): Jordan reports JWT 80% done, blocked on Redis config. Alex to assist.",
        },
        "tier": "L4",
        "meta": {
            "duration_seconds": 72,
            "speakers": ["Jordan", "Alex"],
            "mime_type": "audio/mpeg",
            "size_bytes": 1080000,
            "recorded_at": "2024-01-16T09:00:00Z",
        },
    },

    # === VIDEO PAGES ===
    "video_demo_001": {
        "modality": "video",
        "levels": {
            0: "https://example.com/video/auth_demo_full.mp4",  # Full video
            1: {  # Keyframes + transcript
                "scenes": [
                    {"timestamp": 0, "description": "Title screen: 'JWT Authentication Demo'", "keyframe_url": "https://example.com/video/auth_demo/frame_0.jpg"},
                    {"timestamp": 15, "description": "Browser showing login form with email/password fields", "keyframe_url": "https://example.com/video/auth_demo/frame_15.jpg"},
                    {"timestamp": 45, "description": "Network tab showing POST /auth/login request", "keyframe_url": "https://example.com/video/auth_demo/frame_45.jpg"},
                    {"timestamp": 60, "description": "Response payload with access_token and refresh_token visible", "keyframe_url": "https://example.com/video/auth_demo/frame_60.jpg"},
                    {"timestamp": 90, "description": "Subsequent API call with Authorization: Bearer header", "keyframe_url": "https://example.com/video/auth_demo/frame_90.jpg"},
                    {"timestamp": 120, "description": "Token expiry simulation - 401 response shown", "keyframe_url": "https://example.com/video/auth_demo/frame_120.jpg"},
                    {"timestamp": 150, "description": "Automatic token refresh via /auth/refresh endpoint", "keyframe_url": "https://example.com/video/auth_demo/frame_150.jpg"},
                    {"timestamp": 180, "description": "End screen with 'Implementation Complete' message", "keyframe_url": "https://example.com/video/auth_demo/frame_180.jpg"},
                ],
                "transcript": "This demo shows our JWT authentication flow. First, user logs in... token is returned... used in subsequent requests... automatic refresh when expired...",
            },
            2: "Demo video (3 min): Shows complete JWT auth flow - login, token usage, expiry handling, and automatic refresh. 8 key scenes from login form to successful token refresh.",
        },
        "tier": "L4",
        "meta": {
            "duration_seconds": 185,
            "dimensions": [1920, 1080],
            "mime_type": "video/mp4",
            "size_bytes": 45000000,
            "fps": 30,
            "recorded_at": "2024-01-17T14:00:00Z",
        },
    },
    "video_architecture_walkthrough": {
        "modality": "video",
        "levels": {
            0: "https://example.com/video/arch_walkthrough_full.mp4",
            1: {
                "scenes": [
                    {"timestamp": 0, "description": "Whiteboard with system overview diagram", "keyframe_url": "https://example.com/video/arch/frame_0.jpg"},
                    {"timestamp": 60, "description": "Zoomed in on API Gateway component", "keyframe_url": "https://example.com/video/arch/frame_60.jpg"},
                    {"timestamp": 180, "description": "Auth service microservice details", "keyframe_url": "https://example.com/video/arch/frame_180.jpg"},
                    {"timestamp": 300, "description": "Database layer and Redis cache", "keyframe_url": "https://example.com/video/arch/frame_300.jpg"},
                    {"timestamp": 420, "description": "Request flow animation", "keyframe_url": "https://example.com/video/arch/frame_420.jpg"},
                    {"timestamp": 540, "description": "Scaling considerations diagram", "keyframe_url": "https://example.com/video/arch/frame_540.jpg"},
                ],
                "transcript": "Let me walk through our architecture... starting with the API gateway... auth service handles all token operations... Redis for caching and blacklists... here's how a request flows through...",
            },
            2: "Architecture walkthrough (10 min): Detailed explanation of microservices auth system. Covers API gateway, auth service, database layer, Redis caching, and scaling approach.",
        },
        "tier": "L4",
        "meta": {
            "duration_seconds": 600,
            "dimensions": [1920, 1080],
            "mime_type": "video/mp4",
            "size_bytes": 150000000,
            "fps": 30,
            "recorded_at": "2024-01-18T11:00:00Z",
        },
    },

    # === STRUCTURED DATA PAGES ===
    "data_api_metrics": {
        "modality": "structured",
        "levels": {
            0: {
                "period": "2024-01-15 to 2024-01-21",
                "endpoints": {
                    "/auth/login": {"requests": 45230, "p50_ms": 120, "p95_ms": 340, "p99_ms": 890, "error_rate": 0.02},
                    "/auth/refresh": {"requests": 12450, "p50_ms": 45, "p95_ms": 95, "p99_ms": 210, "error_rate": 0.001},
                    "/api/users": {"requests": 89340, "p50_ms": 35, "p95_ms": 80, "p99_ms": 150, "error_rate": 0.005},
                    "/api/orders": {"requests": 156780, "p50_ms": 55, "p95_ms": 120, "p99_ms": 280, "error_rate": 0.008},
                },
                "total_requests": 303800,
                "availability": 99.94,
            },
            2: "API metrics (Jan 15-21): 303K total requests, 99.94% availability. Login endpoint slowest (p95: 340ms). Refresh endpoint fastest and most reliable (0.1% errors).",
        },
        "tier": "L3",
        "meta": {
            "schema": "api_metrics_v1",
            "generated_at": "2024-01-22T00:00:00Z",
        },
    },
}


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
    """Build VM:MANIFEST_JSON from page store."""
    working_set = []
    for page_id in working_set_ids:
        if page_id in PAGE_STORE:
            page = PAGE_STORE[page_id]
            # Estimate tokens (rough: 4 chars per token)
            content = page["levels"].get(0, page["levels"].get(2, ""))
            if isinstance(content, str):
                tokens_est = len(content) // 4
            else:
                tokens_est = len(str(content)) // 4
            working_set.append({
                "page_id": page_id,
                "modality": page["modality"],
                "level": 0 if 0 in page["levels"] else 2,
                "tokens_est": tokens_est,
                "importance": 0.8,
            })

    available_pages = []
    for page_id in all_page_ids:
        if page_id not in working_set_ids and page_id in PAGE_STORE:
            page = PAGE_STORE[page_id]
            modality = page["modality"]
            meta = page.get("meta", {})

            # Get summary-level content for hint
            hint_content = page["levels"].get(2, page["levels"].get(0, ""))
            if isinstance(hint_content, str):
                hint = hint_content[:80] + "..." if len(hint_content) > 80 else hint_content
            else:
                hint = str(hint_content)[:80]

            # Build level descriptions for this page
            level_hints = get_level_hints(modality)
            levels_available = sorted(page["levels"].keys())
            level_info = {lvl: level_hints.get(lvl, f"level {lvl}") for lvl in levels_available}

            # Add modality-specific metadata hints
            extra_info = {}
            if modality == "audio":
                duration = meta.get("duration_seconds", 0)
                mins, secs = divmod(duration, 60)
                extra_info["duration"] = f"{int(mins)}:{int(secs):02d}"
                extra_info["speakers"] = meta.get("speakers", [])
            elif modality == "video":
                duration = meta.get("duration_seconds", 0)
                mins, secs = divmod(duration, 60)
                extra_info["duration"] = f"{int(mins)}:{int(secs):02d}"
                if 1 in page["levels"] and isinstance(page["levels"][1], dict):
                    extra_info["scene_count"] = len(page["levels"][1].get("scenes", []))
            elif modality == "image":
                if "dimensions" in meta:
                    extra_info["dimensions"] = meta["dimensions"]
            elif modality == "structured":
                if "schema" in meta:
                    extra_info["schema"] = meta["schema"]

            available_pages.append({
                "page_id": page_id,
                "modality": modality,
                "tier": page["tier"],
                "levels": level_info,  # Now includes what each level provides
                "hint": hint,
                **extra_info,
            })

    return {
        "session_id": "demo_session_001",
        "working_set": working_set,
        "available_pages": available_pages,
        "policies": {
            "faults_allowed": True,
            "max_faults_per_turn": 2,
            "upgrade_budget_tokens": 4096,
            "prefer_levels": [2, 1, 0],
            "level_guidance": "Use level 2 for summaries/overviews, level 1 for details/timestamps/transcripts, level 0 for full content/URLs",
        },
    }


def build_context(working_set_ids: list[str]) -> str:
    """Build VM:CONTEXT from working set."""
    lines = []
    for page_id in working_set_ids:
        if page_id in PAGE_STORE:
            page = PAGE_STORE[page_id]
            modality = page["modality"]
            meta = page.get("meta", {})

            # Determine prefix based on modality
            prefix_map = {
                "text": "T",
                "image": "I",
                "audio": "D",  # 'D' for auDio (A is taken)
                "video": "V",
                "structured": "X",
            }

            # Override for specific text types
            if page_id.startswith("msg_"):
                idx = int(page_id.split("_")[1])
                prefix = "U" if idx % 2 == 1 else "A"
            elif page_id.startswith("summary_"):
                prefix = "S"
            elif page_id.startswith("code_"):
                prefix = "C"
            else:
                prefix = prefix_map.get(modality, "?")

            # Build display based on modality
            if modality == "text":
                content = page["levels"].get(0, page["levels"].get(2, ""))
                if isinstance(content, str):
                    display = content[:200] + "..." if len(content) > 200 else content
                else:
                    display = str(content)[:200]
                lines.append(f'{prefix} ({page_id}): "{display}"')

            elif modality == "image":
                caption = page["levels"].get(2, "No caption")
                dims = meta.get("dimensions", [0, 0])
                lines.append(f'{prefix} ({page_id}): [IMAGE: {dims[0]}x{dims[1]}, "{caption[:100]}..."]')

            elif modality == "audio":
                summary = page["levels"].get(2, "No summary")
                duration = meta.get("duration_seconds", 0)
                speakers = meta.get("speakers", [])
                mins = duration // 60
                secs = duration % 60
                lines.append(f'{prefix} ({page_id}): [AUDIO: {mins}:{secs:02d}, speakers: {", ".join(speakers)}, "{summary[:80]}..."]')

            elif modality == "video":
                summary = page["levels"].get(2, "No summary")
                duration = meta.get("duration_seconds", 0)
                dims = meta.get("dimensions", [0, 0])
                mins = duration // 60
                secs = duration % 60
                lines.append(f'{prefix} ({page_id}): [VIDEO: {mins}:{secs:02d}, {dims[0]}x{dims[1]}, "{summary[:80]}..."]')

            elif modality == "structured":
                summary = page["levels"].get(2, "No summary")
                schema = meta.get("schema", "unknown")
                lines.append(f'{prefix} ({page_id}): [DATA: schema={schema}, "{summary[:100]}..."]')

            else:
                lines.append(f'{prefix} ({page_id}): [UNKNOWN MODALITY: {modality}]')

    return "\n".join(lines)


def handle_page_fault(page_id: str, target_level: int) -> dict:
    """Handle a page_fault tool call."""
    if page_id not in PAGE_STORE:
        return {
            "error": f"Page '{page_id}' not found",
            "effects": {"promoted_to_working_set": False, "tokens_est": 0},
        }

    page = PAGE_STORE[page_id]
    modality = page["modality"]
    meta = page.get("meta", {})

    # Find best available level
    available_levels = sorted(page["levels"].keys())
    actual_level = target_level
    if target_level not in available_levels:
        # Find closest available level
        actual_level = min(available_levels, key=lambda x: abs(x - target_level))

    content = page["levels"][actual_level]

    # Estimate tokens based on content type
    if isinstance(content, str):
        tokens_est = len(content) // 4
    elif isinstance(content, dict):
        tokens_est = len(json.dumps(content)) // 4
    else:
        tokens_est = 100  # Default for complex objects

    # Build content based on modality
    if modality == "text":
        content_obj = {"text": content}

    elif modality == "image":
        if actual_level == 0:
            # Full resolution - return URL
            content_obj = {
                "url": content,
                "dimensions": meta.get("dimensions"),
                "alt_text": meta.get("alt_text", ""),
            }
        elif actual_level == 1:
            # Thumbnail
            content_obj = {
                "thumbnail_url": content,
                "dimensions": meta.get("dimensions"),
            }
        else:
            # Caption only (level 2+)
            content_obj = {
                "caption": content,
                "dimensions": meta.get("dimensions"),
            }

    elif modality == "audio":
        # For audio, model can't "listen" - always include transcript when available
        transcript_data = page["levels"].get(1, {})
        if isinstance(transcript_data, dict):
            full_transcript = transcript_data.get("transcript", "")
            timestamps = transcript_data.get("timestamps", [])
        else:
            full_transcript = ""
            timestamps = []

        if actual_level == 0:
            # Full audio - URL plus full transcript
            content_obj = {
                "url": content,
                "transcript": full_transcript,
                "timestamps": timestamps,
                "duration_seconds": meta.get("duration_seconds"),
                "speakers": meta.get("speakers", []),
                "note": "Full audio available at URL. Transcript provided for text analysis.",
            }
        elif actual_level == 1:
            # Transcript with timestamps (no URL)
            content_obj = {
                "transcript": full_transcript,
                "timestamps": timestamps,
                "duration_seconds": meta.get("duration_seconds"),
                "speakers": meta.get("speakers", []),
            }
        else:
            # Summary only (level 2+)
            content_obj = {
                "summary": content,
                "duration_seconds": meta.get("duration_seconds"),
                "speakers": meta.get("speakers", []),
            }

    elif modality == "video":
        if actual_level == 0:
            # Full video URL
            content_obj = {
                "url": content,
                "duration_seconds": meta.get("duration_seconds"),
                "dimensions": meta.get("dimensions"),
            }
        elif actual_level == 1:
            # Keyframes + transcript
            if isinstance(content, dict):
                content_obj = {
                    "scenes": content.get("scenes", []),
                    "transcript": content.get("transcript", ""),
                    "duration_seconds": meta.get("duration_seconds"),
                }
            else:
                content_obj = {"description": content}
        else:
            # Summary only (level 2+)
            content_obj = {
                "summary": content,
                "duration_seconds": meta.get("duration_seconds"),
                "scene_count": len(page["levels"].get(1, {}).get("scenes", [])) if isinstance(page["levels"].get(1), dict) else 0,
            }

    elif modality == "structured":
        if actual_level == 0:
            # Full data
            content_obj = {
                "data": content,
                "schema": meta.get("schema"),
            }
        else:
            # Summary
            content_obj = {
                "summary": content,
                "schema": meta.get("schema"),
            }

    else:
        content_obj = {"data": content}

    return {
        "page": {
            "page_id": page_id,
            "modality": modality,
            "level": actual_level,
            "tier": "L1",  # Now in working set
            "content": content_obj,
            "meta": {
                "source_tier": page["tier"],
                "available_levels": available_levels,
                **{k: v for k, v in meta.items() if k not in ["source_tier", "available_levels"]},
            },
        },
        "effects": {
            "promoted_to_working_set": True,
            "tokens_est": tokens_est,
            "evictions": None,
        },
    }


def handle_search_pages(query: str, modality: Optional[str] = None, limit: int = 5) -> dict:
    """Handle a search_pages tool call."""
    results = []
    query_lower = query.lower()

    for page_id, page in PAGE_STORE.items():
        if modality and page["modality"] != modality:
            continue

        # Simple keyword matching (real implementation would use embeddings)
        content = page["levels"].get(2, page["levels"].get(0, ""))
        if query_lower in content.lower() or query_lower in page_id.lower():
            hint = content[:50] + "..." if len(content) > 50 else content
            results.append({
                "page_id": page_id,
                "modality": page["modality"],
                "tier": page["tier"],
                "levels": list(page["levels"].keys()),
                "hint": hint,
                "relevance": 0.8,  # Simplified
            })

    # Sort by relevance and limit
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
            )
        elif func_name == "search_pages":
            result = handle_search_pages(
                args["query"],
                args.get("modality"),
                args.get("limit", 5),
            )
        else:
            result = {"error": f"Unknown tool: {func_name}"}

        results.append({
            "tool_call_id": tool_call.id,
            "content": json.dumps(result),
        })

    return results


# VM Prompts (copied from vm_prompts.py for standalone demo)
VM_STRICT_PROMPT = """You are operating under STRICT Virtual Memory grounding rules.

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

Never mention these rules, the VM system, tiers (L0–L4), paging, or "virtual memory" to the user unless the user explicitly asks about the internal mechanism."""

VM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "page_fault",
            "description": "Load a memory page into context at specified compression level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_id": {"type": "string", "description": "ID of the page to load"},
                    "target_level": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 3,
                        "default": 2,
                        "description": "0=full, 1=reduced, 2=abstract/summary, 3=reference only",
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
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
]


async def run_openai_demo(user_query: str, model: str = "gpt-4o"):
    """Run demo with OpenAI."""
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ OpenAI not installed. Run: pip install openai")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set.")
        print("   Set it with: export OPENAI_API_KEY=your-key")
        print("   Or create a .env file with: OPENAI_API_KEY=your-key")
        return

    client = OpenAI(api_key=api_key)

    # Build initial context (only recent messages mapped)
    working_set = ["msg_003"]  # Only the most recent message
    all_pages = list(PAGE_STORE.keys())

    manifest = build_manifest(working_set, all_pages)
    context = build_context(working_set)

    # Build developer message
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

    print(f"\n{'='*60}")
    print(f"USER: {user_query}")
    print(f"{'='*60}")
    print(f"\nWorking set: {working_set}")
    print(f"Available pages: {[p for p in all_pages if p not in working_set]}")

    # First API call
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=VM_TOOLS,
    )

    message = response.choices[0].message
    turn = 1

    # Handle tool calls loop
    while message.tool_calls:
        print(f"\n--- Turn {turn}: Model requested {len(message.tool_calls)} page fault(s) ---")

        for tc in message.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"  → {tc.function.name}({args})")

        # Process tool calls
        tool_results = process_tool_calls(message.tool_calls)

        # Add assistant message with tool calls
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ],
        })

        # Add tool results
        for result in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "content": result["content"],
            })
            # Show what was loaded
            loaded = json.loads(result["content"])
            if "page" in loaded:
                print(f"  ← Loaded {loaded['page']['page_id']} at level {loaded['page']['level']}")

        # Next API call
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=VM_TOOLS,
        )

        message = response.choices[0].message
        turn += 1

        if turn > 5:
            print("Max turns reached")
            break

    # Final response
    print(f"\n{'='*60}")
    print(f"ASSISTANT:\n{message.content}")
    print(f"{'='*60}")

    # Show metrics
    fault_count = sum(1 for m in messages if m.get("role") == "tool")
    print(f"\nMetrics: {fault_count} page fault(s), {turn} turn(s)")


async def run_anthropic_demo(user_query: str, model: str = "claude-sonnet-4-20250514"):
    """Run demo with Anthropic."""
    try:
        import anthropic
    except ImportError:
        print("❌ Anthropic not installed. Run: pip install anthropic")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set.")
        print("   Set it with: export ANTHROPIC_API_KEY=your-key")
        print("   Or create a .env file with: ANTHROPIC_API_KEY=your-key")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # Build initial context
    working_set = ["msg_003"]
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

    # Convert tools to Anthropic format
    anthropic_tools = [
        {
            "name": t["function"]["name"],
            "description": t["function"]["description"],
            "input_schema": t["function"]["parameters"],
        }
        for t in VM_TOOLS
    ]

    messages = [{"role": "user", "content": user_query}]

    print(f"\n{'='*60}")
    print(f"USER: {user_query}")
    print(f"{'='*60}")

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

        # Extract tool uses
        tool_uses = [block for block in response.content if block.type == "tool_use"]

        # Build assistant message
        messages.append({"role": "assistant", "content": response.content})

        # Process each tool use
        tool_results = []
        for tu in tool_uses:
            print(f"  → {tu.name}({tu.input})")

            if tu.name == "page_fault":
                result = handle_page_fault(tu.input["page_id"], tu.input.get("target_level", 2))
            elif tu.name == "search_pages":
                result = handle_search_pages(
                    tu.input["query"],
                    tu.input.get("modality"),
                    tu.input.get("limit", 5),
                )
            else:
                result = {"error": f"Unknown tool: {tu.name}"}

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(result),
            })

            if "page" in result:
                print(f"  ← Loaded {result['page']['page_id']} at level {result['page']['level']}")

        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_content,
            tools=anthropic_tools,
            messages=messages,
        )

        turn += 1
        if turn > 5:
            break

    # Extract final text
    final_text = "".join(
        block.text for block in response.content if hasattr(block, "text")
    )

    print(f"\n{'='*60}")
    print(f"ASSISTANT:\n{final_text}")
    print(f"{'='*60}")


async def run_ollama_demo(user_query: str, model: str = "llama3.2"):
    """Run demo with Ollama (local)."""
    try:
        from openai import OpenAI
    except ImportError:
        print("OpenAI not installed. Run: pip install openai")
        return

    # Ollama exposes OpenAI-compatible API
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # Required but not used
    )

    # Same logic as OpenAI demo
    working_set = ["msg_003"]
    all_pages = list(PAGE_STORE.keys())

    manifest = build_manifest(working_set, all_pages)
    context = build_context(working_set)

    # For Ollama, use system role instead of developer
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

    print(f"\n{'='*60}")
    print(f"USER: {user_query}")
    print(f"{'='*60}")
    print(f"\nNote: Ollama tool support varies by model. Using {model}.")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=VM_TOOLS,
        )

        message = response.choices[0].message
        print(f"\n{'='*60}")
        print(f"ASSISTANT:\n{message.content}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running: ollama serve")


def main():
    parser = argparse.ArgumentParser(description="AI Virtual Memory Demo")
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
        default="What did we discuss about JWT tokens earlier? I need the specific recommendations.",
        help="User query to test",
    )

    args = parser.parse_args()

    # Set default models
    if args.model is None:
        args.model = {
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-20250514",
            "ollama": "llama3.2",
        }[args.provider]

    print(f"Running VM demo with {args.provider} ({args.model})")
    print(f"Query: {args.query}")

    if args.provider == "openai":
        asyncio.run(run_openai_demo(args.query, args.model))
    elif args.provider == "anthropic":
        asyncio.run(run_anthropic_demo(args.query, args.model))
    elif args.provider == "ollama":
        asyncio.run(run_ollama_demo(args.query, args.model))


if __name__ == "__main__":
    main()
