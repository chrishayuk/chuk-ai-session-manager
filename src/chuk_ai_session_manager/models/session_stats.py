# chuk_ai_session_manager/models/session_stats.py
"""Session statistics model."""

from __future__ import annotations

from chuk_ai_session_manager.base_models import DictCompatModel


class SessionStats(DictCompatModel):
    """Statistics for a session or session chain."""

    session_id: str = ""
    session_segments: int = 1
    session_chain: list[str] | None = None
    total_messages: int = 0
    total_events: int = 0
    user_messages: int = 0
    ai_messages: int = 0
    tool_calls: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    created_at: str = ""
    last_update: str = ""
    infinite_context: bool = False
