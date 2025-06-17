# a2a_accounts/models/access_levels.py
"""
Access levels
"""
from __future__ import annotations
from enum import Enum


class AccessLevel(str, Enum):
    """Access level for sessions and projects."""
    PRIVATE = "private"
    PUBLIC = "public"
    SHARED = "shared"