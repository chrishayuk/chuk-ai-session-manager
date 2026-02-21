# chuk_ai_session_manager/guards/constants.py
"""Shared constants for the guards subsystem.

Centralizes magic strings, patterns, and constants to avoid duplication.
"""

from __future__ import annotations

import re

# Reference pattern: $v1, $v2, ${v1}, ${myalias}
REFERENCE_PATTERN = re.compile(r"\$\{?([a-zA-Z_][a-zA-Z0-9_]*|v\d+)\}?")

# Reference prefix for string matching
REFERENCE_PREFIX = "$v"

# Binding sources (used in extract_bindings_from_text)
BINDING_SOURCE_ASSISTANT_TEXT = "assistant_text"
BINDING_SOURCE_EXTRACTED = "extracted"
