# chuk_ai_session_manager/base_models.py
"""Base model with backward-compatible dict-style access."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class DictCompatModel(BaseModel):
    """Base for models that support dict-style access for backward compatibility.

    Allows ``obj["key"]`` and ``"key" in obj`` so callers that previously
    consumed raw dicts continue to work without changes.
    """

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            return key in self.model_fields
        return False

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            return self.model_dump() == other
        return super().__eq__(other)
