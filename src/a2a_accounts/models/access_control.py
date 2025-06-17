# a2a_accounts/models/access_control.py
from __future__ import annotations
from typing import Any, Dict, Set
from uuid import uuid4
from pydantic import BaseModel, Field

# session mananager imports
from a2a_accounts.models.access_levels import AccessLevel

class AccessControlled(BaseModel):
    """Mixin for access control."""
    account_id: str
    access_level: AccessLevel = AccessLevel.PRIVATE
    shared_with: Set[str] = Field(default_factory=set)

    @property
    def is_public(self) -> bool:
        return self.access_level == AccessLevel.PUBLIC

    @property
    def is_shared(self) -> bool:
        return self.access_level == AccessLevel.SHARED and bool(self.shared_with)

    def has_access(self, account_id: str) -> bool:
        """Return True if the given account_id may access this resource."""
        if self.is_public:
            return True
        if account_id == self.account_id:
            return True
        if self.is_shared and account_id in self.shared_with:
            return True
        return False