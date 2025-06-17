# a2a_accounts/models/project.py
from __future__ import annotations
from enum import Enum
from uuid import uuid4
from pydantic import Field
from typing import Any, List, Optional

# imports
from a2a_accounts.models.access_control import AccessControlled

class Project(AccessControlled):
    """A project that contains Sessions and is owned by an Account."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    owner_id: Optional[str] = None  # could be a user or system owner
    status: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    session_ids: List[str] = Field(default_factory=list)

    def add_session(self, session: Any) -> None:
        """Associate a Session with this Project."""
        if session.id not in self.session_ids:
            self.session_ids.append(session.id)

    def remove_session(self, session: Any) -> None:
        """Disassociate a Session from this Project."""
        if session.id in self.session_ids:
            self.session_ids.remove(session.id)

class ProjectStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
