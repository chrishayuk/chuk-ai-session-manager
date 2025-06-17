# a2a_accounts/models/account.py
from __future__ import annotations
from uuid import uuid4
from pydantic import BaseModel, Field
from typing import List

class Account(BaseModel):
    """Represents an Account which owns Projects."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    owner_user_id: str
    project_ids: List[str] = Field(default_factory=list)

    def add_project(self, project: Any) -> None:
        """Associate a Project with this Account."""
        if project.id not in self.project_ids:
            self.project_ids.append(project.id)

    def remove_project(self, project: Any) -> None:
        """Disassociate a Project from this Account."""
        if project.id in self.project_ids:
            self.project_ids.remove(project.id)
