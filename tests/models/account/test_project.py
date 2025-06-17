# tests/test_project.py
import pytest
from datetime import datetime, timezone
from uuid import uuid4
import time

from a2a_accounts.models.project import Project, ProjectStatus
from a2a_accounts.models.access_levels import AccessLevel


def make_project(**kwargs):
    defaults = {
        'name': 'Test Project',
        'account_id': 'acct1'
    }
    defaults.update(kwargs)
    return Project(**defaults)


def test_minimal_project_creation():
    project = make_project()
    assert project.name == 'Test Project'
    assert project.account_id == 'acct1'
    assert project.description is None
    assert project.access_level == AccessLevel.PRIVATE
    assert isinstance(project.id, str)
    # New fields
    assert project.session_ids == []


def test_full_project_creation():
    project_id = str(uuid4())
    project = make_project(
        id=project_id,
        description='Desc',
        owner_id='owner1',
        access_level=AccessLevel.SHARED,
        shared_with={'u1', 'u2'},
        tags=['a', 'b'],
        status='archived',
        session_ids=['s1', 's2'],
    )
    assert project.id == project_id
    assert project.description == 'Desc'
    assert project.owner_id == 'owner1'
    assert project.access_level == AccessLevel.SHARED
    assert project.shared_with == {'u1', 'u2'}
    assert project.tags == ['a', 'b']
    assert project.status == 'archived'
    assert project.session_ids == ['s1', 's2']


def test_adding_and_removing_sessions():
    project = make_project()
    project.add_session(type('S', (), {'id': 'sess1'})())
    assert project.session_ids == ['sess1']
    # Adding duplicate has no effect
    project.add_session(type('S', (), {'id': 'sess1'})())
    assert project.session_ids == ['sess1']
    # Remove session
    project.remove_session(type('S', (), {'id': 'sess1'})())
    assert project.session_ids == []

@pytest.mark.parametrize("level,public,shared", [
    (AccessLevel.PRIVATE, False, False),
    (AccessLevel.PUBLIC, True, False),
    (AccessLevel.SHARED, False, False),
])
def test_public_and_shared_flags(level, public, shared):
    proj = make_project(access_level=level)
    assert proj.is_public is public
    assert proj.is_shared is shared


def test_has_access():
    proj = make_project(access_level=AccessLevel.PUBLIC)
    assert proj.has_access('any')
    proj = make_project(access_level=AccessLevel.PRIVATE)
    assert proj.has_access('acct1')
    assert not proj.has_access('other')
    proj = make_project(access_level=AccessLevel.SHARED, shared_with={'u1'})
    assert proj.has_access('acct1')
    assert proj.has_access('u1')
    assert not proj.has_access('u2')


def test_status_enum_values():
    assert ProjectStatus.ACTIVE.value == 'active'
    assert ProjectStatus.ARCHIVED.value == 'archived'
    assert ProjectStatus.DELETED.value == 'deleted'

@pytest.mark.parametrize("ids", [['s1'], ['s1', 's2']])
def test_serialization_sessions(ids):
    proj = make_project(session_ids=ids)
    data = proj.model_dump()
    assert data['session_ids'] == ids
    json_str = proj.model_dump_json()
    for sid in ids:
        assert sid in json_str
