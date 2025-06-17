# tests/test_access_control.py
import pytest
from a2a_accounts.models.access_control import AccessControlled
from a2a_accounts.models.access_levels import AccessLevel


def make_control(account_id="owner_acc", access_level=AccessLevel.PRIVATE, shared_with=None):
    shared = set(shared_with) if shared_with is not None else set()
    return AccessControlled(account_id=account_id, access_level=access_level, shared_with=shared)


def test_default_private_access():
    ctrl = make_control()
    # Owner always has access
    assert ctrl.has_access("owner_acc") is True
    # Other accounts have no access
    assert ctrl.has_access("other_acc") is False
    # Check properties
    assert ctrl.is_public is False
    assert ctrl.is_shared is False


def test_public_access_allows_anyone():
    ctrl = make_control(access_level=AccessLevel.PUBLIC)
    assert ctrl.is_public
    assert not ctrl.is_shared
    for acc in ["owner_acc", "alice", "bob", "random"]:
        assert ctrl.has_access(acc) is True


def test_shared_access_with_explicit_accounts():
    shared_accounts = {"alice", "bob"}
    ctrl = make_control(access_level=AccessLevel.SHARED, shared_with=shared_accounts)
    assert ctrl.is_shared
    # Owner always has access
    assert ctrl.has_access("owner_acc")
    # Shared accounts have access
    assert ctrl.has_access("alice")
    assert ctrl.has_access("bob")
    # Non-shared accounts do not
    assert ctrl.has_access("charlie") is False
    assert ctrl.is_public is False


def test_shared_level_without_shared_with_empty():
    # SHARED level but empty set yields not shared
    ctrl = make_control(access_level=AccessLevel.SHARED, shared_with=set())
    assert ctrl.is_shared is False
    # Owner still has access
    assert ctrl.has_access("owner_acc")
    # Others do not
    assert ctrl.has_access("alice") is False

@pytest.mark.parametrize("level,expected_public", [
    (AccessLevel.PRIVATE, False),
    (AccessLevel.PUBLIC, True),
    (AccessLevel.SHARED, False),
])
def test_is_public_property(level, expected_public):
    ctrl = make_control(access_level=level)
    assert ctrl.is_public is expected_public

@pytest.mark.parametrize("level,shared,expected_shared", [
    (AccessLevel.PRIVATE, {"x"}, False),
    (AccessLevel.PUBLIC, {"x"}, False),
    (AccessLevel.SHARED, {"x"}, True),
    (AccessLevel.SHARED, set(), False),
])
def test_is_shared_property(level, shared, expected_shared):
    ctrl = make_control(access_level=level, shared_with=shared)
    assert ctrl.is_shared is expected_shared

