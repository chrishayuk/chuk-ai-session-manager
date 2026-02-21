# tests/test_base_models.py
"""Tests for DictCompatModel backward-compatible dict-style access."""

import pytest

from chuk_ai_session_manager.base_models import DictCompatModel


class _SampleModel(DictCompatModel):
    """Minimal subclass for testing."""

    name: str = "default"
    count: int = 0


class TestDictCompatModelGetItem:
    """Test bracket-notation access."""

    def test_getitem_returns_field_value(self):
        m = _SampleModel(name="alice", count=5)
        assert m["name"] == "alice"
        assert m["count"] == 5

    def test_getitem_missing_key_raises(self):
        m = _SampleModel()
        with pytest.raises(AttributeError):
            m["nonexistent"]


class TestDictCompatModelContains:
    """Test 'in' operator."""

    def test_contains_existing_field(self):
        m = _SampleModel()
        assert "name" in m
        assert "count" in m

    def test_contains_missing_field(self):
        m = _SampleModel()
        assert "nonexistent" not in m

    def test_contains_non_string_key(self):
        m = _SampleModel()
        assert 42 not in m


class TestDictCompatModelEq:
    """Test equality comparisons."""

    def test_eq_matching_dict(self):
        m = _SampleModel(name="bob", count=3)
        assert m == {"name": "bob", "count": 3}

    def test_eq_non_matching_dict(self):
        m = _SampleModel(name="bob", count=3)
        assert m != {"name": "bob", "count": 999}

    def test_eq_same_model(self):
        m1 = _SampleModel(name="bob", count=3)
        m2 = _SampleModel(name="bob", count=3)
        assert m1 == m2

    def test_eq_different_model(self):
        m1 = _SampleModel(name="bob", count=3)
        m2 = _SampleModel(name="alice", count=1)
        assert m1 != m2
