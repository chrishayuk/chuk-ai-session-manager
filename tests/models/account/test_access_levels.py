# tests/test_access_levels.py
"""
Unit tests for access levels.
"""
import pytest
from enum import Enum

from a2a_accounts.models.access_levels import AccessLevel


def test_access_level_is_enum():
    """Test that AccessLevel is an Enum."""
    assert issubclass(AccessLevel, Enum)


def test_access_level_is_str_enum():
    """Test that AccessLevel is a string Enum."""
    assert issubclass(AccessLevel, str)


def test_access_level_values():
    """Test the values of AccessLevel."""
    assert AccessLevel.PRIVATE == "private"
    assert AccessLevel.PUBLIC == "public"
    assert AccessLevel.SHARED == "shared"


def test_access_level_members():
    """Test the members of AccessLevel."""
    assert set(AccessLevel.__members__.keys()) == {"PRIVATE", "PUBLIC", "SHARED"}
    assert len(AccessLevel) == 3


def test_access_level_comparison():
    """Test comparing AccessLevel values."""
    assert AccessLevel.PRIVATE == AccessLevel.PRIVATE
    assert AccessLevel.PRIVATE != AccessLevel.PUBLIC
    assert AccessLevel.PRIVATE != AccessLevel.SHARED


def test_access_level_string_comparison():
    """Test comparing AccessLevel with strings."""
    assert AccessLevel.PRIVATE == "private"
    assert AccessLevel.PUBLIC == "public"
    assert AccessLevel.SHARED == "shared"
    assert AccessLevel.PRIVATE != "public"


def test_access_level_from_string():
    """Test creating AccessLevel from string."""
    assert AccessLevel("private") == AccessLevel.PRIVATE
    assert AccessLevel("public") == AccessLevel.PUBLIC
    assert AccessLevel("shared") == AccessLevel.SHARED


def test_access_level_invalid_string():
    """Test that creating AccessLevel from invalid string raises ValueError."""
    with pytest.raises(ValueError):
        AccessLevel("invalid")


def test_access_level_case_sensitivity():
    """Test that AccessLevel strings are case-sensitive."""
    with pytest.raises(ValueError):
        AccessLevel("PRIVATE")
    
    with pytest.raises(ValueError):
        AccessLevel("Private")


def test_access_level_in_dict():
    """Test using AccessLevel as a dictionary key."""
    access_map = {
        AccessLevel.PRIVATE: "Only the owner can access",
        AccessLevel.PUBLIC: "Anyone can access", 
        AccessLevel.SHARED: "Specific users can access"
    }
    
    assert access_map[AccessLevel.PRIVATE] == "Only the owner can access"
    assert access_map[AccessLevel.PUBLIC] == "Anyone can access"
    assert access_map[AccessLevel.SHARED] == "Specific users can access"
    
    # Test with string keys
    assert access_map["private"] == "Only the owner can access"
    assert access_map["public"] == "Anyone can access"
    assert access_map["shared"] == "Specific users can access"


def test_access_level_serialization():
    """Test that AccessLevel serializes to string."""
    # The default string representation of an enum includes the class name
    assert str(AccessLevel.PRIVATE) == "AccessLevel.PRIVATE"
    assert str(AccessLevel.PUBLIC) == "AccessLevel.PUBLIC" 
    assert str(AccessLevel.SHARED) == "AccessLevel.SHARED"
    
    # To get just the value, we need to use the value property or cast
    assert AccessLevel.PRIVATE.value == "private"
    assert AccessLevel.PUBLIC.value == "public"
    assert AccessLevel.SHARED.value == "shared"