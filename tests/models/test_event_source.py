# tests/event_source.py
import pytest
from chuk_ai_session_manager.models.event_source import EventSource


def test_event_source_members():
    # Ensure all expected members are present
    assert hasattr(EventSource, 'USER')
    assert hasattr(EventSource, 'LLM')
    assert hasattr(EventSource, 'SYSTEM')
    
    # Check values
    assert EventSource.USER.value == 'user'
    assert EventSource.LLM.value == 'llm'
    assert EventSource.SYSTEM.value == 'system'


def test_event_source_iteration():
    # Iteration should yield all enum members
    members = [e for e in EventSource]
    assert set(members) == {EventSource.USER, EventSource.LLM, EventSource.SYSTEM}


def test_event_source_equality_and_identity():
    # Equality between same named and value
    assert EventSource('user') is EventSource.USER
    assert EventSource.USER == EventSource.USER
    
    # Different enums are not equal
    assert EventSource.USER != EventSource.LLM


def test_event_source_invalid_value():
    # Constructing from an invalid value should raise ValueError
    with pytest.raises(ValueError):
        _ = EventSource('invalid')

@pytest.mark.parametrize("source,expected", [
    ('user', EventSource.USER),
    ('llm', EventSource.LLM),
    ('system', EventSource.SYSTEM),
])
def test_event_source_from_value(source, expected):
    # Casting from string returns correct enum member
    assert EventSource(source) == expected
