# tests/event_type.py
"""
Tests for the EventType enumeration.
"""

import pytest
from chuk_ai_session_manager.models.event_type import EventType


def test_event_type_values():
    """Test the event type string values."""
    assert EventType.MESSAGE.value == "message"
    assert EventType.SUMMARY.value == "summary"
    assert EventType.TOOL_CALL.value == "tool_call"
    assert EventType.REFERENCE.value == "reference"
    assert EventType.CONTEXT_BRIDGE.value == "context_bridge"
    # Add any other event types that are part of your enum


def test_event_type_comparison():
    """Test that event types can be compared with string values."""
    assert EventType.MESSAGE == "message"
    assert EventType.SUMMARY == "summary"
    assert EventType.TOOL_CALL == "tool_call"
    assert EventType.REFERENCE == "reference"
    assert EventType.CONTEXT_BRIDGE == "context_bridge"
    # Add any other event types that are part of your enum


def test_event_type_iteration():
    """Test that event types can be iterated."""
    # This test should include all event types
    expected_types = {
        EventType.MESSAGE, 
        EventType.SUMMARY, 
        EventType.TOOL_CALL,
        EventType.REFERENCE,
        EventType.CONTEXT_BRIDGE
        # Add any other event types that are part of your enum
    }
    
    # Get actual types through iteration
    actual_types = set(e for e in EventType)
    
    # Compare sets to ensure all types are included
    assert actual_types == expected_types