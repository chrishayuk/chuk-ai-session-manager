# tests/session_metadata.py
import pytest
from datetime import datetime, timezone, timedelta
import time

from chuk_ai_session_manager.models.session_metadata import SessionMetadata

def test_default_timestamps_are_set_and_utc():
    before = datetime.now(timezone.utc)
    time.sleep(0.001)
    meta = SessionMetadata()
    time.sleep(0.001)
    after = datetime.now(timezone.utc)

    # created_at and updated_at should be between before and after
    assert before < meta.created_at < after
    assert before < meta.updated_at < after

    # timezone info must be UTC
    assert meta.created_at.tzinfo == timezone.utc
    assert meta.updated_at.tzinfo == timezone.utc

def test_properties_initially_empty():
    meta = SessionMetadata()
    assert isinstance(meta.properties, dict)
    assert meta.properties == {}

@pytest.mark.asyncio
async def test_set_and_get_property():
    meta = SessionMetadata()
    assert await meta.get_property("nonexistent") is None

    await meta.set_property("foo", 123)
    assert "foo" in meta.properties
    assert await meta.get_property("foo") == 123

    # Overwriting works
    await meta.set_property("foo", "bar")
    assert await meta.get_property("foo") == "bar"

@pytest.mark.asyncio
async def test_model_dump_includes_properties_and_timestamps():
    meta = SessionMetadata()
    await meta.set_property("x", True)
    d = meta.model_dump()

    assert "created_at" in d
    assert "updated_at" in d
    assert d["properties"] == {"x": True}

@pytest.mark.asyncio
async def test_updating_updated_at_manually():
    # You might manually bump updated_at
    meta = SessionMetadata()
    old = meta.updated_at
    new_ts = old + timedelta(days=1)
    meta.updated_at = new_ts

    assert meta.updated_at == new_ts
    # created_at remains unchanged
    assert meta.created_at < meta.updated_at

@pytest.mark.asyncio
async def test_update_timestamp():
    meta = SessionMetadata()
    old_ts = meta.updated_at
    time.sleep(0.001)
    
    await meta.update_timestamp()
    
    assert meta.updated_at > old_ts
    assert meta.updated_at.tzinfo == timezone.utc

@pytest.mark.asyncio
async def test_create_class_method():
    properties = {"test_key": "test_value"}
    meta = await SessionMetadata.create(properties)
    
    assert isinstance(meta, SessionMetadata)
    assert meta.properties == properties
    assert isinstance(meta.created_at, datetime)
    assert isinstance(meta.updated_at, datetime)