# tests/conftest.py
"""
Shared pytest fixtures and configuration for chuk_ai_session_manager tests.

This configuration avoids circular imports by using lazy imports in fixtures.
"""

import asyncio
import pytest
import logging
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)
logging.getLogger("chuk_ai_session_manager").setLevel(logging.DEBUG)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_chuk_session_manager():
    """Mock CHUK SessionManager for testing."""
    mock = AsyncMock()
    mock.validate_session.return_value = True
    mock.get_session_info.return_value = {
        "custom_metadata": {
            "ai_session_data": json.dumps(
                {
                    "id": "test-session",
                    "metadata": {
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "properties": {},
                    },
                    "parent_id": None,
                    "child_ids": [],
                    "task_ids": [],
                    "runs": [],
                    "events": [],
                    "state": {},
                    "token_summary": {
                        "total_prompt_tokens": 0,
                        "total_completion_tokens": 0,
                        "total_tokens": 0,
                        "usage_by_model": {},
                        "total_estimated_cost_usd": 0.0,
                    },
                }
            ),
            "event_count": 0,
            "session_type": "ai_session_manager",
        }
    }
    mock.allocate_session = AsyncMock()
    mock.delete_session = AsyncMock()
    mock.extend_session_ttl = AsyncMock(return_value=True)
    # get_cache_stats should be synchronous and return a dict directly
    mock.get_cache_stats = MagicMock(return_value={"cache_size": 10, "hit_rate": 0.8})
    return mock


@pytest.fixture
async def mock_session_store():
    """Mock session store that tracks sessions in memory."""
    sessions = {}

    async def mock_get(session_id):
        return sessions.get(session_id)

    async def mock_save(session):
        sessions[session.id] = session

    async def mock_delete(session_id):
        sessions.pop(session_id, None)

    async def mock_list_sessions(prefix=""):
        return [sid for sid in sessions.keys() if sid.startswith(prefix)]

    mock_store = AsyncMock()
    mock_store.get.side_effect = mock_get
    mock_store.save.side_effect = mock_save
    mock_store.delete.side_effect = mock_delete
    mock_store.list_sessions.side_effect = mock_list_sessions

    return mock_store, sessions


@pytest.fixture
async def sample_session():
    """Create a sample session for testing."""
    # Import locally inside the fixture to avoid circular imports during collection
    from chuk_ai_session_manager.models.session_metadata import SessionMetadata
    from chuk_ai_session_manager.models.token_usage import TokenSummary

    # Create a minimal session without using Session.create() to avoid storage imports
    class MockSession:
        def __init__(self):
            self.id = "sample-session-123"
            self.metadata = SessionMetadata()
            self.parent_id = None
            self.child_ids = []
            self.task_ids = []
            self.runs = []
            self.events = []
            self.state = {}
            self.token_summary = TokenSummary()

        async def add_event(self, event):
            self.events.append(event)
            if event.token_usage:
                await self.token_summary.add_usage(event.token_usage)

        @property
        def total_tokens(self):
            return self.token_summary.total_tokens

        @property
        def total_cost(self):
            return self.token_summary.total_estimated_cost_usd

        @property
        def last_update_time(self):
            if not self.events:
                return self.metadata.created_at
            return max(evt.timestamp for evt in self.events)

    session = MockSession()

    # Create sample events using direct imports to avoid circular dependencies
    try:
        from chuk_ai_session_manager.models.session_event import SessionEvent
        from chuk_ai_session_manager.models.event_source import EventSource
        from chuk_ai_session_manager.models.event_type import EventType

        # Add some sample events
        user_event = await SessionEvent.create_with_tokens(
            message="Hello, how are you?",
            prompt="Hello, how are you?",
            model="gpt-3.5-turbo",
            source=EventSource.USER,
            type=EventType.MESSAGE,
        )

        ai_event = await SessionEvent.create_with_tokens(
            message="I'm doing well, thank you!",
            prompt="",
            completion="I'm doing well, thank you!",
            model="gpt-3.5-turbo",
            source=EventSource.LLM,
            type=EventType.MESSAGE,
        )

        await session.add_event(user_event)
        await session.add_event(ai_event)
    except ImportError:
        # If we can't import event classes, just return the basic session
        pass

    return session


@pytest.fixture
async def mock_tool_processor():
    """Mock tool processor for testing."""

    # Create a mock ToolResult class
    class MockToolResult:
        def __init__(self, tool, result, error=None):
            self.tool = tool
            self.result = result
            self.error = error

    mock_processor = MagicMock()
    mock_executor = AsyncMock()

    # Default successful tool result
    mock_executor.execute.return_value = [
        MockToolResult(tool="test_tool", result={"success": True}, error=None)
    ]

    mock_processor.executor = mock_executor
    return mock_processor


@pytest.fixture
async def mock_llm_callback():
    """Mock LLM callback for testing summarization."""

    async def callback(messages):
        # Simple mock that creates summaries based on message content
        user_messages = [m for m in messages if m.get("role") == "user"]
        if user_messages:
            topics = []
            for msg in user_messages:
                content = msg.get("content", "")
                if "?" in content:
                    topics.append(content.split("?")[0].strip())

            if topics:
                return f"Discussion about: {', '.join(topics[:3])}"
            else:
                return f"Conversation with {len(user_messages)} user messages"
        else:
            return "Empty conversation"

    return callback


@pytest.fixture
def mock_datetime():
    """Mock datetime for consistent testing."""
    fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    with patch("chuk_ai_session_manager.models.session_metadata.datetime") as mock_dt:
        mock_dt.now.return_value = fixed_time
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
        yield fixed_time


@pytest.fixture
def temp_session_data():
    """Generate temporary session data for tests."""

    def _generate_data(session_id="test-session", event_count=0):
        return {
            "id": session_id,
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "properties": {},
            },
            "parent_id": None,
            "child_ids": [],
            "task_ids": [],
            "runs": [],
            "events": [],
            "state": {},
            "token_summary": {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "usage_by_model": {},
                "total_estimated_cost_usd": 0.0,
            },
        }

    return _generate_data


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "models: Model tests")


# Custom assertions
def assert_session_valid(session):
    """Assert that a session is valid."""
    assert session is not None
    assert hasattr(session, "id")
    assert hasattr(session, "events")
    assert hasattr(session, "metadata")
    assert isinstance(session.events, list)


def assert_event_valid(event):
    """Assert that an event is valid."""
    assert event is not None
    assert hasattr(event, "id")
    assert hasattr(event, "timestamp")
    assert hasattr(event, "type")
    assert hasattr(event, "source")


# Add custom assertions to pytest namespace
pytest.assert_session_valid = assert_session_valid
pytest.assert_event_valid = assert_event_valid


# Mock for storage backend to avoid import issues
@pytest.fixture
def mock_storage_backend():
    """Mock storage backend that avoids import issues."""

    class MockSessionStorage:
        def __init__(self):
            self.cache = {}

        async def get(self, session_id):
            return self.cache.get(session_id)

        async def save(self, session):
            self.cache[session.id] = session

        async def delete(self, session_id):
            self.cache.pop(session_id, None)

        async def list_sessions(self, prefix=""):
            return [sid for sid in self.cache.keys() if sid.startswith(prefix)]

    return MockSessionStorage()


# Skip conditions
def skip_if_no_tiktoken():
    """Skip test if tiktoken is not available."""
    try:
        import tiktoken  # noqa: F401

        return False
    except ImportError:
        return True


# Parametrized fixtures for different scenarios
@pytest.fixture(params=["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet"])
def model_name(request):
    """Parametrized fixture for different model names."""
    return request.param


@pytest.fixture(autouse=True)
def mock_storage_imports():
    """Mock storage-related imports to avoid circular import issues during test collection."""

    # This fixture will automatically patch problematic imports during test collection
    # We need to patch both the main module and the tool processor module imports
    with (
        patch(
            "chuk_ai_session_manager.session_storage.get_backend"
        ) as mock_get_backend,
        patch(
            "chuk_ai_session_manager.session_storage.ChukSessionsStore"
        ) as mock_store,
        patch(
            "chuk_ai_session_manager.session_aware_tool_processor.get_backend"
        ) as mock_get_backend_tool,
        patch(
            "chuk_ai_session_manager.session_aware_tool_processor.ChukSessionsStore"
        ) as mock_store_tool,
    ):
        # Create a mock backend with proper async methods
        mock_backend = AsyncMock()

        # Create the get_stats method to return a coroutine that yields the dict
        async def mock_get_stats():
            return {
                "backend": "chuk_sessions",
                "sandbox_id": "test",
                "cached_ai_sessions": 0,
                "chuk_sessions_stats": {"cache_size": 10, "hit_rate": 0.8},
            }

        # Assign the coroutine function (not its result) to get_stats
        mock_backend.get_stats = mock_get_stats

        # Also ensure other backend methods are async
        mock_backend.get = AsyncMock(return_value=None)
        mock_backend.save = AsyncMock()
        mock_backend.delete = AsyncMock()
        mock_backend.list_sessions = AsyncMock(return_value=[])

        # Both imports should return the same backend
        mock_get_backend.return_value = mock_backend
        mock_get_backend_tool.return_value = mock_backend

        # Create a mock store with proper async methods
        mock_store_instance = AsyncMock()
        mock_store_instance.backend = (
            mock_backend  # Store should have reference to backend
        )
        mock_store_instance.get = AsyncMock(return_value=None)
        mock_store_instance.save = AsyncMock()
        mock_store_instance.delete = AsyncMock()
        mock_store_instance.list_sessions = AsyncMock(return_value=[])

        # Both imports should return the same store constructor
        mock_store.return_value = mock_store_instance
        mock_store_tool.return_value = mock_store_instance

        yield mock_backend, mock_store_instance
