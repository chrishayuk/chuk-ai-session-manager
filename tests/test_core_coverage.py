# tests/test_core_coverage.py
"""
Comprehensive tests targeting >90%% coverage for core files.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent


def _make_mock_store_and_sessions():
    sessions = {}
    mock_store = AsyncMock()

    async def _get(session_id):
        return sessions.get(session_id)

    async def _save(session):
        sessions[session.id] = session

    mock_store.get.side_effect = _get
    mock_store.save.side_effect = _save
    return mock_store, sessions


class TestInitModule:
    def test_get_version(self):
        from chuk_ai_session_manager import __version__, get_version

        result = get_version()
        assert result == __version__
        assert isinstance(result, str)

    def test_configure_storage_success(self):
        from chuk_ai_session_manager import configure_storage

        with patch("chuk_ai_session_manager.setup_chuk_sessions_storage") as mock_setup:
            mock_setup.return_value = MagicMock()
            result = configure_storage(sandbox_id="test-sandbox", default_ttl_hours=12)
            assert result is True
            mock_setup.assert_called_once_with(sandbox_id="test-sandbox", default_ttl_hours=12)

    def test_configure_storage_failure(self):
        from chuk_ai_session_manager import configure_storage

        with patch(
            "chuk_ai_session_manager.setup_chuk_sessions_storage",
            side_effect=RuntimeError("boom"),
        ):
            result = configure_storage()
            assert result is False

    def test_is_available_returns_dict_with_expected_keys(self):
        from chuk_ai_session_manager import is_available

        info = is_available()
        assert isinstance(info, dict)
        expected_keys = {
            "core_enums",
            "core_models",
            "simple_api",
            "storage",
            "infinite_context",
            "tool_processor",
            "prompt_builder",
            "token_tracking",
            "exceptions",
            "session_manager",
            "redis_support",
            "enhanced_token_counting",
        }
        assert expected_keys == set(info.keys())
        assert info["core_enums"] is True
        assert isinstance(info["redis_support"], bool)
        assert isinstance(info["enhanced_token_counting"], bool)

    def test_is_available_redis_import_fails(self):
        import builtins

        from chuk_ai_session_manager import is_available

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "redis":
                raise ImportError("no redis")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            info = is_available()
            assert info["redis_support"] is False

    def test_is_available_tiktoken_import_fails(self):
        import builtins

        from chuk_ai_session_manager import is_available

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "tiktoken":
                raise ImportError("no tiktoken")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            info = is_available()
            assert info["enhanced_token_counting"] is False

    async def test_get_storage_info_success(self):
        from chuk_ai_session_manager import get_storage_info

        mock_backend = MagicMock()
        mock_backend.get_stats.return_value = {
            "backend": "chuk_sessions",
            "sandbox_id": "test-sb",
        }
        with patch(
            "chuk_ai_session_manager.session_storage.get_backend",
            return_value=mock_backend,
        ):
            info = get_storage_info()
            assert "provider" in info
            assert "backend" in info
            assert info["backend"] == "chuk_sessions"
            assert "stats" in info

    async def test_get_storage_info_error(self):
        from chuk_ai_session_manager import get_storage_info

        with patch(
            "chuk_ai_session_manager.session_storage.get_backend",
            side_effect=RuntimeError("storage dead"),
        ):
            info = get_storage_info()
            assert "error" in info
            assert "storage dead" in info["error"]
            assert "provider" in info


class TestSimpleApiCoverage:
    async def test_track_tool_use(self):
        from chuk_ai_session_manager.api.simple_api import track_tool_use

        mock_store, sessions = _make_mock_store_and_sessions()
        with (
            patch(
                "chuk_ai_session_manager.session_storage.get_backend",
                return_value=mock_store,
            ),
            patch("chuk_ai_session_manager.models.session.Session.create") as mock_create,
        ):
            session = Session()
            session.id = "tool-session-1"
            mock_create.return_value = session
            sid = await track_tool_use(
                tool_name="calculator",
                arguments={"operation": "add", "a": 1, "b": 2},
                result={"result": 3},
                session_id=None,
                error=None,
                extra_info="test_meta",
            )
            assert sid == "tool-session-1"
            assert len(session.events) == 1
            assert session.events[0].type == EventType.TOOL_CALL
            assert session.events[0].message["tool"] == "calculator"
            assert session.events[0].message["success"] is True

    async def test_track_tool_use_with_error(self):
        from chuk_ai_session_manager.api.simple_api import track_tool_use

        mock_store, sessions = _make_mock_store_and_sessions()
        with (
            patch(
                "chuk_ai_session_manager.session_storage.get_backend",
                return_value=mock_store,
            ),
            patch("chuk_ai_session_manager.models.session.Session.create") as mock_create,
        ):
            session = Session()
            session.id = "tool-err-session"
            mock_create.return_value = session
            sid = await track_tool_use(
                tool_name="search",
                arguments={"query": "test"},
                result=None,
                error="timeout",
            )
            assert sid == "tool-err-session"
            assert session.events[0].message["success"] is False

    async def test_track_tool_use_with_session_id(self):
        from chuk_ai_session_manager.api.simple_api import track_tool_use

        mock_store, sessions = _make_mock_store_and_sessions()
        existing = Session()
        existing.id = "existing-tool-session"
        sessions["existing-tool-session"] = existing
        with (
            patch(
                "chuk_ai_session_manager.session_storage.get_backend",
                return_value=mock_store,
            ),
            patch("chuk_ai_session_manager.session_storage.ChukSessionsStore") as mock_css,
        ):
            store_inst = AsyncMock()
            store_inst.get.return_value = existing
            store_inst.save = AsyncMock()
            mock_css.return_value = store_inst
            sid = await track_tool_use(
                tool_name="weather",
                arguments={"location": "NYC"},
                result={"temp": 72},
                session_id="existing-tool-session",
            )
            assert sid == "existing-tool-session"

    async def test_get_session_stats(self):
        from chuk_ai_session_manager.api.simple_api import get_session_stats

        mock_store, sessions = _make_mock_store_and_sessions()
        with (
            patch(
                "chuk_ai_session_manager.session_storage.get_backend",
                return_value=mock_store,
            ),
            patch("chuk_ai_session_manager.models.session.Session.create") as mock_create,
        ):
            session = Session()
            session.id = "stats-session"
            mock_create.return_value = session
            from chuk_ai_session_manager.session_manager import SessionManager

            sm = SessionManager(session_id="stats-session")
            await sm.user_says("Hi")
            await sm.ai_responds("Hello", model="gpt-4")
            stats = await get_session_stats("stats-session")
            assert isinstance(stats, dict)
            assert "session_id" in stats

    async def test_get_session_stats_with_segments(self):
        from chuk_ai_session_manager.api.simple_api import get_session_stats

        mock_store, sessions = _make_mock_store_and_sessions()
        with (
            patch(
                "chuk_ai_session_manager.session_storage.get_backend",
                return_value=mock_store,
            ),
            patch("chuk_ai_session_manager.models.session.Session.create") as mock_create,
        ):
            session = Session()
            session.id = "stats-seg-session"
            mock_create.return_value = session
            stats = await get_session_stats("stats-seg-session", include_all_segments=True)
            assert isinstance(stats, dict)

    async def test_get_conversation_history(self):
        from chuk_ai_session_manager.api.simple_api import get_conversation_history

        mock_store, sessions = _make_mock_store_and_sessions()
        with (
            patch(
                "chuk_ai_session_manager.session_storage.get_backend",
                return_value=mock_store,
            ),
            patch("chuk_ai_session_manager.models.session.Session.create") as mock_create,
        ):
            session = Session()
            session.id = "history-session"
            mock_create.return_value = session
            from chuk_ai_session_manager.session_manager import SessionManager

            sm = SessionManager(session_id="history-session")
            await sm.user_says("Msg1")
            await sm.ai_responds("Resp1")
            history = await get_conversation_history("history-session")
            assert isinstance(history, list)

    async def test_get_conversation_history_with_segments(self):
        from chuk_ai_session_manager.api.simple_api import get_conversation_history

        mock_store, sessions = _make_mock_store_and_sessions()
        with (
            patch(
                "chuk_ai_session_manager.session_storage.get_backend",
                return_value=mock_store,
            ),
            patch("chuk_ai_session_manager.models.session.Session.create") as mock_create,
        ):
            session = Session()
            session.id = "history-seg-session"
            mock_create.return_value = session
            history = await get_conversation_history("history-seg-session", include_all_segments=True)
            assert isinstance(history, list)


class TestInfiniteConversationCoverage:
    async def test_process_message_triggers_segmentation(self):
        from chuk_ai_session_manager.infinite_conversation import (
            InfiniteConversationManager,
            SummarizationStrategy,
        )

        mock_store, sessions = _make_mock_store_and_sessions()
        session = Session()
        session.id = "seg-trigger-session"
        session.token_summary.total_tokens = 200
        sessions[session.id] = session

        async def llm_callback(messages):
            return "Summary of the conversation so far."

        manager = InfiniteConversationManager(
            token_threshold=100,
            max_turns_per_segment=50,
            summarization_strategy=SummarizationStrategy.BASIC,
        )
        with (
            patch(
                "chuk_ai_session_manager.infinite_conversation.get_backend",
                return_value=mock_store,
            ),
            patch("chuk_ai_session_manager.infinite_conversation.ChukSessionsStore") as mock_css,
        ):
            store_inst = AsyncMock()
            store_inst.get.side_effect = lambda sid: sessions.get(sid)
            store_inst.save.side_effect = lambda s: sessions.__setitem__(s.id, s)
            mock_css.return_value = store_inst
            with patch("chuk_ai_session_manager.models.session.Session.create") as mock_create:
                new_session = Session()
                new_session.id = "new-segment-id"
                new_session.parent_id = session.id
                mock_create.return_value = new_session
                result_id = await manager.process_message(
                    session_id=session.id,
                    message="Triggers segmentation",
                    source=EventSource.USER,
                    llm_callback=llm_callback,
                    model="gpt-4",
                )
                assert result_id == "new-segment-id"
                summary_events = [e for e in session.events if e.type == EventType.SUMMARY]
                assert len(summary_events) == 1

    async def test_get_summarization_prompt_query_focused(self):
        from chuk_ai_session_manager.infinite_conversation import (
            InfiniteConversationManager,
            SummarizationStrategy,
        )

        mgr = InfiniteConversationManager(summarization_strategy=SummarizationStrategy.QUERY_FOCUSED)
        prompt = mgr._get_summarization_prompt()
        assert "questions" in prompt.lower()

    async def test_get_summarization_prompt_default_fallback(self):
        from chuk_ai_session_manager.infinite_conversation import (
            InfiniteConversationManager,
        )

        mgr = InfiniteConversationManager()
        mgr.summarization_strategy = "unknown_strategy"
        prompt = mgr._get_summarization_prompt()
        assert "summary" in prompt.lower()

    async def test_build_context_for_llm_with_ancestor_summaries(self):
        from chuk_ai_session_manager.infinite_conversation import (
            InfiniteConversationManager,
        )

        mock_store, sessions = _make_mock_store_and_sessions()
        parent = Session()
        parent.id = "parent-session"
        summary_event = SessionEvent(
            message="Parent conversation summary: discussed weather.",
            source=EventSource.SYSTEM,
            type=EventType.SUMMARY,
        )
        await parent.add_event(summary_event)
        sessions[parent.id] = parent
        child = Session()
        child.id = "child-session"
        child.parent_id = "parent-session"
        msg_event = SessionEvent(
            message="What about tomorrow?",
            source=EventSource.USER,
            type=EventType.MESSAGE,
        )
        await child.add_event(msg_event)
        sessions[child.id] = child
        mgr = InfiniteConversationManager()
        with (
            patch(
                "chuk_ai_session_manager.infinite_conversation.get_backend",
                return_value=mock_store,
            ),
            patch("chuk_ai_session_manager.infinite_conversation.ChukSessionsStore") as mock_css,
        ):
            store_inst = AsyncMock()
            store_inst.get.side_effect = lambda sid: sessions.get(sid)
            mock_css.return_value = store_inst
            with patch(
                "chuk_ai_session_manager.models.session.Session.ancestors",
                new_callable=AsyncMock,
                return_value=[parent],
            ):
                context = await mgr.build_context_for_llm(
                    session_id="child-session",
                    max_messages=10,
                    include_summaries=True,
                )
                # Context should contain at least the child's user message
                assert len(context) >= 1
                user_msgs = [m for m in context if m.get("role") == "user"]
                assert len(user_msgs) >= 1

    async def test_get_session_chain(self):
        from chuk_ai_session_manager.infinite_conversation import (
            InfiniteConversationManager,
        )

        mock_store, sessions = _make_mock_store_and_sessions()
        root = Session()
        root.id = "root-session"
        sessions[root.id] = root
        child = Session()
        child.id = "child-session"
        child.parent_id = "root-session"
        sessions[child.id] = child
        mgr = InfiniteConversationManager()
        with (
            patch(
                "chuk_ai_session_manager.infinite_conversation.get_backend",
                return_value=mock_store,
            ),
            patch("chuk_ai_session_manager.infinite_conversation.ChukSessionsStore") as mock_css,
        ):
            store_inst = AsyncMock()
            store_inst.get.side_effect = lambda sid: sessions.get(sid)
            mock_css.return_value = store_inst
            with patch(
                "chuk_ai_session_manager.models.session.Session.ancestors",
                new_callable=AsyncMock,
                return_value=[root],
            ):
                chain = await mgr.get_session_chain("child-session")
                assert len(chain) == 2
                assert chain[0].id == "root-session"
                assert chain[1].id == "child-session"

    async def test_get_session_chain_not_found(self):
        from chuk_ai_session_manager.infinite_conversation import (
            InfiniteConversationManager,
        )

        mock_store, sessions = _make_mock_store_and_sessions()
        mgr = InfiniteConversationManager()
        with (
            patch(
                "chuk_ai_session_manager.infinite_conversation.get_backend",
                return_value=mock_store,
            ),
            patch("chuk_ai_session_manager.infinite_conversation.ChukSessionsStore") as mock_css,
        ):
            store_inst = AsyncMock()
            store_inst.get.return_value = None
            mock_css.return_value = store_inst
            with pytest.raises(ValueError, match="not found"):
                await mgr.get_session_chain("nonexistent-id")

    async def test_get_full_conversation_history(self):
        from chuk_ai_session_manager.infinite_conversation import (
            InfiniteConversationManager,
        )

        mock_store, sessions = _make_mock_store_and_sessions()
        s1 = Session()
        s1.id = "s1"
        await s1.add_event(SessionEvent(message="Hello", source=EventSource.USER, type=EventType.MESSAGE))
        await s1.add_event(SessionEvent(message="Hi!", source=EventSource.LLM, type=EventType.MESSAGE))
        sessions[s1.id] = s1
        s2 = Session()
        s2.id = "s2"
        s2.parent_id = "s1"
        await s2.add_event(SessionEvent(message="How are you?", source=EventSource.USER, type=EventType.MESSAGE))
        sessions[s2.id] = s2
        mgr = InfiniteConversationManager()
        with (
            patch(
                "chuk_ai_session_manager.infinite_conversation.get_backend",
                return_value=mock_store,
            ),
            patch("chuk_ai_session_manager.infinite_conversation.ChukSessionsStore") as mock_css,
        ):
            store_inst = AsyncMock()
            store_inst.get.side_effect = lambda sid: sessions.get(sid)
            mock_css.return_value = store_inst
            with patch.object(mgr, "get_session_chain", return_value=[s1, s2]):
                history = await mgr.get_full_conversation_history("s2")
                assert len(history) == 3
                assert history[0] == ("user", EventSource.USER, "Hello")
                assert history[1] == ("assistant", EventSource.LLM, "Hi!")
                assert history[2] == ("user", EventSource.USER, "How are you?")


class TestSessionManagerCoverage:
    async def test_ensure_initialized_error_path_creates_new_session(self):
        from chuk_ai_session_manager.session_manager import SessionManager

        store_inst = AsyncMock()
        store_inst.get.side_effect = RuntimeError("storage exploded")
        store_inst.save = AsyncMock()
        with patch("chuk_ai_session_manager.models.session.Session.create") as mock_create:
            new_session = Session()
            new_session.id = "recovery-session"
            mock_create.return_value = new_session
            sm = SessionManager(session_id="some-valid-id", store=store_inst)
            await sm._ensure_initialized()
            assert sm._session is not None
            assert sm._initialized is True

    async def test_ensure_initialized_error_path_with_infinite_context(self):
        from chuk_ai_session_manager.session_manager import SessionManager

        store_inst = AsyncMock()
        store_inst.get.side_effect = RuntimeError("conn lost")
        store_inst.save = AsyncMock()
        with (
            patch(
                "chuk_ai_session_manager.session_storage.get_backend",
                return_value=AsyncMock(),
            ),
            patch("chuk_ai_session_manager.session_storage.ChukSessionsStore") as mock_css,
        ):
            mock_css.return_value = store_inst
            with patch("chuk_ai_session_manager.models.session.Session.create") as mock_create:
                new_session = Session()
                new_session.id = "inf-recovery"
                mock_create.return_value = new_session
                sm = SessionManager(
                    session_id="fail-id",
                    infinite_context=True,
                    system_prompt="Be helpful",
                    metadata={"user": "test"},
                )
                await sm._ensure_initialized()
                assert sm._infinite_context is True
                assert sm._session_chain == ["fail-id"]

    async def test_ensure_initialized_not_found_creates_new(self):
        from chuk_ai_session_manager.session_manager import SessionManager

        store_inst = AsyncMock()
        store_inst.get.return_value = None
        store_inst.save = AsyncMock()
        with (
            patch(
                "chuk_ai_session_manager.session_storage.get_backend",
                return_value=AsyncMock(),
            ),
            patch("chuk_ai_session_manager.session_storage.ChukSessionsStore") as mock_css,
        ):
            mock_css.return_value = store_inst
            with patch("chuk_ai_session_manager.models.session.Session.create") as mock_create:
                new_session = Session()
                new_session.id = "brand-new-session"
                mock_create.return_value = new_session
                sm = SessionManager(
                    session_id="brand-new-session",
                    system_prompt="Test prompt",
                    metadata={"key": "val"},
                    infinite_context=True,
                )
                await sm._ensure_initialized()
                assert sm._initialized is True
                assert sm._session is not None

    async def test_create_summary_with_questions(self):
        from chuk_ai_session_manager.session_manager import SessionManager

        mock_store, sessions = _make_mock_store_and_sessions()
        with (
            patch(
                "chuk_ai_session_manager.session_storage.get_backend",
                return_value=mock_store,
            ),
            patch("chuk_ai_session_manager.models.session.Session.create") as mock_create,
        ):
            session = Session()
            session.id = "summary-q-session"
            mock_create.return_value = session
            sm = SessionManager(infinite_context=True)
            await sm.user_says("What is the meaning of life and everything?")
            await sm.ai_responds("42")
            await sm.user_says("How does quantum computing work in practice?")
            await sm.ai_responds("It uses qubits.")
            await sm.user_says("Can you explain general relativity to me?")
            await sm.ai_responds("Space-time curves.")
            summary = await sm._create_summary()
            assert isinstance(summary, str)
            assert "discussed" in summary.lower() or "user" in summary.lower()

    async def test_create_summary_with_more_than_3_topics(self):
        from chuk_ai_session_manager.session_manager import SessionManager

        mock_store, sessions = _make_mock_store_and_sessions()
        with (
            patch(
                "chuk_ai_session_manager.session_storage.get_backend",
                return_value=mock_store,
            ),
            patch("chuk_ai_session_manager.models.session.Session.create") as mock_create,
        ):
            session = Session()
            session.id = "many-topics-session"
            mock_create.return_value = session
            sm = SessionManager(infinite_context=True)
            questions = [
                "What is the capital of France in Europe?",
                "How does photosynthesis work in plants?",
                "What is the speed of light in a vacuum?",
                "How do computers process information internally?",
                "What causes the northern lights phenomenon?",
            ]
            for q in questions:
                await sm.user_says(q)
                await sm.ai_responds("An answer.")
            summary = await sm._create_summary()
            assert "other topic" in summary.lower()

    async def test_create_summary_no_questions(self):
        from chuk_ai_session_manager.session_manager import SessionManager

        mock_store, sessions = _make_mock_store_and_sessions()
        with (
            patch(
                "chuk_ai_session_manager.session_storage.get_backend",
                return_value=mock_store,
            ),
            patch("chuk_ai_session_manager.models.session.Session.create") as mock_create,
        ):
            session = Session()
            session.id = "no-q-session"
            mock_create.return_value = session
            sm = SessionManager(infinite_context=True)
            await sm.user_says("Hello there")
            await sm.ai_responds("Hi!")
            await sm.user_says("Tell me about dogs")
            await sm.ai_responds("Dogs are great.")
            summary = await sm._create_summary()
            assert "conversation" in summary.lower()

    async def test_create_summary_with_llm_callback(self):
        from chuk_ai_session_manager.session_manager import SessionManager

        mock_store, sessions = _make_mock_store_and_sessions()
        with (
            patch(
                "chuk_ai_session_manager.session_storage.get_backend",
                return_value=mock_store,
            ),
            patch("chuk_ai_session_manager.models.session.Session.create") as mock_create,
        ):
            session = Session()
            session.id = "cb-summary-session"
            mock_create.return_value = session
            sm = SessionManager(infinite_context=True)
            await sm.user_says("Test message")

            async def custom_callback(messages):
                return "Custom LLM summary"

            summary = await sm._create_summary(llm_callback=custom_callback)
            assert summary == "Custom LLM summary"

    async def test_get_stats_infinite_with_chain_reconstruction(self):
        from chuk_ai_session_manager.session_manager import SessionManager

        session1 = Session()
        session1.id = "chain-1"
        session2 = Session()
        session2.id = "chain-2"
        session2.parent_id = "chain-1"
        sessions_db = {"chain-1": session1, "chain-2": session2}
        mock_store_inst = AsyncMock()
        mock_store_inst.get.side_effect = lambda sid: sessions_db.get(sid)
        mock_store_inst.save = AsyncMock()
        with (
            patch(
                "chuk_ai_session_manager.session_storage.get_backend",
                return_value=AsyncMock(),
            ),
            patch("chuk_ai_session_manager.session_storage.ChukSessionsStore") as mock_css,
        ):
            mock_css.return_value = mock_store_inst
            with patch("chuk_ai_session_manager.models.session.Session.create") as mock_create:
                mock_create.return_value = session2
                sm = SessionManager(
                    session_id="chain-2",
                    infinite_context=True,
                    store=mock_store_inst,
                )
                await sm._ensure_initialized()
                sm._total_segments = 2
                sm._session_chain = ["chain-2"]
                sm._full_conversation = [
                    {"role": "user", "content": "msg1"},
                    {"role": "assistant", "content": "resp1"},
                ]
                stats = await sm.get_stats(include_all_segments=True)
                assert stats["infinite_context"] is True
                assert "chain-1" in stats["session_chain"]
                assert "chain-2" in stats["session_chain"]
                assert stats["session_segments"] == len(stats["session_chain"])

    async def test_get_stats_infinite_exception_in_chain_load(self):
        from chuk_ai_session_manager.session_manager import SessionManager

        session = Session()
        session.id = "except-chain"
        mock_store_inst = AsyncMock()

        async def get_side_effect(sid):
            if sid == "except-chain":
                return session
            raise RuntimeError("storage error")

        mock_store_inst.get.side_effect = get_side_effect
        mock_store_inst.save = AsyncMock()
        with (
            patch(
                "chuk_ai_session_manager.session_storage.get_backend",
                return_value=AsyncMock(),
            ),
            patch("chuk_ai_session_manager.session_storage.ChukSessionsStore") as mock_css,
        ):
            mock_css.return_value = mock_store_inst
            with patch("chuk_ai_session_manager.models.session.Session.create") as mock_create:
                mock_create.return_value = session
                sm = SessionManager(
                    session_id="except-chain",
                    infinite_context=True,
                    store=mock_store_inst,
                )
                await sm._ensure_initialized()
                sm._session_chain = ["bad-id", "except-chain"]
                sm._total_segments = 2
                sm._full_conversation = []
                stats = await sm.get_stats(include_all_segments=True)
                assert isinstance(stats, dict)

    async def test_set_summary_callback(self):
        from chuk_ai_session_manager.session_manager import SessionManager

        sm = SessionManager(infinite_context=True)

        async def my_callback(messages):
            return "callback summary"

        sm.set_summary_callback(my_callback)
        assert sm._summary_callback is my_callback

    async def test_load_session_chain_basic(self):
        from chuk_ai_session_manager.session_manager import SessionManager

        parent = Session()
        parent.id = "lsc-parent"
        await parent.add_event(SessionEvent(message="Parent msg", source=EventSource.USER, type=EventType.MESSAGE))
        child = Session()
        child.id = "lsc-child"
        child.parent_id = "lsc-parent"
        await child.add_event(SessionEvent(message="Child msg", source=EventSource.LLM, type=EventType.MESSAGE))
        sessions_db = {"lsc-parent": parent, "lsc-child": child}
        mock_store_inst = AsyncMock()
        mock_store_inst.get.side_effect = lambda sid: sessions_db.get(sid)
        mock_store_inst.save = AsyncMock()
        with (
            patch(
                "chuk_ai_session_manager.session_storage.get_backend",
                return_value=AsyncMock(),
            ),
            patch("chuk_ai_session_manager.session_storage.ChukSessionsStore") as mock_css,
        ):
            mock_css.return_value = mock_store_inst
            with patch("chuk_ai_session_manager.models.session.Session.create") as mock_create:
                mock_create.return_value = child
                sm = SessionManager(
                    session_id="lsc-child",
                    infinite_context=True,
                    store=mock_store_inst,
                )
                await sm._ensure_initialized()
                await sm.load_session_chain()
                assert len(sm._session_chain) == 2
                assert "lsc-parent" in sm._session_chain
                assert "lsc-child" in sm._session_chain
                assert sm._total_segments == 2
                assert len(sm._full_conversation) >= 2

    async def test_load_session_chain_no_infinite(self):
        from chuk_ai_session_manager.session_manager import SessionManager

        sm = SessionManager(infinite_context=False)
        await sm.load_session_chain()
        assert sm._session_chain == []

    async def test_load_session_chain_session_not_found(self):
        from chuk_ai_session_manager.session_manager import SessionManager

        mock_store_inst = AsyncMock()
        mock_store_inst.get.return_value = None
        mock_store_inst.save = AsyncMock()
        with (
            patch(
                "chuk_ai_session_manager.session_storage.get_backend",
                return_value=AsyncMock(),
            ),
            patch("chuk_ai_session_manager.session_storage.ChukSessionsStore") as mock_css,
        ):
            mock_css.return_value = mock_store_inst
            with patch("chuk_ai_session_manager.models.session.Session.create") as mock_create:
                s = Session()
                s.id = "lsc-missing"
                mock_create.return_value = s
                sm = SessionManager(
                    session_id="lsc-missing",
                    infinite_context=True,
                    store=mock_store_inst,
                )
                await sm._ensure_initialized()
                await sm.load_session_chain()
                assert "lsc-missing" in sm._session_chain


class TestSampleTools:
    async def test_calculator_add(self):
        from chuk_ai_session_manager.sample_tools import CalculatorTool

        calc = CalculatorTool()
        result = await calc.execute(operation="add", a=5, b=3)
        assert result["result"] == 8
        assert result["operation"] == "add"
        assert "timestamp" in result

    async def test_calculator_subtract(self):
        from chuk_ai_session_manager.sample_tools import CalculatorTool

        result = await CalculatorTool().execute(operation="subtract", a=10, b=4)
        assert result["result"] == 6

    async def test_calculator_multiply(self):
        from chuk_ai_session_manager.sample_tools import CalculatorTool

        result = await CalculatorTool().execute(operation="multiply", a=7, b=6)
        assert result["result"] == 42

    async def test_calculator_divide(self):
        from chuk_ai_session_manager.sample_tools import CalculatorTool

        result = await CalculatorTool().execute(operation="divide", a=20, b=4)
        assert result["result"] == 5.0

    async def test_calculator_divide_by_zero(self):
        from chuk_ai_session_manager.sample_tools import CalculatorTool

        with pytest.raises(ValueError, match="Cannot divide by zero"):
            await CalculatorTool().execute(operation="divide", a=10, b=0)

    async def test_calculator_unknown_operation(self):
        from chuk_ai_session_manager.sample_tools import CalculatorTool

        with pytest.raises(ValueError, match="Unknown operation"):
            await CalculatorTool().execute(operation="modulo", a=10, b=3)

    async def test_weather_miami(self):
        from chuk_ai_session_manager.sample_tools import WeatherTool

        result = await WeatherTool().execute(location="Miami")
        assert result["location"] == "Miami"
        assert isinstance(result["temperature"], float)
        assert isinstance(result["condition"], str)
        assert isinstance(result["humidity"], int)
        assert isinstance(result["wind_speed"], float)
        assert "timestamp" in result
        assert "feels_like" in result

    async def test_weather_moscow(self):
        from chuk_ai_session_manager.sample_tools import WeatherTool

        result = await WeatherTool().execute(location="Moscow")
        assert result["location"] == "Moscow"
        assert isinstance(result["temperature"], float)

    async def test_weather_london(self):
        from chuk_ai_session_manager.sample_tools import WeatherTool

        result = await WeatherTool().execute(location="London")
        assert result["location"] == "London"
        assert isinstance(result["temperature"], float)

    async def test_weather_tokyo(self):
        from chuk_ai_session_manager.sample_tools import WeatherTool

        result = await WeatherTool().execute(location="Tokyo")
        assert result["location"] == "Tokyo"
        assert isinstance(result["temperature"], float)

    async def test_weather_generic_location(self):
        from chuk_ai_session_manager.sample_tools import WeatherTool

        result = await WeatherTool().execute(location="Timbuktu")
        assert result["location"] == "Timbuktu"
        assert isinstance(result["temperature"], float)

    async def test_search_climate_query(self):
        from chuk_ai_session_manager.sample_tools import SearchTool

        result = await SearchTool().execute(query="climate change adaptation")
        assert result["query"] == "climate change adaptation"
        assert result["results_count"] <= 3
        assert len(result["results"]) > 0
        assert "climate" in result["results"][0]["title"].lower()
        assert "timestamp" in result

    async def test_search_weather_query(self):
        from chuk_ai_session_manager.sample_tools import SearchTool

        result = await SearchTool().execute(query="weather forecast today")
        assert "weather" in result["results"][0]["title"].lower()

    async def test_search_generic_query(self):
        from chuk_ai_session_manager.sample_tools import SearchTool

        result = await SearchTool().execute(query="python programming")
        assert result["results_count"] <= 3
        assert len(result["results"]) > 0

    async def test_search_max_results_parameter(self):
        from chuk_ai_session_manager.sample_tools import SearchTool

        result = await SearchTool().execute(query="python programming", max_results=1)
        assert result["results_count"] == 1
        assert len(result["results"]) == 1

    async def test_search_max_results_exceeds_templates(self):
        from chuk_ai_session_manager.sample_tools import SearchTool

        result = await SearchTool().execute(query="climate change adaptation", max_results=10)
        assert result["results_count"] <= 10
        assert len(result["results"]) > 0

    async def test_search_environment_query(self):
        from chuk_ai_session_manager.sample_tools import SearchTool

        result = await SearchTool().execute(query="environment protection")
        assert len(result["results"]) > 0
        first_title = result["results"][0]["title"].lower()
        assert "climate" in first_title or "environment" in first_title


class TestAdditionalCoverage:
    async def test_ensure_initialized_double_lock(self):
        from chuk_ai_session_manager.session_manager import SessionManager

        mock_store, sessions = _make_mock_store_and_sessions()
        with (
            patch(
                "chuk_ai_session_manager.session_storage.get_backend",
                return_value=mock_store,
            ),
            patch("chuk_ai_session_manager.models.session.Session.create") as mock_create,
        ):
            session = Session()
            session.id = "double-lock-session"
            mock_create.return_value = session
            sm = SessionManager()
            await sm._ensure_initialized()
            await sm._ensure_initialized()
            assert sm._initialized is True
            mock_create.assert_called_once()

    async def test_get_stats_non_infinite(self):
        from chuk_ai_session_manager.session_manager import SessionManager

        mock_store, sessions = _make_mock_store_and_sessions()
        with (
            patch(
                "chuk_ai_session_manager.session_storage.get_backend",
                return_value=mock_store,
            ),
            patch("chuk_ai_session_manager.models.session.Session.create") as mock_create,
        ):
            session = Session()
            session.id = "stats-non-inf"
            mock_create.return_value = session
            sm = SessionManager(infinite_context=False)
            await sm.user_says("Hello")
            await sm.ai_responds("Hi")
            stats = await sm.get_stats()
            assert stats["infinite_context"] is False
            assert stats["session_segments"] == 1
            assert stats["user_messages"] == 1
            assert stats["ai_messages"] == 1

    async def test_init_module_auto_setup(self):
        import chuk_ai_session_manager

        assert hasattr(chuk_ai_session_manager, "__version__")
        assert hasattr(chuk_ai_session_manager, "configure_storage")
        assert hasattr(chuk_ai_session_manager, "get_storage_info")

    async def test_init_module_get_storage_info_log_exception(self):
        from chuk_ai_session_manager import get_storage_info

        with patch(
            "chuk_ai_session_manager.session_storage.get_backend",
            side_effect=Exception("total failure"),
        ):
            info = get_storage_info()
            assert "error" in info

    async def test_ensure_initialized_with_metadata_and_system_prompt(self):
        from chuk_ai_session_manager.session_manager import SessionManager

        mock_store, sessions = _make_mock_store_and_sessions()
        with (
            patch(
                "chuk_ai_session_manager.session_storage.get_backend",
                return_value=mock_store,
            ),
            patch("chuk_ai_session_manager.models.session.Session.create") as mock_create,
        ):
            session = Session()
            session.id = "meta-session"
            mock_create.return_value = session
            sm = SessionManager(system_prompt="You are helpful", metadata={"team": "backend"})
            await sm._ensure_initialized()
            assert sm._system_prompt == "You are helpful"
            assert sm._initialized is True
