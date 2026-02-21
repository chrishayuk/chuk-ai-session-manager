# src/chuk_ai_session_manager/session_manager.py
"""
SessionManager - High-level API for managing AI conversation sessions.

This module provides the main SessionManager class which offers:
- Automatic conversation tracking
- Token usage monitoring
- System prompt management
- Infinite context support with automatic summarization
- Tool call logging
- Session persistence and retrieval
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

from chuk_ai_session_manager.config import DEFAULT_TOKEN_MODEL
from chuk_ai_session_manager.memory.manager import MemoryManager
from chuk_ai_session_manager.memory.models import (
    MessageRole,
    PageType,
    StorageTier,
    VMMode,
)
from chuk_ai_session_manager.memory.working_set import WorkingSetConfig
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.session_storage import ChukSessionsStore

logger = logging.getLogger(__name__)

# Default importance scores for VM page creation
VM_IMPORTANCE_USER = 0.6
VM_IMPORTANCE_AI = 0.5
VM_IMPORTANCE_TOOL = 0.4
VM_IMPORTANCE_SUMMARY = 0.9

# Page ID prefixes for VM
PAGE_ID_PREFIX_MSG = "msg"


class SessionManager:
    """
    High-level session manager for AI conversations.

    Provides an easy-to-use interface for tracking conversations, managing
    system prompts, handling infinite context, and monitoring usage.

    Examples:
        Basic usage:
        ```python
        sm = SessionManager()
        await sm.user_says("Hello!")
        await sm.ai_responds("Hi there!", model="gpt-4")
        ```

        With system prompt:
        ```python
        sm = SessionManager(system_prompt="You are a helpful assistant.")
        await sm.user_says("What can you do?")
        ```

        Infinite context:
        ```python
        sm = SessionManager(infinite_context=True, token_threshold=4000)
        # Automatically handles long conversations
        ```
    """

    def __init__(
        self,
        session_id: str | None = None,
        system_prompt: str | None = None,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        store: ChukSessionsStore | None = None,
        infinite_context: bool = False,
        token_threshold: int = 4000,
        max_turns_per_segment: int = 20,
        default_model: str = DEFAULT_TOKEN_MODEL,
        enable_vm: bool = False,
        vm_config: WorkingSetConfig | None = None,
        vm_mode: VMMode = VMMode.STRICT,
        vm_eviction_policy: Any | None = None,
        vm_compressor_registry: Any | None = None,
    ):
        """
        Initialize a SessionManager.

        Args:
            session_id: Optional session ID. If not provided, a new one will be generated.
            system_prompt: Optional system prompt to set the context for the AI assistant.
            parent_id: Optional parent session ID for creating child sessions.
            metadata: Optional metadata to attach to the session.
            store: Optional session store. If not provided, the default will be used.
            infinite_context: Enable automatic infinite context handling.
            token_threshold: Token limit before creating new session (infinite mode).
            max_turns_per_segment: Turn limit before creating new session (infinite mode).
            default_model: Model name used for token counting (default: gpt-4o-mini).
            enable_vm: Enable AI Virtual Memory subsystem for context management.
            vm_config: Optional WorkingSetConfig for VM working set sizing.
            vm_mode: VM mode (STRICT, RELAXED, or PASSIVE). Default: STRICT.
            vm_eviction_policy: Optional eviction policy for VM working set.
            vm_compressor_registry: Optional compressor registry for compress-before-evict.
        """
        # Core session management
        self._session_id = session_id
        self._system_prompt = system_prompt
        self._parent_id = parent_id
        self._metadata = metadata or {}
        self._store = store or ChukSessionsStore()
        self._session: Session | None = None
        self._initialized = False
        self._lock = asyncio.Lock()
        self._loaded_from_storage = False  # Track if loaded from storage
        self._default_model = default_model
        self._summary_callback: Callable | None = None

        # Infinite context settings
        self._infinite_context = infinite_context
        self._token_threshold = token_threshold
        self._max_turns_per_segment = max_turns_per_segment

        # Infinite context state
        self._session_chain: list[str] = []
        self._full_conversation: list[dict[str, Any]] = []
        self._total_segments = 1

        # Virtual Memory subsystem
        self._vm: MemoryManager | None = None
        if enable_vm:
            self._vm = MemoryManager(
                session_id=self.session_id,
                config=vm_config,
                mode=vm_mode,
                eviction_policy=vm_eviction_policy,
                compressor_registry=vm_compressor_registry,
            )

    @property
    def session_id(self) -> str:
        """Get the current session ID."""
        if self._session:
            return self._session.id
        elif self._session_id:
            return self._session_id
        else:
            self._session_id = str(uuid.uuid4())
            return self._session_id

    @property
    def system_prompt(self) -> str | None:
        """Get the current system prompt."""
        return self._system_prompt

    @property
    def is_infinite(self) -> bool:
        """Check if infinite context is enabled."""
        return self._infinite_context

    @property
    def _is_new(self) -> bool:
        """Check if this is a new session."""
        if not self._initialized:
            return True
        return not self._loaded_from_storage

    @property
    def vm(self) -> MemoryManager | None:
        """Access the Virtual Memory manager (None if VM is disabled)."""
        return self._vm

    def get_vm_context(
        self,
        model_id: str = "",
        token_budget: int | None = None,
    ) -> dict[str, Any] | None:
        """
        Get the full VM context for an LLM call.

        Returns None if VM is disabled. Otherwise returns a dict with:
        - developer_message: str with VM rules, manifest, and context
        - tools: list of VM tool definitions
        - manifest: VMManifest
        - packed_context: PackedContext

        Args:
            model_id: Optional model identifier for context sizing.
            token_budget: Optional token budget override.
        """
        if not self._vm:
            return None

        return self._vm.build_context(
            system_prompt=self._system_prompt or "",
            model_id=model_id,
            token_budget=token_budget,
        )

    async def _ensure_session(self) -> Session | None:
        """Ensure session is initialized and return it."""
        await self._ensure_initialized()
        return self._session

    async def update_system_prompt(self, prompt: str) -> None:
        """
        Update the system prompt for the session.

        Args:
            prompt: The new system prompt to use.
        """
        async with self._lock:
            self._system_prompt = prompt

            # Store in session metadata
            if self._session:
                self._session.metadata.properties["system_prompt"] = prompt
                await self._save_session()
            else:
                # Store for when session is initialized
                self._metadata["system_prompt"] = prompt

        logger.debug(f"Updated system prompt for session {self.session_id}")

    async def _create_and_save_session(self, session_id: str | None = None) -> Session:
        """Create a new session with metadata and save it."""
        session_metadata: dict[str, Any] = {}
        if self._metadata:
            session_metadata.update(self._metadata)
        if self._system_prompt:
            session_metadata["system_prompt"] = self._system_prompt

        session = await Session.create(
            session_id=session_id,
            parent_id=self._parent_id,
            metadata=session_metadata,
        )

        if session_metadata:
            session.metadata.properties.update(session_metadata)

        await self._store.save(session)
        return session

    async def _ensure_initialized(self) -> None:
        """Ensure the session is initialized."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:  # Double-check after acquiring lock
                return

            store = self._store

            if self._session_id:
                # Try to load existing session
                try:
                    self._session = await store.get(self._session_id)

                    if self._session:
                        # Mark as loaded from storage
                        self._loaded_from_storage = True

                        # Load system prompt from session if not already set
                        if not self._system_prompt and self._session.metadata.properties:
                            self._system_prompt = self._session.metadata.properties.get("system_prompt")

                        # Initialize session chain for infinite context
                        if self._infinite_context:
                            self._session_chain = [self._session_id]
                    else:
                        # Session not found - create a new session with the provided ID
                        self._session = await self._create_and_save_session(self._session_id)
                        self._loaded_from_storage = False

                        if self._infinite_context:
                            self._session_chain = [self._session_id]
                except Exception as e:
                    logger.warning(f"Failed to load session {self._session_id}: {e}")
                    self._session = await self._create_and_save_session(self._session_id)
                    self._loaded_from_storage = False

                    if self._infinite_context:
                        self._session_chain = [self._session_id]
            else:
                # Create new session
                self._session = await self._create_and_save_session()
                self._session_id = self._session.id
                self._loaded_from_storage = False

                if self._infinite_context:
                    self._session_chain = [self._session_id]

            self._initialized = True

    async def _save_session(self) -> None:
        """Save the current session."""
        if self._session:
            await self._store.save(self._session)

    async def _should_create_new_segment(self) -> bool:
        """Check if we should create a new session segment."""
        if not self._infinite_context:
            return False

        await self._ensure_initialized()
        assert self._session is not None

        # Check token threshold
        if self._session.total_tokens >= self._token_threshold:
            return True

        # Check turn threshold
        message_events = [e for e in self._session.events if e.type == EventType.MESSAGE]
        return len(message_events) >= self._max_turns_per_segment

    async def _create_summary(self, llm_callback: Callable | None = None) -> str:
        """
        Create a summary of the current session.

        Args:
            llm_callback: Optional async function to generate summary using an LLM.
                         Should accept List[Dict] messages and return str summary.
        """
        await self._ensure_initialized()
        assert self._session is not None
        message_events = [e for e in self._session.events if e.type == EventType.MESSAGE]

        # Use LLM callback if provided
        if llm_callback:
            messages = await self.get_messages_for_llm(include_system=False)
            return await llm_callback(messages)

        # Simple summary generation
        user_messages = [e for e in message_events if e.source == EventSource.USER]

        topics = []
        for event in user_messages:
            content = str(event.message)
            if "?" in content:
                question = content.split("?")[0].strip()
                if len(question) > 10:
                    topics.append(question[:50])

        if topics:
            summary = f"User discussed: {'; '.join(topics[:3])}"
            if len(topics) > 3:
                summary += f" and {len(topics) - 3} other topics"
        else:
            summary = f"Conversation with {len(user_messages)} user messages and {len(message_events) - len(user_messages)} responses"

        return summary

    async def _create_new_segment(self, llm_callback: Callable | None = None) -> str:
        """
        Create a new session segment with summary.

        Args:
            llm_callback: Optional async function to generate summary using an LLM.

        Returns:
            The new session ID.
        """
        # Use the instance callback if no explicit callback provided
        callback = llm_callback or self._summary_callback

        # Create summary of current session
        summary = await self._create_summary(callback)

        # Add summary to current session
        summary_event = SessionEvent(message=summary, source=EventSource.SYSTEM, type=EventType.SUMMARY)
        await self._ensure_initialized()
        assert self._session is not None
        await self._session.add_event_and_save(summary_event)

        # VM: create summary page, pin it, evict old segment pages
        if self._vm:
            summary_page = self._vm.create_page(
                content=summary,
                page_type=PageType.SUMMARY,
                importance=VM_IMPORTANCE_SUMMARY,
                hint=f"segment summary: {summary[:100]}",
            )
            self._vm.pin_page(summary_page.page_id)
            await self._vm.add_to_working_set(summary_page)
            self._vm.set_last_segment_summary(summary_page.page_id)
            await self._vm.evict_segment_pages(StorageTier.L2)

        # Create new session with current as parent
        new_session = await Session.create(parent_id=self._session_id)

        # Copy system prompt to new session
        if self._system_prompt:
            new_session.metadata.properties["system_prompt"] = self._system_prompt

        # Save new session
        await self._store.save(new_session)

        # Update our state
        old_session_id = self._session_id
        self._session_id = new_session.id
        self._session = new_session
        self._session_chain.append(self._session_id)
        self._total_segments += 1

        # VM: update session_id and advance turn
        if self._vm:
            self._vm.update_session_id(self._session_id or "")
            self._vm.new_turn()

        logger.info(f"Created new session segment: {old_session_id} -> {self._session_id}")
        return self._session_id

    async def user_says(self, message: str, **metadata) -> str:
        """
        Track a user message.

        Args:
            message: What the user said.
            **metadata: Optional metadata to attach to the event.

        Returns:
            The current session ID (may change in infinite mode).
        """
        # Check for segmentation before adding message
        if await self._should_create_new_segment():
            await self._create_new_segment()

        # VM: run demand paging pre-pass before event creation
        if self._vm:
            await self._vm.demand_pre_pass(message)

        await self._ensure_initialized()
        assert self._session is not None

        # Create and add the event
        event = await SessionEvent.create_with_tokens(
            message=message,
            prompt=message,
            model=self._default_model,
            source=EventSource.USER,
            type=EventType.MESSAGE,
        )

        # Add metadata
        for key, value in metadata.items():
            await event.set_metadata(key, value)

        await self._session.add_event_and_save(event)

        # Track in full conversation for infinite context
        if self._infinite_context:
            self._full_conversation.append(
                {
                    "role": MessageRole.USER.value,
                    "content": message,
                    "timestamp": event.timestamp.isoformat(),
                    "session_id": self._session_id,
                }
            )

        # Add to VM working set
        if self._vm:
            vm_page = self._vm.create_page(
                content=message,
                page_type=PageType.TRANSCRIPT,
                importance=VM_IMPORTANCE_USER,
                page_id=f"{PAGE_ID_PREFIX_MSG}_{event.id[:8]}" if event.id else None,
                hint=f"[user] {message[:120]}",
            )
            await self._vm.add_to_working_set(vm_page)

        return self._session_id

    async def ai_responds(
        self,
        response: str,
        model: str = "unknown",
        provider: str = "unknown",
        **metadata,
    ) -> str:
        """
        Track an AI response.

        Args:
            response: The AI's response.
            model: Model name used.
            provider: Provider name (openai, anthropic, etc).
            **metadata: Optional metadata to attach.

        Returns:
            The current session ID (may change in infinite mode).
        """
        # Check for segmentation before adding message
        if await self._should_create_new_segment():
            await self._create_new_segment()

        await self._ensure_initialized()
        assert self._session is not None

        # Create and add the event
        event = await SessionEvent.create_with_tokens(
            message=response,
            prompt="",
            completion=response,
            model=model,
            source=EventSource.LLM,
            type=EventType.MESSAGE,
        )

        # Add metadata
        full_metadata = {
            "model": model,
            "provider": provider,
            "timestamp": datetime.now().isoformat(),
            **metadata,
        }

        for key, value in full_metadata.items():
            await event.set_metadata(key, value)

        await self._session.add_event_and_save(event)

        # Track in full conversation for infinite context
        if self._infinite_context:
            self._full_conversation.append(
                {
                    "role": MessageRole.ASSISTANT.value,
                    "content": response,
                    "timestamp": event.timestamp.isoformat(),
                    "session_id": self._session_id,
                    "model": model,
                    "provider": provider,
                }
            )

        # Add to VM working set
        if self._vm:
            vm_page = self._vm.create_page(
                content=response,
                page_type=PageType.TRANSCRIPT,
                importance=VM_IMPORTANCE_AI,
                page_id=f"{PAGE_ID_PREFIX_MSG}_{event.id[:8]}" if event.id else None,
                hint=f"[assistant] {response[:120]}",
            )
            await self._vm.add_to_working_set(vm_page)

        return self._session_id

    async def tool_used(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        error: str | None = None,
        **metadata,
    ) -> str:
        """
        Track a tool call.

        Args:
            tool_name: Name of the tool called.
            arguments: Arguments passed to the tool.
            result: Result returned by the tool.
            error: Optional error message if tool failed.
            **metadata: Optional metadata to attach.

        Returns:
            The current session ID.
        """
        await self._ensure_initialized()
        assert self._session is not None

        tool_message = {
            "tool": tool_name,
            "arguments": arguments,
            "result": result,
            "error": error,
            "success": error is None,
        }

        # Create event with explicit type TOOL_CALL
        event = SessionEvent(
            message=tool_message,
            source=EventSource.SYSTEM,
            type=EventType.TOOL_CALL,
        )

        for key, value in metadata.items():
            await event.set_metadata(key, value)

        await self._session.add_event_and_save(event)

        tool_events = [e for e in self._session.events if e.type == EventType.TOOL_CALL]
        logger.debug(f"Tool events after adding: {len(tool_events)}")

        # Add to VM as artifact page
        if self._vm:
            tool_content = f"{tool_name}({arguments}) -> {result}"
            if error:
                tool_content += f" [error: {error}]"
            vm_page = self._vm.create_page(
                content=tool_content,
                page_type=PageType.ARTIFACT,
                importance=VM_IMPORTANCE_TOOL,
            )
            await self._vm.add_to_working_set(vm_page)

        return self._session_id

    async def get_messages_for_llm(self, include_system: bool = True) -> list[dict[str, str]]:
        """
        Get messages formatted for LLM consumption, optionally including system prompt.

        Args:
            include_system: Whether to include the system prompt as the first message.

        Returns:
            List of message dictionaries with 'role' and 'content' keys.
        """
        await self._ensure_initialized()
        assert self._session is not None

        # VM mode: replace system prompt with VM-packed context
        # Only when include_system=True (avoid breaking _create_summary)
        if self._vm and include_system:
            ctx = self._vm.build_context(system_prompt=self._system_prompt or "")
            messages: list[dict[str, str]] = [{"role": MessageRole.SYSTEM.value, "content": ctx["developer_message"]}]
            for event in self._session.events:
                if event.type == EventType.MESSAGE:
                    if event.source == EventSource.USER:
                        messages.append(
                            {
                                "role": MessageRole.USER.value,
                                "content": str(event.message),
                            }
                        )
                    elif event.source == EventSource.LLM:
                        messages.append(
                            {
                                "role": MessageRole.ASSISTANT.value,
                                "content": str(event.message),
                            }
                        )
            return messages

        messages = []

        # Add system prompt if available and requested (and not empty)
        if include_system and self._system_prompt and self._system_prompt.strip():
            messages.append({"role": MessageRole.SYSTEM.value, "content": self._system_prompt})

        # Add conversation messages
        for event in self._session.events:
            if event.type == EventType.MESSAGE:
                if event.source == EventSource.USER:
                    messages.append(
                        {
                            "role": MessageRole.USER.value,
                            "content": str(event.message),
                        }
                    )
                elif event.source == EventSource.LLM:
                    messages.append(
                        {
                            "role": MessageRole.ASSISTANT.value,
                            "content": str(event.message),
                        }
                    )

        return messages

    async def get_conversation(self, include_all_segments: bool | None = None) -> list[dict[str, Any]]:
        """
        Get conversation history.

        Args:
            include_all_segments: Include all segments (defaults to infinite_context setting).

        Returns:
            List of conversation turns.
        """
        if include_all_segments is None:
            include_all_segments = self._infinite_context

        if self._infinite_context and include_all_segments:
            # Return full conversation across all segments
            return self._full_conversation.copy()
        else:
            # Return current session only
            await self._ensure_initialized()
            assert self._session is not None
            conversation = []
            for event in self._session.events:
                if event.type == EventType.MESSAGE:
                    turn = {
                        "role": (
                            MessageRole.USER.value if event.source == EventSource.USER else MessageRole.ASSISTANT.value
                        ),
                        "content": str(event.message),
                        "timestamp": event.timestamp.isoformat(),
                    }
                    conversation.append(turn)

            return conversation

    async def get_session_chain(self) -> list[str]:
        """Get the chain of session IDs (infinite context only)."""
        if self._infinite_context:
            return self._session_chain.copy()
        else:
            return [self.session_id]

    async def get_stats(self, include_all_segments: bool | None = None) -> dict[str, Any]:
        """
        Get conversation statistics.

        Args:
            include_all_segments: Include all segments (defaults to infinite_context setting).

        Returns:
            Dictionary with conversation stats including:
            - session_id: Current session ID
            - total_messages: Total number of messages
            - user_messages: Number of user messages
            - ai_messages: Number of AI messages
            - tool_calls: Number of tool calls
            - total_tokens: Total tokens used
            - estimated_cost: Estimated cost in USD
            - created_at: Session creation time
            - last_update: Last update time
            - session_segments: Number of segments (infinite context)
            - infinite_context: Whether infinite context is enabled
        """
        if include_all_segments is None:
            include_all_segments = self._infinite_context

        await self._ensure_initialized()
        assert self._session is not None

        if self._infinite_context and include_all_segments:
            # For infinite context, build the complete chain if needed
            if len(self._session_chain) < self._total_segments:
                # Need to reconstruct the chain
                store = self._store
                chain: list[str] = []
                current_id: str | None = self._session_id

                # Walk backwards to find all segments
                while current_id:
                    chain.insert(0, current_id)
                    session = await store.get(current_id)
                    if session and session.parent_id:
                        current_id = session.parent_id
                    else:
                        break

                self._session_chain = chain
                self._total_segments = len(chain)

            # Calculate stats across all segments
            user_messages = len([t for t in self._full_conversation if t["role"] == "user"])
            ai_messages = len([t for t in self._full_conversation if t["role"] == "assistant"])

            # Get token/cost stats by loading all sessions in chain
            total_tokens = 0
            total_cost = 0.0
            total_events = 0
            tool_calls = 0

            store = self._store

            for session_id in self._session_chain:
                try:
                    # For the current session, use self._session directly
                    # to ensure we have the latest in-memory state
                    sess: Session | None
                    if session_id == self._session_id:
                        sess = self._session
                    else:
                        sess = await store.get(session_id)

                    if sess:
                        total_tokens += sess.total_tokens
                        total_cost += sess.total_cost
                        total_events += len(sess.events)
                        tool_calls += sum(1 for e in sess.events if e.type == EventType.TOOL_CALL)
                except Exception as e:
                    logger.warning(f"Failed to load session {session_id} in chain: {e}")

            return {
                "session_id": self._session_id,
                "session_segments": self._total_segments,
                "session_chain": self._session_chain.copy(),
                "total_messages": user_messages + ai_messages,
                "total_events": total_events,
                "user_messages": user_messages,
                "ai_messages": ai_messages,
                "tool_calls": tool_calls,
                "total_tokens": total_tokens,
                "estimated_cost": total_cost,
                "created_at": self._session.metadata.created_at.isoformat(),
                "last_update": self._session.last_update_time.isoformat(),
                "infinite_context": True,
            }
        else:
            # Current session stats only
            user_messages = sum(
                1 for e in self._session.events if e.type == EventType.MESSAGE and e.source == EventSource.USER
            )
            ai_messages = sum(
                1 for e in self._session.events if e.type == EventType.MESSAGE and e.source == EventSource.LLM
            )
            tool_calls = sum(1 for e in self._session.events if e.type == EventType.TOOL_CALL)

            return {
                "session_id": self._session.id,
                "session_segments": 1,
                "total_messages": user_messages + ai_messages,
                "total_events": len(self._session.events),
                "user_messages": user_messages,
                "ai_messages": ai_messages,
                "tool_calls": tool_calls,
                "total_tokens": self._session.total_tokens,
                "estimated_cost": self._session.total_cost,
                "created_at": self._session.metadata.created_at.isoformat(),
                "last_update": self._session.last_update_time.isoformat(),
                "infinite_context": self._infinite_context,
            }

    def set_summary_callback(self, callback: Callable[[list[dict]], str]) -> None:
        """
        Set a custom callback for generating summaries in infinite context mode.

        Args:
            callback: Async function that takes messages and returns a summary string.
        """
        self._summary_callback = callback

    async def load_session_chain(self) -> None:
        """
        Load the full session chain for infinite context sessions.

        This reconstructs the conversation history from all linked sessions.
        """
        if not self._infinite_context:
            return

        await self._ensure_initialized()
        store = self._store

        # Start from current session and work backwards
        assert self._session_id is not None
        current_id: str = self._session_id
        chain: list[str] = [current_id]
        conversation: list[dict[str, Any]] = []

        while current_id:
            session = await store.get(current_id)
            if not session:
                break

            # Extract messages from this session
            for event in reversed(session.events):
                if event.type == EventType.MESSAGE:
                    conversation.insert(
                        0,
                        {
                            "role": (
                                MessageRole.USER.value
                                if event.source == EventSource.USER
                                else MessageRole.ASSISTANT.value
                            ),
                            "content": str(event.message),
                            "timestamp": event.timestamp.isoformat(),
                            "session_id": current_id,
                        },
                    )

            # Move to parent
            if session.parent_id:
                chain.insert(0, session.parent_id)
                current_id = session.parent_id
            else:
                break

        self._session_chain = chain
        self._full_conversation = conversation
        self._total_segments = len(chain)
