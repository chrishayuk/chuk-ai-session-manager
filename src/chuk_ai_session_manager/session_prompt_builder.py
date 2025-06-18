# chuk_ai_session_manager/session_prompt_builder.py
"""
Build optimized prompts for LLM calls from Session objects with async support.

This module provides flexible prompt construction from session data,
with support for token management, relevance-based selection,
and hierarchical context awareness.
"""

from __future__ import annotations
import json
import logging
from typing import List, Dict, Any, Optional, Literal, Union
from enum import Enum
import asyncio 

from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.token_usage import TokenUsage
from chuk_ai_session_manager.session_storage import get_backend, ChukSessionsStore

logger = logging.getLogger(__name__)

class PromptStrategy(str, Enum):
    """Different strategies for building prompts."""
    MINIMAL = "minimal"         # Original minimal approach
    TASK_FOCUSED = "task"       # Focus on the task with minimal context
    TOOL_FOCUSED = "tool"       # Emphasize tool usage and results
    CONVERSATION = "conversation"  # Include more conversation history
    HIERARCHICAL = "hierarchical"  # Include parent session context


async def build_prompt_from_session(
    session: Session,
    strategy: Union[PromptStrategy, str] = PromptStrategy.MINIMAL,
    max_tokens: Optional[int] = None,
    model: str = "gpt-3.5-turbo",
    include_parent_context: bool = False,
    current_query: Optional[str] = None,
    max_history: int = 5  # Add this parameter for conversation strategy
) -> List[Dict[str, str]]:
    """
    Build a prompt for the next LLM call from a Session asynchronously.
    
    Args:
        session: The session to build a prompt from
        strategy: Prompt building strategy to use
        max_tokens: Maximum tokens to include (if specified)
        model: Model to use for token counting
        include_parent_context: Whether to include context from parent sessions
        current_query: Current user query for relevance-based context selection
        max_history: Maximum number of messages to include for conversation strategy
        
    Returns:
        A list of message dictionaries suitable for LLM API calls
    """
    if not session.events:
        return []
    
    # Convert string strategy to enum if needed
    if isinstance(strategy, str):
        try:
            strategy = PromptStrategy(strategy)
        except ValueError:
            logger.warning(f"Unknown strategy '{strategy}', falling back to MINIMAL")
            strategy = PromptStrategy.MINIMAL
    
    # Use the appropriate strategy
    if strategy == PromptStrategy.MINIMAL:
        return await _build_minimal_prompt(session)
    elif strategy == PromptStrategy.TASK_FOCUSED:
        return await _build_task_focused_prompt(session)
    elif strategy == PromptStrategy.TOOL_FOCUSED:
        return await _build_tool_focused_prompt(session)
    elif strategy == PromptStrategy.CONVERSATION:
        return await _build_conversation_prompt(session, max_history)
    elif strategy == PromptStrategy.HIERARCHICAL:
        return await _build_hierarchical_prompt(session, include_parent_context)
    else:
        # Default to minimal
        return await _build_minimal_prompt(session)


async def _build_minimal_prompt(session: Session) -> List[Dict[str, str]]:
    """
    Build a minimal prompt from a session.
    
    This follows the original implementation's approach:
    - Include the first USER message (task)
    - Include the latest assistant MESSAGE with content set to None
    - Include TOOL_CALL children as tool role messages
    - Fall back to SUMMARY retry note if no TOOL_CALL children exist
    """
    # First USER message
    first_user = next(
        (
            e
            for e in session.events
            if e.type == EventType.MESSAGE and e.source == EventSource.USER
        ),
        None,
    )

    # Latest assistant MESSAGE
    assistant_msg = next(
        (
            ev
            for ev in reversed(session.events)
            if ev.type == EventType.MESSAGE and ev.source != EventSource.USER
        ),
        None,
    )
    
    if assistant_msg is None:
        # Only the user message exists so far
        return [{"role": "user", "content": _extract_content(first_user.message)}] if first_user else []

    # Children of that assistant
    children = [
        e
        for e in session.events
        if e.metadata.get("parent_event_id") == assistant_msg.id
    ]
    tool_calls = [c for c in children if c.type == EventType.TOOL_CALL]
    summaries = [c for c in children if c.type == EventType.SUMMARY]

    # Assemble prompt
    prompt: List[Dict[str, str]] = []
    if first_user:
        prompt.append({"role": "user", "content": _extract_content(first_user.message)})

    # ALWAYS add the assistant marker - but strip its free text
    prompt.append({"role": "assistant", "content": None})

    if tool_calls:
        for tc in tool_calls:
            # Extract relevant information from the tool call
            # Handle both new and legacy formats
            if isinstance(tc.message, dict):
                tool_name = tc.message.get("tool_name", tc.message.get("tool", "unknown"))
                tool_result = tc.message.get("result", {})
            else:
                # Legacy format or unexpected type
                tool_name = "unknown"
                tool_result = tc.message

            prompt.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": json.dumps(tool_result, default=str),
                }
            )
    elif summaries:
        # Use the latest summary
        summary = summaries[-1]
        if isinstance(summary.message, dict) and "note" in summary.message:
            prompt.append({"role": "system", "content": summary.message["note"]})
        else:
            # Handle legacy or unexpected format
            prompt.append({"role": "system", "content": str(summary.message)})

    return prompt


def _extract_content(message: Any) -> str:
    """
    Extract content string from a message that could be a string or dict.
    
    Args:
        message: The message content (string, dict, or other)
        
    Returns:
        The extracted content as a string
    """
    if isinstance(message, str):
        return message
    elif isinstance(message, dict) and "content" in message:
        return message["content"]
    else:
        return str(message)


async def _build_task_focused_prompt(session: Session) -> List[Dict[str, str]]:
    """
    Build a task-focused prompt.
    
    This strategy emphasizes the original task and latest context:
    - Includes the first USER message as the main task
    - Includes the most recent USER message for current context
    - Includes only the most recent and successful tool results
    """
    # Get first and most recent user messages
    user_messages = [
        e for e in session.events
        if e.type == EventType.MESSAGE and e.source == EventSource.USER
    ]
    
    if not user_messages:
        return []
        
    first_user = user_messages[0]
    latest_user = user_messages[-1] if len(user_messages) > 1 else None
    
    # Latest assistant MESSAGE
    assistant_msg = next(
        (
            ev
            for ev in reversed(session.events)
            if ev.type == EventType.MESSAGE and ev.source != EventSource.USER
        ),
        None,
    )
    
    # Build prompt
    prompt = []
    
    # Always include the first user message (the main task)
    prompt.append({"role": "user", "content": _extract_content(first_user.message)})
    
    # Include the latest user message if different from the first
    if latest_user and latest_user.id != first_user.id:
        prompt.append({"role": "user", "content": _extract_content(latest_user.message)})
    
    # Include assistant response placeholder
    if assistant_msg:
        prompt.append({"role": "assistant", "content": None})
        
        # Find successful tool calls
        children = [
            e for e in session.events
            if e.metadata.get("parent_event_id") == assistant_msg.id
        ]
        tool_calls = [c for c in children if c.type == EventType.TOOL_CALL]
        
        # Only include successful tool results
        for tc in tool_calls:
            # Extract and check if result indicates success
            if isinstance(tc.message, dict):
                tool_name = tc.message.get("tool_name", tc.message.get("tool", "unknown"))
                tool_result = tc.message.get("result", {})
                
                # Skip error results
                if isinstance(tool_result, dict) and tool_result.get("status") == "error":
                    continue
                    
                prompt.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": json.dumps(tool_result, default=str),
                })
    
    return prompt


async def _build_tool_focused_prompt(session: Session) -> List[Dict[str, str]]:
    """
    Build a tool-focused prompt.
    
    This strategy emphasizes tool usage:
    - Includes the latest user query
    - Includes detailed information about tool calls and results
    - Includes error information from failed tool calls
    """
    # Get the latest user message
    latest_user = next(
        (e for e in reversed(session.events) 
         if e.type == EventType.MESSAGE and e.source == EventSource.USER),
        None
    )
    
    if not latest_user:
        return []
    
    # Get the latest assistant message
    assistant_msg = next(
        (ev for ev in reversed(session.events)
         if ev.type == EventType.MESSAGE and ev.source != EventSource.USER),
        None
    )
    
    # Build prompt
    prompt = []
    
    # Include user message
    prompt.append({"role": "user", "content": _extract_content(latest_user.message)})
    
    # Include assistant placeholder
    if assistant_msg:
        prompt.append({"role": "assistant", "content": None})
        
        # Get all tool calls for this assistant
        children = [
            e for e in session.events
            if e.metadata.get("parent_event_id") == assistant_msg.id
        ]
        tool_calls = [c for c in children if c.type == EventType.TOOL_CALL]
        
        # Add all tool calls with status information
        for tc in tool_calls:
            if isinstance(tc.message, dict):
                tool_name = tc.message.get("tool_name", tc.message.get("tool", "unknown"))
                tool_result = tc.message.get("result", {})
                error = tc.message.get("error", None)
                
                # Include status information in the tool response
                content = tool_result
                if error:
                    content = {"error": error, "details": tool_result}
                
                prompt.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": json.dumps(content, default=str),
                })
    
    return prompt


async def _build_conversation_prompt(
    session: Session, 
    max_history: int = 5
) -> List[Dict[str, str]]:
    """
    Build a conversation-style prompt with recent history.
    
    This strategy creates a more natural conversation:
    - Includes up to max_history recent messages in order
    - Preserves conversation flow
    - Still handles tool calls appropriately
    """
    # Get relevant message events
    message_events = [
        e for e in session.events
        if e.type == EventType.MESSAGE
    ]
    
    # Take the most recent messages
    recent_messages = message_events[-max_history:] if len(message_events) > max_history else message_events
    
    # Build the conversation history
    prompt = []
    for i, msg in enumerate(recent_messages):
        role = "user" if msg.source == EventSource.USER else "assistant"
        content = _extract_content(msg.message)
        
        # For the last assistant message, set content to None and add tool calls
        if (role == "assistant" and 
            msg == recent_messages[-1] and 
            msg.source != EventSource.USER):
            
            # Add the message first with None content
            prompt.append({"role": role, "content": None})
            
            # Add tool call results for this assistant message
            tool_calls = [
                e for e in session.events
                if e.type == EventType.TOOL_CALL and e.metadata.get("parent_event_id") == msg.id
            ]
            
            # Add tool results
            for tc in tool_calls:
                if isinstance(tc.message, dict):
                    tool_name = tc.message.get("tool_name", tc.message.get("tool", "unknown"))
                    tool_result = tc.message.get("result", {})
                    
                    prompt.append({
                        "role": "tool",
                        "name": tool_name,
                        "content": json.dumps(tool_result, default=str),
                    })
        else:
            # Regular message
            prompt.append({"role": role, "content": content})
    
    return prompt


async def _build_hierarchical_prompt(
    session: Session,
    include_parent_context: bool = True
) -> List[Dict[str, str]]:
    """
    Build a prompt that includes hierarchical context.
    
    This strategy leverages the session hierarchy:
    - Starts with the minimal prompt
    - Includes summaries from parent sessions if available
    """
    # Start with the minimal prompt
    prompt = await _build_minimal_prompt(session)
    
    # If parent context is enabled and session has a parent
    if include_parent_context and session.parent_id:
        try:
            # Get the storage backend and create store
            backend = get_backend()
            store = ChukSessionsStore(backend)
            parent = await store.get(session.parent_id)
            
            if parent:
                # Find the most recent summary in parent
                summary_event = next(
                    (e for e in reversed(parent.events) 
                     if e.type == EventType.SUMMARY),
                    None
                )
                
                if summary_event:
                    # Extract summary content
                    summary_content = summary_event.message
                    if isinstance(summary_content, dict) and "note" in summary_content:
                        summary_content = summary_content["note"]
                    elif isinstance(summary_content, dict) and "content" in summary_content:
                        summary_content = summary_content["content"]
                    else:
                        summary_content = str(summary_content)
                        
                    # Add parent context at the beginning
                    prompt.insert(0, {
                        "role": "system",
                        "content": f"Context from previous conversation: {summary_content}"
                    })
        except Exception as e:
            # If we can't load parent context, just continue with minimal prompt
            logger.warning(f"Could not load parent context for session {session.parent_id}: {e}")
    
    return prompt

async def truncate_prompt_to_token_limit(
    prompt: List[Dict[str, str]],
    max_tokens: int,
    model: str = "gpt-3.5-turbo",
) -> List[Dict[str, str]]:
    """
    Trim a prompt so its total token count is ≤ `max_tokens`.

    Strategy:
    • If already within limit → return unchanged
    • Otherwise keep:
        - the very first user message
        - everything from the last assistant message onward
        - (optionally) one tool message so the model still sees a result
    """
    if not prompt:
        return []

    # ------------------------------------------------------------------ #
    # quick overall count
    text = "\n".join(f"{m.get('role', 'unknown')}: {m.get('content') or ''}" for m in prompt)
    total = TokenUsage.count_tokens(text, model)
    total = await total if asyncio.iscoroutine(total) else total
    if total <= max_tokens:
        return prompt

    # ------------------------------------------------------------------ #
    # decide which messages to keep
    first_user_idx = next((i for i, m in enumerate(prompt) if m["role"] == "user"), None)
    last_asst_idx = next(
        (len(prompt) - 1 - i for i, m in enumerate(reversed(prompt)) if m["role"] == "assistant"),
        None,
    )

    kept: List[Dict[str, str]] = []
    if first_user_idx is not None:
        kept.append(prompt[first_user_idx])
    if last_asst_idx is not None:
        kept.extend(prompt[last_asst_idx:])

    # ------------------------------------------------------------------ #
    # re-count and maybe drop / add tool messages
    remaining = TokenUsage.count_tokens(str(kept), model)
    remaining = await remaining if asyncio.iscoroutine(remaining) else remaining

    if remaining > max_tokens:
        # remove any tool messages we just added
        kept = [m for m in kept if m["role"] != "tool"]
        # but guarantee at least one tool message (the first) if it'll fit
        first_tool = next((m for m in prompt if m["role"] == "tool"), None)
        if first_tool:
            kept.append(first_tool)

    return kept