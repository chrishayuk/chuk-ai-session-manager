# src/chuk_ai_session_manager/api/simple_api.py
"""
Simple API convenience functions for the CHUK AI Session Manager.

This module provides easy-to-use functions for common session management tasks,
building on top of the SessionManager class.

Usage:
    from chuk_ai_session_manager import track_conversation
    
    # Quick tracking
    await track_conversation("Hello!", "Hi there!")
    
    # Track with model info
    await track_conversation(
        "What's the weather?",
        "It's sunny and 72°F",
        model="gpt-4",
        provider="openai"
    )
    
    # Infinite context
    await track_infinite_conversation(
        "Tell me a long story",
        "Once upon a time...",
        token_threshold=4000
    )
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, Callable

from chuk_ai_session_manager.session_manager import SessionManager

logger = logging.getLogger(__name__)


async def track_conversation(
    user_message: str,
    ai_response: str,
    model: str = "unknown",
    provider: str = "unknown",
    session_id: Optional[str] = None,
    infinite_context: bool = False,
    token_threshold: int = 4000
) -> str:
    """
    Quick way to track a single conversation turn.
    
    This is the simplest way to track a conversation exchange between
    a user and an AI assistant.
    
    Args:
        user_message: What the user said.
        ai_response: What the AI responded.
        model: The model used (e.g., "gpt-4", "claude-3").
        provider: The provider (e.g., "openai", "anthropic").
        session_id: Optional existing session ID to continue.
        infinite_context: Enable infinite context support.
        token_threshold: Token limit for infinite context segmentation.
        
    Returns:
        The session ID (useful for continuing the conversation).
        
    Example:
        ```python
        session_id = await track_conversation(
            "What's the capital of France?",
            "The capital of France is Paris.",
            model="gpt-3.5-turbo",
            provider="openai"
        )
        ```
    """
    sm = SessionManager(
        session_id=session_id,
        infinite_context=infinite_context,
        token_threshold=token_threshold
    )
    await sm.user_says(user_message)
    session_id = await sm.ai_responds(ai_response, model=model, provider=provider)
    return session_id


async def track_llm_call(
    user_input: str,
    llm_function: Callable[[str], Union[str, Any]],
    model: str = "unknown",
    provider: str = "unknown",
    session_manager: Optional[SessionManager] = None,
    session_id: Optional[str] = None,
    infinite_context: bool = False,
    token_threshold: int = 4000
) -> tuple[str, str]:
    """
    Track an LLM call automatically.
    
    This function wraps your LLM call and automatically tracks both the
    input and output in a session.
    
    Args:
        user_input: The user's input to the LLM.
        llm_function: Function that calls the LLM (sync or async).
        model: The model being used.
        provider: The provider being used.
        session_manager: Optional existing SessionManager to use.
        session_id: Optional session ID if not using session_manager.
        infinite_context: Enable infinite context support.
        token_threshold: Token limit for infinite context.
        
    Returns:
        Tuple of (response_text, session_id).
        
    Example:
        ```python
        async def call_openai(prompt):
            # Your OpenAI call here
            return response.choices[0].message.content
            
        response, session_id = await track_llm_call(
            "Explain quantum computing",
            call_openai,
            model="gpt-4",
            provider="openai"
        )
        ```
    """
    if session_manager is None:
        session_manager = SessionManager(
            session_id=session_id,
            infinite_context=infinite_context,
            token_threshold=token_threshold
        )
    
    await session_manager.user_says(user_input)
    
    # Call the LLM function
    if asyncio.iscoroutinefunction(llm_function):
        ai_response = await llm_function(user_input)
    else:
        ai_response = llm_function(user_input)
    
    # Handle different response formats
    if isinstance(ai_response, dict) and "choices" in ai_response:
        # OpenAI format
        response_text = ai_response["choices"][0]["message"]["content"]
    elif hasattr(ai_response, "content"):
        # Object with content attribute
        response_text = ai_response.content
    else:
        # Plain string or other
        response_text = str(ai_response)
    
    session_id = await session_manager.ai_responds(
        response_text, model=model, provider=provider
    )
    
    return response_text, session_id


async def quick_conversation(
    user_message: str,
    ai_response: str,
    model: str = "unknown",
    provider: str = "unknown",
    infinite_context: bool = False
) -> Dict[str, Any]:
    """
    Quickest way to track a conversation and get basic stats.
    
    This is perfect for one-off tracking where you want immediate
    statistics about the conversation.
    
    Args:
        user_message: What the user said.
        ai_response: What the AI responded.
        model: The model used.
        provider: The provider used.
        infinite_context: Enable infinite context support.
        
    Returns:
        Dictionary with conversation statistics.
        
    Example:
        ```python
        stats = await quick_conversation(
            "Hello!",
            "Hi there! How can I help you today?",
            model="gpt-3.5-turbo"
        )
        print(f"Tokens used: {stats['total_tokens']}")
        print(f"Cost: ${stats['estimated_cost']:.4f}")
        ```
    """
    # Create a new session manager
    sm = SessionManager(infinite_context=infinite_context)
    
    # Track the conversation
    await sm.user_says(user_message)
    await sm.ai_responds(ai_response, model=model, provider=provider)
    
    # Return stats directly
    return await sm.get_stats()


async def track_infinite_conversation(
    user_message: str,
    ai_response: str,
    model: str = "unknown",
    provider: str = "unknown",
    session_id: Optional[str] = None,
    token_threshold: int = 4000,
    max_turns: int = 20
) -> str:
    """
    Track a conversation with infinite context support.
    
    This automatically handles long conversations by creating new
    session segments when limits are reached, maintaining context
    through summaries.
    
    Args:
        user_message: What the user said.
        ai_response: What the AI responded.
        model: The model used.
        provider: The provider used.
        session_id: Optional existing session ID to continue.
        token_threshold: Create new segment after this many tokens.
        max_turns: Create new segment after this many turns.
        
    Returns:
        The current session ID (may be different if segmented).
        
    Example:
        ```python
        # First message
        session_id = await track_infinite_conversation(
            "Tell me about the history of computing",
            "Computing history begins with...",
            model="gpt-4"
        )
        
        # Continue the conversation
        session_id = await track_infinite_conversation(
            "What about quantum computers?",
            "Quantum computing represents...",
            session_id=session_id,
            model="gpt-4"
        )
        ```
    """
    return await track_conversation(
        user_message, 
        ai_response, 
        model=model, 
        provider=provider,
        session_id=session_id,
        infinite_context=True, 
        token_threshold=token_threshold
    )


async def track_tool_use(
    tool_name: str,
    arguments: Dict[str, Any],
    result: Any,
    session_id: Optional[str] = None,
    error: Optional[str] = None,
    **metadata
) -> str:
    """
    Track a tool/function call in a session.
    
    Args:
        tool_name: Name of the tool that was called.
        arguments: Arguments passed to the tool.
        result: Result returned by the tool.
        session_id: Optional existing session ID.
        error: Optional error if the tool failed.
        **metadata: Additional metadata to store.
        
    Returns:
        The session ID.
        
    Example:
        ```python
        session_id = await track_tool_use(
            "calculator",
            {"operation": "add", "a": 5, "b": 3},
            {"result": 8},
            session_id=session_id
        )
        ```
    """
    sm = SessionManager(session_id=session_id)
    return await sm.tool_used(
        tool_name=tool_name,
        arguments=arguments,
        result=result,
        error=error,
        **metadata
    )


async def get_session_stats(
    session_id: str,
    include_all_segments: bool = False
) -> Dict[str, Any]:
    """
    Get statistics for an existing session.
    
    Args:
        session_id: The session ID to get stats for.
        include_all_segments: For infinite context sessions, include all segments.
        
    Returns:
        Dictionary with session statistics.
        
    Example:
        ```python
        stats = await get_session_stats("session-123")
        print(f"Total messages: {stats['total_messages']}")
        print(f"Total cost: ${stats['estimated_cost']:.4f}")
        ```
    """
    sm = SessionManager(session_id=session_id)
    return await sm.get_stats(include_all_segments=include_all_segments)


async def get_conversation_history(
    session_id: str,
    include_all_segments: bool = False
) -> List[Dict[str, Any]]:
    """
    Get the conversation history for a session.
    
    Args:
        session_id: The session ID to get history for.
        include_all_segments: For infinite context sessions, include all segments.
        
    Returns:
        List of conversation turns.
        
    Example:
        ```python
        history = await get_conversation_history("session-123")
        for turn in history:
            print(f"{turn['role']}: {turn['content']}")
        ```
    """
    sm = SessionManager(session_id=session_id)
    return await sm.get_conversation(include_all_segments=include_all_segments)


# Backwards compatibility aliases
track_llm_interaction = track_llm_call
quick_stats = quick_conversation