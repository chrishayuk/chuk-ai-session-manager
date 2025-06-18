#!/usr/bin/env python3
"""
Session-aware Tool Processor for chuk_tool_processor 0.1.x

This module provides a thin wrapper around chuk_tool_processor that adds
comprehensive session tracking without duplicating the tool processor's
built-in reliability features.

Separation of Concerns:
- chuk_tool_processor: Handles retries, timeouts, error recovery, execution strategies
- SessionAwareToolProcessor: Handles session tracking, caching, and audit trail

Usage:
    processor = await SessionAwareToolProcessor.create(session_id)
    results = await processor.process_llm_message(openai_message)
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from chuk_tool_processor.core.processor import ToolProcessor
from chuk_tool_processor.models.tool_call import ToolCall
from chuk_tool_processor.models.tool_result import ToolResult

from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.session_storage import get_backend, ChukSessionsStore

logger = logging.getLogger(__name__)


class SessionAwareToolProcessor:
    """
    Session tracking wrapper for chuk_tool_processor.
    
    This class focuses solely on session tracking and caching while delegating
    all execution reliability (retries, timeouts, error handling) to the
    underlying chuk_tool_processor.
    
    Features:
    - Complete session tracking and audit trail
    - Optional result caching with TTL
    - Performance metrics collection
    - Parent-child event relationships
    - Proper status indicators (✅/❌)
    """

    def __init__(
        self,
        session_id: str,
        *,
        enable_caching: bool = True,
        cache_ttl_seconds: int = 300,  # 5 minutes
        enable_metrics: bool = True,
    ) -> None:
        """
        Initialize the session-aware tool processor.
        
        Args:
            session_id: ID of the session to track operations in
            enable_caching: Whether to cache tool results
            cache_ttl_seconds: Cache time-to-live in seconds
            enable_metrics: Whether to collect performance metrics
        """
        self.session_id = session_id
        self.enable_caching = enable_caching
        self.cache_ttl_seconds = cache_ttl_seconds
        self.enable_metrics = enable_metrics
        
        # Simple cache with TTL
        self.cache: Dict[str, Dict[str, Any]] = {}  # key -> {result, timestamp}
        
        # Basic metrics
        self.metrics = {
            "total_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "successful_calls": 0,
            "failed_calls": 0,
        }

        # Initialize tool processor - let it handle all execution concerns
        self._tp = ToolProcessor()
        if not hasattr(self._tp, "executor"):
            raise AttributeError(
                "Installed chuk_tool_processor is missing `.executor` attribute. "
                "Please ensure you have a compatible version installed."
            )

    @classmethod
    async def create(cls, session_id: str, **kwargs) -> "SessionAwareToolProcessor":
        """
        Create a session-aware tool processor with session validation.
        
        Args:
            session_id: ID of the session to track operations in
            **kwargs: Additional arguments passed to __init__
            
        Returns:
            Initialized SessionAwareToolProcessor instance
            
        Raises:
            ValueError: If the session doesn't exist
        """
        backend = get_backend()
        store = ChukSessionsStore(backend)
        session = await store.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        logger.info(f"Created SessionAwareToolProcessor for session {session_id[:8]}...")
        return cls(session_id=session_id, **kwargs)

    def _get_cache_key(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Generate a deterministic cache key for tool calls."""
        key_data = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if a cache entry is still valid based on TTL."""
        if not self.enable_caching:
            return False
        
        timestamp = cache_entry.get("timestamp", 0)
        age = datetime.now().timestamp() - timestamp
        return age < self.cache_ttl_seconds

    async def _log_tool_call_event(
        self,
        session,
        parent_id: str,
        result: ToolResult,
        cached: bool = False,
    ) -> None:
        """
        Log a tool execution event to the session.
        
        Args:
            session: Session object to log to
            parent_id: ID of the parent LLM message event
            result: Tool execution result from chuk_tool_processor
            cached: Whether this was a cache hit
        """
        # Convert result to string for session storage
        result_str = str(result.result) if result.result is not None else "null"
        
        # Create the event message with all relevant information
        event_message = {
            "tool": result.tool,
            "arguments": getattr(result, "arguments", None),
            "result": result_str,
            "error": result.error,
            "cached": cached,
            "success": result.error is None,
        }

        # Create the session event with token tracking
        event = await SessionEvent.create_with_tokens(
            message=event_message,
            prompt=f"{result.tool}({json.dumps(getattr(result, 'arguments', None), default=str)})",
            completion=result_str,
            model="tool-execution",
            source=EventSource.SYSTEM,
            type=EventType.TOOL_CALL,
        )
        
        # Add comprehensive metadata
        await event.set_metadata("parent_event_id", parent_id)
        await event.set_metadata("tool_name", result.tool)
        await event.set_metadata("cached", cached)
        await event.set_metadata("success", result.error is None)
        await event.set_metadata("timestamp", datetime.now().isoformat())
        
        if result.error:
            await event.set_metadata("error_type", type(result.error).__name__ if hasattr(result.error, '__class__') else "unknown")
        
        # Save to session
        await session.add_event_and_save(event)

    async def process_llm_message(
        self, 
        llm_msg: Dict[str, Any], 
        context: Optional[Dict[str, Any]] = None
    ) -> List[ToolResult]:
        """
        Process tool calls from an LLM message with session tracking.
        
        This method:
        1. Logs the LLM message to the session
        2. Checks cache for each tool call
        3. Delegates execution to chuk_tool_processor (which handles retries/reliability)
        4. Logs all results to the session
        5. Updates cache and metrics
        
        Args:
            llm_msg: OpenAI-style message with tool_calls
            context: Optional context information for logging
            
        Returns:
            List of tool execution results from chuk_tool_processor
            
        Raises:
            ValueError: If the session is not found
        """
        # Get the session
        backend = get_backend()
        store = ChukSessionsStore(backend)
        session = await store.get(self.session_id)
        if not session:
            raise ValueError(f"Session {self.session_id} not found")

        # Log the LLM message as a parent event
        parent_event = await SessionEvent.create_with_tokens(
            message=llm_msg,
            prompt="",  # LLM message doesn't have a prompt
            completion=json.dumps(llm_msg, ensure_ascii=False),
            model="gpt-4o-mini",  # Could be parameterized
            source=EventSource.LLM,
            type=EventType.MESSAGE,
        )
        
        # Add context metadata if provided
        if context:
            for key, value in context.items():
                await parent_event.set_metadata(key, value)
        
        await session.add_event_and_save(parent_event)

        # Extract tool calls
        tool_calls = llm_msg.get("tool_calls", [])
        if not tool_calls:
            logger.debug("No tool calls found in LLM message")
            return []

        logger.info(f"Processing {len(tool_calls)} tool calls for session {self.session_id[:8]}...")
        
        results: List[ToolResult] = []
        chuk_tool_calls: List[ToolCall] = []
        cache_info: List[Dict[str, Any]] = []  # Track which calls are cached
        
        # Check cache for each tool call first
        for call in tool_calls:
            function = call.get("function", {})
            tool_name = function.get("name", "unknown_tool")
            
            try:
                arguments = json.loads(function.get("arguments", "{}"))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse arguments for {tool_name}: {e}")
                arguments = {"raw_arguments": function.get("arguments", "")}

            if self.enable_metrics:
                self.metrics["total_calls"] += 1

            # Check cache
            cache_key = self._get_cache_key(tool_name, arguments) if self.enable_caching else None
            
            if cache_key and cache_key in self.cache:
                cache_entry = self.cache[cache_key]
                if self._is_cache_valid(cache_entry):
                    # Cache hit!
                    cached_result = cache_entry["result"]
                    logger.debug(f"Cache hit for {tool_name}")
                    
                    if self.enable_metrics:
                        self.metrics["cache_hits"] += 1
                        if cached_result.error is None:
                            self.metrics["successful_calls"] += 1
                        else:
                            self.metrics["failed_calls"] += 1
                    
                    # Log the cached result
                    await self._log_tool_call_event(
                        session, parent_event.id, cached_result, cached=True
                    )
                    results.append(cached_result)
                    
                    # Mark this call as cached
                    cache_info.append({"cached": True, "result": cached_result})
                    continue
                else:
                    # Cache expired
                    del self.cache[cache_key]

            # Cache miss - prepare for execution
            if self.enable_metrics:
                self.metrics["cache_misses"] += 1
            
            chuk_tool_calls.append(ToolCall(tool=tool_name, arguments=arguments))
            cache_info.append({"cached": False, "cache_key": cache_key})

        # Execute non-cached calls with chuk_tool_processor
        # This handles ALL reliability concerns (retries, timeouts, error recovery)
        if chuk_tool_calls:
            logger.debug(f"Executing {len(chuk_tool_calls)} tools via chuk_tool_processor")
            
            try:
                execution_results = await self._tp.executor.execute(chuk_tool_calls)
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                # Create error results for each failed call
                execution_results = [
                    ToolResult(tool=tc.tool, result=None, error=str(e))
                    for tc in chuk_tool_calls
                ]
            
            # Process execution results
            exec_index = 0
            for i, info in enumerate(cache_info):
                if info["cached"]:
                    continue  # Already handled above
                
                if exec_index < len(execution_results):
                    result = execution_results[exec_index]
                    exec_index += 1
                    
                    # Update metrics
                    if self.enable_metrics:
                        if result.error is None:
                            self.metrics["successful_calls"] += 1
                        else:
                            self.metrics["failed_calls"] += 1
                    
                    # Cache successful results (if caching enabled)
                    if self.enable_caching and result.error is None and "cache_key" in info:
                        self.cache[info["cache_key"]] = {
                            "result": result,
                            "timestamp": datetime.now().timestamp()
                        }
                        logger.debug(f"Cached result for {result.tool}")
                    
                    # Log the execution result
                    await self._log_tool_call_event(
                        session, parent_event.id, result, cached=False
                    )
                    results.append(result)

        logger.info(f"Completed processing {len(tool_calls)} tool calls")
        return results

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this processor."""
        if not self.enable_metrics:
            return {"metrics_disabled": True}
        
        total_non_cached = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        cache_hit_rate = (
            self.metrics["cache_hits"] / total_non_cached
            if total_non_cached > 0 else 0.0
        )
        
        success_rate = (
            self.metrics["successful_calls"] / self.metrics["total_calls"]
            if self.metrics["total_calls"] > 0 else 0.0
        )
        
        return {
            **self.metrics,
            "cache_hit_rate": cache_hit_rate,
            "success_rate": success_rate,
            "cache_size": len(self.cache),
            "session_id": self.session_id,
        }

    def clear_cache(self) -> None:
        """Clear the tool result cache."""
        self.cache.clear()
        logger.info("Tool result cache cleared")

    def clear_metrics(self) -> None:
        """Reset performance metrics."""
        self.metrics = {
            "total_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "successful_calls": 0,
            "failed_calls": 0,
        }
        logger.info("Performance metrics cleared")


# Convenience function for quick usage
async def process_tools_for_session(
    session_id: str,
    llm_message: Dict[str, Any],
    **processor_kwargs
) -> List[ToolResult]:
    """
    Convenience function to process tools for a session.
    
    Args:
        session_id: ID of the session
        llm_message: OpenAI-style message with tool_calls
        **processor_kwargs: Additional arguments for the processor
        
    Returns:
        List of tool execution results
    """
    processor = await SessionAwareToolProcessor.create(session_id, **processor_kwargs)
    return await processor.process_llm_message(llm_message)