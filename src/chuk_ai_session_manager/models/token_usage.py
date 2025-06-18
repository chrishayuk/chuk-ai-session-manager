# chuk_ai_session_manager/models/token_usage.py
"""
Token usage tracking models for the chuk session manager.

This module provides models for tracking token usage in LLM interactions
with proper async support.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, Optional, Union, List, Any
from uuid import uuid4
from pydantic import BaseModel, Field, ConfigDict
import asyncio

# Try to import tiktoken, but make it optional
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class TokenUsage(BaseModel):
    """
    Tracks token usage for LLM interactions.
    
    Attributes:
        prompt_tokens: Number of tokens in the prompt/input
        completion_tokens: Number of tokens in the completion/output
        total_tokens: Total tokens (prompt + completion)
        model: The model used for the interaction (helps with pricing calculations)
        estimated_cost_usd: Estimated cost in USD (if pricing info is available)
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = Field(default=0)
    model: str = ""
    estimated_cost_usd: Optional[float] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        # Auto-calculate total tokens if not explicitly provided
        if self.total_tokens == 0 and (self.prompt_tokens > 0 or self.completion_tokens > 0):
            self.total_tokens = self.prompt_tokens + self.completion_tokens
            
        # Auto-calculate estimated cost if model is provided
        if self.model and self.estimated_cost_usd is None:
            self.estimated_cost_usd = self._calculate_cost_sync()
    
    def _calculate_cost_sync(self) -> float:
        """
        Synchronous implementation of calculate_cost.
        
        Returns:
            Estimated cost in USD
        """
        # Model pricing per 1000 tokens (approximate as of May 2025)
        pricing = {
            # OpenAI models
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            
            # Claude models
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            
            # Fallback for unknown models
            "default": {"input": 0.001, "output": 0.002}
        }
        
        # Get pricing for this model or use default
        model_pricing = pricing.get(self.model.lower(), pricing["default"])
        
        # Calculate cost
        input_cost = (self.prompt_tokens / 1000) * model_pricing["input"]
        output_cost = (self.completion_tokens / 1000) * model_pricing["output"]
        
        return round(input_cost + output_cost, 6)
    
    async def calculate_cost(self) -> float:
        """
        Async version of calculate_cost.
        
        Returns:
            Estimated cost in USD
        """
        # Token calculation is CPU-bound, so run in executor
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._calculate_cost_sync)
    
    def _update_sync(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """
        Synchronous implementation of update.
        
        Args:
            prompt_tokens: Additional prompt tokens to add
            completion_tokens: Additional completion tokens to add
        """
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        
        if self.model:
            self.estimated_cost_usd = self._calculate_cost_sync()
    
    async def update(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """
        Async version of update.
        
        Args:
            prompt_tokens: Additional prompt tokens to add
            completion_tokens: Additional completion tokens to add
        """
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        
        if self.model:
            self.estimated_cost_usd = await self.calculate_cost()
    
    @classmethod
    def _from_text_sync(
        cls, 
        prompt: str, 
        completion: Optional[str] = None, 
        model: str = "gpt-3.5-turbo"
    ) -> TokenUsage:
        """
        Synchronous implementation of from_text.
        
        Args:
            prompt: The prompt/input text
            completion: The completion/output text (optional)
            model: The model name to use for counting and pricing
            
        Returns:
            A TokenUsage instance with token counts
        """
        prompt_tokens = cls._count_tokens_sync(prompt, model)
        completion_tokens = cls._count_tokens_sync(completion, model) if completion else 0
        
        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model
        )
    
    @classmethod
    async def from_text(
        cls, 
        prompt: str, 
        completion: Optional[str] = None, 
        model: str = "gpt-3.5-turbo"
    ) -> TokenUsage:
        """
        Async version of from_text.
        
        Args:
            prompt: The prompt/input text
            completion: The completion/output text (optional)
            model: The model name to use for counting and pricing
            
        Returns:
            A TokenUsage instance with token counts
        """
        # Run token counting in executor since it's CPU-bound
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, 
            lambda: cls._from_text_sync(prompt, completion, model)
        )
    
    @staticmethod
    def _count_tokens_sync(text: Optional[Union[str, Any]], model: str = "gpt-3.5-turbo") -> int:
        """
        Synchronous implementation of count_tokens.
        
        Args:
            text: The text to count tokens for
            model: The model name to use for counting
            
        Returns:
            The number of tokens
        """
        if text is None:
            return 0
        
        # Convert to string if not already a string
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception:
                return 0
        
        # Empty string has 0 tokens
        if not text:
            return 0
            
        if TIKTOKEN_AVAILABLE:
            try:
                encoding = tiktoken.encoding_for_model(model)
                return len(encoding.encode(text))
            except (KeyError, ValueError):
                # Fall back to cl100k_base encoding if the model is not found
                try:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    return len(encoding.encode(text))
                except Exception:
                    # If all else fails, use the approximation
                    pass
        
        # Simple approximation: ~4 chars per token for English text
        return int(len(text) / 4)
    
    @staticmethod
    async def count_tokens(text: Optional[Union[str, Any]], model: str = "gpt-3.5-turbo") -> int:
        """
        Async version of count_tokens.
        
        Args:
            text: The text to count tokens for
            model: The model name to use for counting
            
        Returns:
            The number of tokens
        """
        # Run in executor since token counting is CPU-bound
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: TokenUsage._count_tokens_sync(text, model)
        )
    
    def __add__(self, other: TokenUsage) -> TokenUsage:
        """
        Add two TokenUsage instances together.
        
        Args:
            other: Another TokenUsage instance
            
        Returns:
            A new TokenUsage instance with combined counts
        """
        # Use the model from self if it exists, otherwise use the other's model
        model = self.model if self.model else other.model
        
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            model=model
        )


class TokenSummary(BaseModel):
    """
    Summarizes token usage across multiple interactions.
    
    Attributes:
        total_prompt_tokens: Total tokens used in prompts
        total_completion_tokens: Total tokens used in completions
        total_tokens: Total tokens overall
        usage_by_model: Breakdown of usage by model
        total_estimated_cost_usd: Total estimated cost across all models
    """
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    usage_by_model: Dict[str, TokenUsage] = Field(default_factory=dict)
    total_estimated_cost_usd: float = 0.0
    
    def _add_usage_sync(self, usage: TokenUsage) -> None:
        """
        Synchronous implementation of add_usage.
        
        Args:
            usage: The TokenUsage to add
        """
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        
        if usage.estimated_cost_usd is not None:
            self.total_estimated_cost_usd += usage.estimated_cost_usd
        
        if usage.model:
            if usage.model in self.usage_by_model:
                self.usage_by_model[usage.model]._update_sync(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens
                )
            else:
                self.usage_by_model[usage.model] = TokenUsage(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    model=usage.model
                )
    
    async def add_usage(self, usage: TokenUsage) -> None:
        """
        Async version of add_usage.
        
        Args:
            usage: The TokenUsage to add
        """
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        
        if usage.estimated_cost_usd is not None:
            self.total_estimated_cost_usd += usage.estimated_cost_usd
        
        if usage.model:
            if usage.model in self.usage_by_model:
                await self.usage_by_model[usage.model].update(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens
                )
            else:
                self.usage_by_model[usage.model] = TokenUsage(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    model=usage.model
                )