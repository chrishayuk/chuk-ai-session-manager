import pytest
import asyncio
import time
from datetime import datetime, timezone

from chuk_ai_session_manager.models.token_usage import TokenUsage, TokenSummary


def test_default_token_usage_initialization():
    """Test default initialization of TokenUsage."""
    usage = TokenUsage()
    
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0
    assert usage.model == ""
    assert usage.estimated_cost_usd is None


def test_token_usage_with_values():
    """Test initialization with values."""
    usage = TokenUsage(
        prompt_tokens=10,
        completion_tokens=20,
        model="gpt-3.5-turbo"
    )
    
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 30  # Should be automatically calculated
    assert usage.model == "gpt-3.5-turbo"
    assert usage.estimated_cost_usd is not None  # Should be automatically calculated


def test_token_usage_auto_calculation():
    """Test automatic calculation of total tokens and cost."""
    usage = TokenUsage(
        prompt_tokens=100,
        completion_tokens=50,
        model="gpt-3.5-turbo"
    )
    
    assert usage.total_tokens == 150
    assert usage.estimated_cost_usd is not None
    assert usage.estimated_cost_usd > 0


@pytest.mark.asyncio
async def test_calculate_cost_async():
    """Test async cost calculation."""
    usage = TokenUsage(
        prompt_tokens=1000,
        completion_tokens=500,
        model="gpt-4"
    )
    
    # Calculate cost asynchronously
    cost = await usage.calculate_cost()
    
    assert cost is not None
    assert cost > 0
    # gpt-4 has higher cost than gpt-3.5-turbo
    assert cost > TokenUsage(
        prompt_tokens=1000,
        completion_tokens=500,
        model="gpt-3.5-turbo"
    ).estimated_cost_usd


@pytest.mark.asyncio
async def test_update_async():
    """Test async update of token counts."""
    usage = TokenUsage(
        prompt_tokens=10,
        completion_tokens=20,
        model="gpt-3.5-turbo"
    )
    
    original_cost = usage.estimated_cost_usd
    
    # Update asynchronously
    await usage.update(prompt_tokens=5, completion_tokens=10)
    
    assert usage.prompt_tokens == 15
    assert usage.completion_tokens == 30
    assert usage.total_tokens == 45
    assert usage.estimated_cost_usd > original_cost


@pytest.mark.asyncio
async def test_from_text_async():
    """Test async creation from text."""
    prompt_text = "This is a test prompt with about 10 tokens."
    completion_text = "This is a test completion with about 10 tokens as well."
    
    # Create asynchronously
    usage = await TokenUsage.from_text(
        prompt=prompt_text,
        completion=completion_text,
        model="gpt-3.5-turbo"
    )
    
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    assert usage.model == "gpt-3.5-turbo"
    assert usage.estimated_cost_usd is not None


@pytest.mark.asyncio
async def test_count_tokens_async():
    """Test async token counting."""
    text = "This is a test sentence to count tokens."
    
    # Count tokens asynchronously
    token_count = await TokenUsage.count_tokens(text, "gpt-3.5-turbo")
    
    assert token_count > 0
    
    # Empty text should have 0 tokens
    assert await TokenUsage.count_tokens("", "gpt-3.5-turbo") == 0
    
    # None should have 0 tokens
    assert await TokenUsage.count_tokens(None, "gpt-3.5-turbo") == 0


def test_token_usage_addition():
    """Test adding two TokenUsage instances."""
    usage1 = TokenUsage(
        prompt_tokens=10,
        completion_tokens=20,
        model="gpt-3.5-turbo"
    )
    
    usage2 = TokenUsage(
        prompt_tokens=30,
        completion_tokens=40,
        model="gpt-3.5-turbo"
    )
    
    combined = usage1 + usage2
    
    assert combined.prompt_tokens == 40
    assert combined.completion_tokens == 60
    assert combined.total_tokens == 100
    assert combined.model == "gpt-3.5-turbo"


def test_token_summary_initialization():
    """Test initialization of TokenSummary."""
    summary = TokenSummary()
    
    assert summary.total_prompt_tokens == 0
    assert summary.total_completion_tokens == 0
    assert summary.total_tokens == 0
    assert summary.usage_by_model == {}
    assert summary.total_estimated_cost_usd == 0.0


@pytest.mark.asyncio
async def test_token_summary_add_usage():
    """Test adding usage to a TokenSummary."""
    summary = TokenSummary()
    
    usage1 = TokenUsage(
        prompt_tokens=10,
        completion_tokens=20,
        model="gpt-3.5-turbo"
    )
    
    usage2 = TokenUsage(
        prompt_tokens=30,
        completion_tokens=40,
        model="gpt-4"
    )
    
    # Add usages asynchronously
    await summary.add_usage(usage1)
    await summary.add_usage(usage2)
    
    assert summary.total_prompt_tokens == 40
    assert summary.total_completion_tokens == 60
    assert summary.total_tokens == 100
    assert summary.total_estimated_cost_usd > 0
    
    # Both models should be tracked
    assert "gpt-3.5-turbo" in summary.usage_by_model
    assert "gpt-4" in summary.usage_by_model
    
    # Model-specific usage should be accurate
    assert summary.usage_by_model["gpt-3.5-turbo"].prompt_tokens == 10
    assert summary.usage_by_model["gpt-3.5-turbo"].completion_tokens == 20
    assert summary.usage_by_model["gpt-4"].prompt_tokens == 30
    assert summary.usage_by_model["gpt-4"].completion_tokens == 40


@pytest.mark.asyncio
async def test_token_summary_add_same_model_multiple_times():
    """Test adding usage for the same model multiple times."""
    summary = TokenSummary()
    
    usage1 = TokenUsage(
        prompt_tokens=10,
        completion_tokens=20,
        model="gpt-3.5-turbo"
    )
    
    usage2 = TokenUsage(
        prompt_tokens=30,
        completion_tokens=40,
        model="gpt-3.5-turbo"
    )
    
    # Add usages asynchronously
    await summary.add_usage(usage1)
    await summary.add_usage(usage2)
    
    assert summary.total_prompt_tokens == 40
    assert summary.total_completion_tokens == 60
    assert summary.total_tokens == 100
    
    # Model usage should be combined
    assert summary.usage_by_model["gpt-3.5-turbo"].prompt_tokens == 40
    assert summary.usage_by_model["gpt-3.5-turbo"].completion_tokens == 60


@pytest.mark.asyncio
async def test_concurrent_token_operations():
    """Test concurrent token operations."""
    # Create a large text sample
    text = "This is a test sentence. " * 100
    
    # Count tokens for the same text concurrently
    tasks = [TokenUsage.count_tokens(text, "gpt-3.5-turbo") for _ in range(10)]
    results = await asyncio.gather(*tasks)
    
    # All results should be the same
    assert len(set(results)) == 1  # Only one unique result
    assert results[0] > 0