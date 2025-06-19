# tests/test_system_prompt.py
"""
Test suite for system prompt functionality in SessionManager.
"""

import pytest
from unittest.mock import AsyncMock, patch

from chuk_ai_session_manager import SessionManager
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.event_source import EventSource


class TestSystemPrompt:
    """Test system prompt functionality."""
    
    async def test_session_manager_with_system_prompt(self):
        """Test creating SessionManager with system prompt."""
        system_prompt = "You are a helpful assistant."
        sm = SessionManager(system_prompt=system_prompt)
        
        assert sm.system_prompt == system_prompt
        
        # Initialize session
        await sm.user_says("Hello")
        
        # Check that system prompt was stored in metadata
        assert sm._session.metadata.properties.get("system_prompt") == system_prompt
    
    async def test_update_system_prompt(self):
        """Test updating system prompt."""
        sm = SessionManager(system_prompt="Initial prompt")
        
        # Initialize session
        await sm.user_says("Hello")
        
        # Update system prompt
        new_prompt = "You are now a technical expert."
        await sm.update_system_prompt(new_prompt)
        
        assert sm.system_prompt == new_prompt
        assert sm._session.metadata.properties.get("system_prompt") == new_prompt
    
    async def test_system_prompt_persistence(self):
        """Test that system prompt persists when loading session."""
        system_prompt = "You are a travel guide."
        
        # Create a mock store that actually persists sessions
        sessions_db = {}
        
        class MockStore:
            async def save(self, session):
                sessions_db[session.id] = session
            
            async def get(self, session_id):
                return sessions_db.get(session_id)
        
        mock_store = MockStore()
        
        # Create session with system prompt using the mock store
        sm1 = SessionManager(system_prompt=system_prompt, store=mock_store)
        await sm1.user_says("Tell me about Paris")
        session_id = sm1.session_id
        
        # Load session in new manager with the same mock store
        sm2 = SessionManager(session_id=session_id, store=mock_store)
        await sm2.user_says("What else?")  # This initializes the session
        
        # System prompt should be loaded from session
        assert sm2.system_prompt == system_prompt
    
    async def test_get_messages_for_llm_with_system(self):
        """Test getting messages formatted for LLM with system prompt."""
        system_prompt = "You are a helpful coding assistant."
        sm = SessionManager(system_prompt=system_prompt)
        
        await sm.user_says("How do I write a for loop in Python?")
        await sm.ai_responds("Here's how to write a for loop in Python...")
        
        # Get messages with system prompt
        messages = await sm.get_messages_for_llm(include_system=True)
        
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == system_prompt
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
    
    async def test_get_messages_for_llm_without_system(self):
        """Test getting messages without system prompt."""
        sm = SessionManager(system_prompt="You are a helpful assistant.")
        
        await sm.user_says("Hello")
        await sm.ai_responds("Hi there!")
        
        # Get messages without system prompt
        messages = await sm.get_messages_for_llm(include_system=False)
        
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
    
    async def test_no_system_prompt(self):
        """Test SessionManager without system prompt."""
        sm = SessionManager()
        
        assert sm.system_prompt is None
        
        await sm.user_says("Hello")
        
        # Get messages - should not include system prompt
        messages = await sm.get_messages_for_llm(include_system=True)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
    
    async def test_update_system_prompt_before_init(self):
        """Test updating system prompt before session initialization."""
        sm = SessionManager()
        
        # Update before any messages
        await sm.update_system_prompt("You are a medical assistant.")
        
        assert sm.system_prompt == "You are a medical assistant."
        
        # Initialize with first message
        await sm.user_says("I have a headache")
        
        # Should be stored in session
        assert sm._session.metadata.properties.get("system_prompt") == "You are a medical assistant."
    
    async def test_system_prompt_in_metadata_init(self):
        """Test passing system prompt through metadata."""
        metadata = {
            "system_prompt": "You are a chef.",
            "cuisine": "Italian"
        }
        
        sm = SessionManager(metadata=metadata)
        await sm.user_says("What's for dinner?")
        
        # Metadata should include both values
        assert sm._session.metadata.properties.get("system_prompt") == "You are a chef."
        assert sm._session.metadata.properties.get("cuisine") == "Italian"
    
    async def test_system_prompt_priority(self):
        """Test that explicit system_prompt parameter takes priority over metadata."""
        metadata = {"system_prompt": "From metadata"}
        explicit_prompt = "Explicit prompt"
        
        sm = SessionManager(system_prompt=explicit_prompt, metadata=metadata)
        await sm.user_says("Hello")
        
        assert sm.system_prompt == explicit_prompt
        assert sm._session.metadata.properties.get("system_prompt") == explicit_prompt
    
    async def test_complex_system_prompt(self):
        """Test with complex multi-line system prompt."""
        complex_prompt = """You are an AI assistant with the following capabilities:
        
        1. Code Analysis: Review and explain code in multiple languages
        2. Debugging: Help identify and fix bugs
        3. Best Practices: Suggest improvements and optimizations
        
        Guidelines:
        - Be concise but thorough
        - Use examples when helpful
        - Admit when you're unsure
        """
        
        sm = SessionManager(system_prompt=complex_prompt)
        await sm.user_says("Review this code: print('hello')")
        
        messages = await sm.get_messages_for_llm()
        assert messages[0]["content"] == complex_prompt
    
    async def test_session_manager_integration_with_prompt(self):
        """Integration test with full conversation flow."""
        sm = SessionManager(
            system_prompt="You are a pirate. Always speak like a pirate.",
            metadata={"theme": "nautical"}
        )
        
        # Simulate conversation
        await sm.user_says("Hello there!")
        await sm.ai_responds("Ahoy matey! Welcome aboard!", model="gpt-4o-mini")
        
        await sm.user_says("Tell me about the weather")
        await sm.ai_responds(
            "Arr, the skies be clear and the winds be favorable for sailing!",
            model="gpt-4o-mini"
        )
        
        # Check stats
        stats = await sm.get_stats()
        assert stats["user_messages"] == 2
        assert stats["ai_messages"] == 2
        
        # Verify system prompt in messages
        messages = await sm.get_messages_for_llm()
        assert messages[0]["role"] == "system"
        assert "pirate" in messages[0]["content"].lower()
    
    async def test_system_prompt_with_infinite_context(self):
        """Test system prompt persistence across session segments."""
        system_prompt = "You are a helpful assistant."
        sm = SessionManager(
            system_prompt=system_prompt,
            infinite_context=True,
            token_threshold=100,  # Low threshold to trigger segmentation
            max_turns_per_segment=2
        )
        
        # First segment
        await sm.user_says("Hello")
        await sm.ai_responds("Hi there!")
        
        # This should trigger a new segment
        await sm.user_says("How are you?")
        await sm.ai_responds("I'm doing well, thanks!")
        
        # Check that we have multiple segments
        chain = await sm.get_session_chain()
        assert len(chain) > 1
        
        # Verify system prompt is preserved
        assert sm.system_prompt == system_prompt
        
        # Get messages should still include system prompt
        messages = await sm.get_messages_for_llm()
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == system_prompt
    
    async def test_empty_system_prompt(self):
        """Test handling of empty string system prompt."""
        sm = SessionManager(system_prompt="")
        
        await sm.user_says("Hello")
        messages = await sm.get_messages_for_llm(include_system=True)
        
        # Empty prompt should not be included
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
    
    async def test_system_prompt_thread_safety(self):
        """Test that system prompt updates are thread-safe."""
        import asyncio
        
        sm = SessionManager(system_prompt="Initial")
        await sm.user_says("Start")
        
        # Concurrent updates
        async def update_prompt(prompt: str):
            await sm.update_system_prompt(prompt)
        
        # Run multiple updates concurrently
        tasks = [
            update_prompt("Prompt 1"),
            update_prompt("Prompt 2"),
            update_prompt("Prompt 3"),
        ]
        await asyncio.gather(*tasks)
        
        # Should have one of the prompts (last one to complete)
        assert sm.system_prompt in ["Prompt 1", "Prompt 2", "Prompt 3"]
        assert sm._session.metadata.properties.get("system_prompt") == sm.system_prompt


class TestSystemPromptExamples:
    """Test real-world system prompt examples."""
    
    async def test_customer_support_prompt(self):
        """Test customer support system prompt."""
        system_prompt = """You are a professional customer support agent for TechCorp.
        - Always be polite and empathetic
        - Acknowledge the customer's issue first
        - Provide step-by-step solutions
        - Offer to escalate if needed"""
        
        sm = SessionManager(system_prompt=system_prompt)
        await sm.user_says("My laptop won't turn on!")
        
        messages = await sm.get_messages_for_llm()
        assert messages[0]["content"] == system_prompt
    
    async def test_code_assistant_prompt(self):
        """Test code assistant system prompt."""
        system_prompt = """You are an expert Python developer.
        - Write clean, documented code
        - Follow PEP 8 style guidelines
        - Explain your code clearly
        - Suggest best practices"""
        
        sm = SessionManager(system_prompt=system_prompt)
        await sm.user_says("How do I read a CSV file?")
        await sm.ai_responds("You can use pandas or the csv module...")
        
        # Verify prompt is included
        messages = await sm.get_messages_for_llm()
        assert "Python developer" in messages[0]["content"]
    
    async def test_multi_language_prompt(self):
        """Test multi-language support prompt."""
        system_prompt = "Detect the user's language and respond in the same language."
        
        sm = SessionManager(system_prompt=system_prompt)
        
        # Test with different languages
        await sm.user_says("Bonjour!")
        await sm.ai_responds("Bonjour! Comment puis-je vous aider?")
        
        await sm.user_says("¡Hola!")
        await sm.ai_responds("¡Hola! ¿En qué puedo ayudarte?")
        
        messages = await sm.get_messages_for_llm()
        assert messages[0]["content"] == system_prompt
        assert len(messages) == 5  # system + 2 user + 2 assistant


if __name__ == "__main__":
    pytest.main([__file__, "-v"])