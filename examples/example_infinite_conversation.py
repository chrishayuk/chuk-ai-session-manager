#!/usr/bin/env python3
# examples/example_infinite_conversation.py
"""
Example demonstrating the InfiniteConversationManager with async support.

This example shows how to:
1. Set up and use the InfiniteConversationManager with the async API
2. Handle conversations that span multiple session segments
3. Build context from hierarchical sessions for LLM calls
4. Simulate a long conversation that crosses token thresholds
5. Try different summarization strategies
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Callable, Union
from datetime import datetime, timezone

# Import session manager components - FIXED for current architecture
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.session_storage import get_backend, ChukSessionsStore, setup_chuk_sessions_storage

# Import the InfiniteConversationManager
from chuk_ai_session_manager.infinite_conversation import (
    InfiniteConversationManager,
    SummarizationStrategy
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Quiet down ALL the noisy loggers
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("anthropic").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("h11").setLevel(logging.ERROR)
logging.getLogger("chuk_llm").setLevel(logging.WARNING)
logging.getLogger("chuk_sessions").setLevel(logging.WARNING)
logging.getLogger("chuk_ai_session_manager").setLevel(logging.WARNING)

# Sample conversation data for simulation
CONVERSATION = [
    {"role": "user", "content": "I'd like to understand quantum computing. Can you explain it to me?"},
    {"role": "assistant", "content": "Quantum computing uses quantum bits or qubits, which can exist in multiple states simultaneously due to superposition. This is unlike classical computing bits that are either 0 or 1. Quantum computers leverage this property along with quantum entanglement to perform certain calculations much faster than classical computers."},
    {"role": "user", "content": "That's interesting. How do qubits maintain their superposition?"},
    {"role": "assistant", "content": "Qubits maintain superposition through careful isolation from the environment, which is challenging because quantum states are fragile and easily disturbed by external interactions - a phenomenon called decoherence. Scientists use various techniques like superconducting circuits cooled to near absolute zero, trapped ions, or topological qubits to maintain these delicate quantum states. The longer a qubit can maintain its state (coherence time), the more complex calculations can be performed."},
    {"role": "user", "content": "What are some practical applications of quantum computing?"},
    {"role": "assistant", "content": "Quantum computing has several promising practical applications. In cryptography, quantum computers could break many current encryption methods but also enable quantum-secure communications. For chemistry and materials science, they can simulate molecular interactions at a quantum level, potentially revolutionizing drug discovery and material design. In optimization problems, quantum algorithms could improve logistics, financial modeling, and machine learning. Though most applications are still theoretical or in early stages, companies like IBM, Google, and D-Wave are making progress in practical quantum computing implementations."},
    {"role": "user", "content": "Could quantum computing help with climate modeling?"},
    {"role": "assistant", "content": "Yes, quantum computing could significantly impact climate modeling. Current climate models require enormous computational power to simulate complex atmospheric and oceanic systems. Quantum computers could potentially handle these complex calculations more efficiently, leading to more accurate and detailed climate predictions. They might excel at modeling quantum-level effects in atmospheric chemistry, optimizing energy systems to reduce emissions, and solving complex differential equations that govern climate dynamics. However, this application is still theoretical, as today's quantum computers aren't yet powerful enough for such large-scale simulations."},
    {"role": "user", "content": "What are the main challenges in building a practical quantum computer?"},
    {"role": "assistant", "content": "Building practical quantum computers faces several major challenges. First is quantum decoherence - quantum states are extremely fragile and lose their quantum properties when interacting with the environment, requiring sophisticated isolation techniques. Error correction is another major hurdle, as quantum operations are prone to errors that must be detected and corrected without disrupting the quantum state. Scaling up quantum systems while maintaining qubit quality and coherence time remains difficult. Additionally, there are engineering challenges in precisely controlling qubits, developing appropriate algorithms for quantum advantages, and creating interfaces between quantum and classical systems. Despite these obstacles, significant progress continues in the field."},
    {"role": "user", "content": "How does quantum entanglement work?"},
    {"role": "assistant", "content": "Quantum entanglement occurs when two or more qubits become correlated in such a way that the quantum state of each particle cannot be described independently of the others. When particles are entangled, measuring the state of one instantly influences the state of the other, regardless of the distance separating them - what Einstein famously called \"spooky action at a distance.\" This doesn't involve information traveling faster than light, but rather reveals pre-existing correlations. Entanglement is a cornerstone of quantum computing, enabling operations across multiple qubits simultaneously and serving as a resource for quantum algorithms, quantum teleportation, and quantum cryptography."},
    {"role": "user", "content": "What's the difference between quantum computers and supercomputers?"},
    {"role": "assistant", "content": "Quantum computers and supercomputers differ fundamentally in their approach to computation. Supercomputers are essentially very powerful classical computers that use traditional bits (0s and 1s) and achieve their performance through massive parallelization of conventional processors. They excel at number crunching and simulations within the classical computing paradigm. Quantum computers, however, use qubits that can exist in superpositions of states and become entangled, allowing them to explore multiple solutions simultaneously through quantum effects. This gives quantum computers theoretical advantages for specific problems like factoring large numbers, database searching, and quantum system simulations. However, quantum computers aren't universally faster - they're designed for particular classes of problems where they can provide exponential speedups, while supercomputers remain superior for many conventional computing tasks."}
]


# Simulate LLM function for generating responses and summaries
async def simulated_llm_call(messages: List[Dict[str, str]], model: str = "gpt-4") -> str:
    """Simulate an LLM call by returning a predefined response or summary."""
    system_msg = next((m for m in messages if m.get("role") == "system"), None)
    
    # If this is a summarization request
    if system_msg and "summary" in system_msg.get("content", "").lower():
        # Create different summaries based on the content of the system message
        if "key points" in system_msg.get("content", "").lower():
            return "Key points from the conversation: 1) Quantum computing uses qubits that leverage superposition and entanglement; 2) Practical applications include cryptography, chemistry, and optimization; 3) Major challenges include decoherence and error correction; 4) Quantum computers differ from supercomputers in fundamental approach."
        elif "topics" in system_msg.get("content", "").lower():
            return "Topics discussed: QUANTUM BASICS: Qubits, superposition, and entanglement; PRACTICAL APPLICATIONS: Cryptography, material science, climate modeling; TECHNICAL CHALLENGES: Decoherence, error correction, scaling; COMPARISONS: Quantum vs. classical/supercomputers."
        elif "user's main questions" in system_msg.get("content", "").lower():
            return "The user inquired about quantum computing fundamentals, how qubits maintain superposition, practical applications including climate modeling, technical challenges, quantum entanglement, and differences from supercomputers."
        else:
            # Basic summary
            return "This conversation covers quantum computing concepts including qubits, superposition, entanglement, practical applications in cryptography and climate modeling, technical challenges like decoherence, and comparisons with classical supercomputers."
    
    # For regular conversations, just return a placeholder
    return "This is a simulated LLM response that would continue the conversation naturally."


async def demonstrate_infinite_conversation(
    strategy: SummarizationStrategy = SummarizationStrategy.BASIC
):
    """Demonstrate the InfiniteConversationManager with a simulated conversation."""
    print(f"\n=== Demonstrating InfiniteConversationManager with {strategy} strategy ===\n")
    
    # Set up CHUK Sessions storage backend
    setup_chuk_sessions_storage(sandbox_id="infinite-conversation-demo", default_ttl_hours=2)
    backend = get_backend()
    store = ChukSessionsStore(backend)
    
    # Create the conversation manager with a low threshold for demonstration
    manager = InfiniteConversationManager(
        token_threshold=1000,  # Low threshold for demo purposes
        summarization_strategy=strategy
    )
    
    # Create the initial session
    initial_session = await Session.create()
    await store.save(initial_session)
    current_session_id = initial_session.id
    
    print(f"Created initial session: {current_session_id}")
    
    # Simulate the conversation
    print("\nStarting conversation simulation...\n")
    for turn in CONVERSATION:
        role = turn["role"]
        content = turn["content"]
        
        print(f"\n{role.upper()}: {content[:50]}...")
        
        # Process the message
        source = EventSource.USER if role == "user" else EventSource.LLM
        new_session_id = await manager.process_message(
            current_session_id,
            content,
            source,
            simulated_llm_call
        )
        
        # Check if we've moved to a new session
        if new_session_id != current_session_id:
            print(f"\n[Session transition: {current_session_id} -> {new_session_id}]")
            current_session_id = new_session_id
    
    print("\nConversation simulation completed.")
    
    # Now build context for an LLM call
    print("\nBuilding context for LLM from the final session...")
    context = await manager.build_context_for_llm(current_session_id)
    
    # Print the context structure
    print(f"\nContext includes {len(context)} messages:")
    for i, msg in enumerate(context):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        print(f"  {i+1}. [{role}]: {content[:50]}...")
    
    # Examine the session hierarchy
    print("\nExamining session hierarchy...")
    session = await store.get(current_session_id)
    ancestors = await session.ancestors()
    print(f"Final session {current_session_id} has {len(ancestors)} ancestors:")
    for i, ancestor in enumerate(ancestors):
        print(f"  Ancestor {i+1}: {ancestor.id}")
        summary_event = next((e for e in reversed(ancestor.events) if e.type == EventType.SUMMARY), None)
        if summary_event:
            print(f"    Summary: {str(summary_event.message)[:100]}...")
    
    # Get full conversation history
    print("\nRetrieving full conversation history...")
    history = await manager.get_full_conversation_history(current_session_id)
    print(f"Full history contains {len(history)} exchanges")
    
    return current_session_id, manager


async def compare_summarization_strategies():
    """Compare different summarization strategies."""
    print("\n=== Comparing Summarization Strategies ===\n")
    
    strategies = [
        SummarizationStrategy.BASIC,
        SummarizationStrategy.KEY_POINTS,
        SummarizationStrategy.QUERY_FOCUSED,
        SummarizationStrategy.TOPIC_BASED
    ]
    
    summaries = []
    
    # Run the conversation with each strategy
    for strategy in strategies:
        print(f"\nTesting {strategy} summarization strategy...")
        
        # Set up fresh storage for each test
        setup_chuk_sessions_storage(sandbox_id=f"infinite-demo-{strategy}", default_ttl_hours=1)
        backend = get_backend()
        store = ChukSessionsStore(backend)
        
        # Create manager with this strategy
        manager = InfiniteConversationManager(
            token_threshold=1000,
            summarization_strategy=strategy
        )
        
        # Create session
        session = await Session.create()
        await store.save(session)
        session_id = session.id
        
        # Add messages until threshold is reached
        for i, turn in enumerate(CONVERSATION):
            role = turn["role"]
            content = turn["content"]
            source = EventSource.USER if role == "user" else EventSource.LLM
            
            new_session_id = await manager.process_message(
                session_id,
                content,
                source,
                simulated_llm_call
            )
            
            # If we've triggered a summarization, capture it and stop
            if new_session_id != session_id:
                # Get the summary from the original session
                orig_session = await store.get(session_id)
                summary_event = next((e for e in reversed(orig_session.events) 
                                     if e.type == EventType.SUMMARY), None)
                
                if summary_event:
                    summary_text = summary_event.message
                    summaries.append((strategy, summary_text))
                    print(f"Generated summary: {summary_text[:100]}...")
                break
            
            # If we've gone through all messages without triggering summarization,
            # force a summary
            if i == len(CONVERSATION) - 1:
                print("Conversation completed without reaching token threshold.")
                print("Forcing summarization...")
                
                # Get the session from store
                current_session = await store.get(session_id)
                
                # Force create a summary
                summary = await manager._create_summary(current_session, simulated_llm_call)
                summaries.append((strategy, summary))
                print(f"Generated summary: {summary[:100]}...")
    
    # Compare the summaries
    print("\nComparison of Summarization Strategies:")
    print("====================================")
    
    for strategy, summary in summaries:
        print(f"\n{strategy} Strategy:")
        print(f"  {summary}")
    
    print("\nSummarization comparison complete.")


async def demonstrate_full_conversation_history():
    """Demonstrate retrieving the full conversation history across segments."""
    print("\n=== Demonstrating Full Conversation History Retrieval ===\n")
    
    # Set up storage
    setup_chuk_sessions_storage(sandbox_id="infinite-history-demo", default_ttl_hours=1)
    backend = get_backend()
    store = ChukSessionsStore(backend)
    
    # Create manager
    manager = InfiniteConversationManager(token_threshold=1000)
    
    # Run the conversation to create multiple segments
    session = await Session.create()
    await store.save(session)
    session_id = session.id
    
    print(f"Created initial session: {session_id}")
    
    # Process all messages
    for turn in CONVERSATION:
        role = turn["role"]
        content = turn["content"]
        source = EventSource.USER if role == "user" else EventSource.LLM
        
        session_id = await manager.process_message(
            session_id,
            content,
            source,
            simulated_llm_call
        )
    
    # Now retrieve the full history
    history = await manager.get_full_conversation_history(session_id)
    
    # Get session chain for count
    session_chain = await manager.get_session_chain(session_id)
    
    print(f"\nRetrieved full conversation history across {len(session_chain)} session segments")
    print(f"History contains {len(history)} total exchanges\n")
    
    # Print a sample of the history
    print("Sample of conversation history:")
    for i, (role, source, content) in enumerate(history[:4]):  # Show first few messages
        print(f"  {i+1}. [{role}]: {content[:50]}...")
    
    print("  ...")
    
    # Show last few messages
    for i, (role, source, content) in enumerate(history[-3:]):  # Show last few messages
        print(f"  {len(history)-2+i}. [{role}]: {content[:50]}...")


async def demonstrate_session_segmentation():
    """Demonstrate how sessions are segmented when token thresholds are exceeded."""
    print("\n=== Demonstrating Session Segmentation ===\n")
    
    # Set up storage
    setup_chuk_sessions_storage(sandbox_id="segmentation-demo", default_ttl_hours=1)
    backend = get_backend()
    store = ChukSessionsStore(backend)
    
    # Create manager with very low threshold to force segmentation
    manager = InfiniteConversationManager(
        token_threshold=500,  # Very low threshold
        max_turns_per_segment=3,  # Also limit turns per segment
        summarization_strategy=SummarizationStrategy.BASIC
    )
    
    # Create initial session
    session = await Session.create()
    await store.save(session)
    session_id = session.id
    session_history = [session_id]
    
    print(f"Starting with session: {session_id}")
    print("Token threshold: 500, Max turns per segment: 3")
    
    # Process conversation and track session changes
    for i, turn in enumerate(CONVERSATION):
        role = turn["role"]
        content = turn["content"]
        source = EventSource.USER if role == "user" else EventSource.LLM
        
        print(f"\nProcessing turn {i+1}: {role.upper()}: {content[:30]}...")
        
        new_session_id = await manager.process_message(
            session_id,
            content,
            source,
            simulated_llm_call
        )
        
        if new_session_id != session_id:
            print(f"  🔄 Session transition: {session_id} -> {new_session_id}")
            session_history.append(new_session_id)
            session_id = new_session_id
        else:
            print(f"  ✅ Staying in session: {session_id}")
    
    print(f"\nSegmentation complete!")
    print(f"Total sessions created: {len(session_history)}")
    print(f"Session chain: {' -> '.join([s[:8] + '...' for s in session_history])}")
    
    # Analyze each session
    print("\nSession Analysis:")
    for i, sess_id in enumerate(session_history):
        sess = await store.get(sess_id)
        if sess:
            message_events = [e for e in sess.events if e.type == EventType.MESSAGE]
            summary_events = [e for e in sess.events if e.type == EventType.SUMMARY]
            
            print(f"  Session {i+1} ({sess_id[:8]}...):")
            print(f"    Messages: {len(message_events)}")
            print(f"    Summaries: {len(summary_events)}")
            print(f"    Total tokens: {sess.total_tokens}")
            
            if summary_events:
                print(f"    Summary: {str(summary_events[-1].message)[:50]}...")


async def main():
    """Run all demonstrations."""
    
    print("🚀 Infinite Conversation Manager Demonstration")
    print("=" * 60)
    
    # Demonstrate basic usage
    await demonstrate_infinite_conversation(SummarizationStrategy.BASIC)
    
    # Compare summarization strategies
    await compare_summarization_strategies()
    
    # Demonstrate full history retrieval
    await demonstrate_full_conversation_history()
    
    # Demonstrate session segmentation
    await demonstrate_session_segmentation()
    
    print("\n✅ All demonstrations completed successfully!")
    print("=" * 60)
    print("🎯 Key Features Demonstrated:")
    print("  • Automatic session segmentation based on token thresholds")
    print("  • Multiple summarization strategies for different use cases")
    print("  • Hierarchical session management with parent-child relationships")
    print("  • Full conversation history retrieval across session segments")
    print("  • Context building for LLM calls with summary inclusion")
    print("  • Production-ready infinite conversation handling")


if __name__ == "__main__":
    asyncio.run(main())