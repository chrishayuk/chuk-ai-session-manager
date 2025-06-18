# examples/05_monitoring.py
"""
📈 PRODUCTION MONITORING: Cost tracking and analytics

Monitor AI usage, costs, and performance in production.
"""
import asyncio
from datetime import datetime
from chuk_ai_session_manager import SessionManager

async def simulate_production_usage():
    """Simulate a day of production AI usage"""
    sessions = []
    
    # Simulate different types of AI interactions
    scenarios = [
        ("customer_support", "gpt-4o", 0.03),  # Higher cost model for support
        ("content_generation", "gpt-4o-mini", 0.0015),  # Cheaper for content
        ("code_review", "gpt-4", 0.06),  # Premium model for code
        ("data_analysis", "gpt-4o", 0.03),  # Analysis tasks
    ]
    
    for use_case, model, cost_per_1k in scenarios:
        for session_num in range(3):  # 3 sessions per use case
            sm = SessionManager()
            
            # Simulate conversation
            await sm.user_says(f"User request for {use_case} #{session_num + 1}")
            await sm.ai_responds(
                f"AI response for {use_case}. This is a longer response to simulate real usage patterns.",
                model=model,
                provider="openai",
                use_case=use_case,
                session_number=session_num + 1
            )
            
            sessions.append((use_case, await sm.get_stats()))
    
    return sessions

async def main():
    print("📈 Production monitoring simulation...")
    
    # Simulate usage
    sessions = await simulate_production_usage()
    
    # Analyze usage by use case
    usage_by_case = {}
    total_cost = 0
    total_tokens = 0
    
    for use_case, stats in sessions:
        if use_case not in usage_by_case:
            usage_by_case[use_case] = {
                "sessions": 0,
                "tokens": 0,
                "cost": 0,
                "messages": 0
            }
        
        usage_by_case[use_case]["sessions"] += 1
        usage_by_case[use_case]["tokens"] += stats["total_tokens"]
        usage_by_case[use_case]["cost"] += stats["estimated_cost"]
        usage_by_case[use_case]["messages"] += stats["total_events"]
        
        total_cost += stats["estimated_cost"]
        total_tokens += stats["total_tokens"]
    
    # Production monitoring report
    print(f"\n📊 Production AI Usage Report")
    print(f"=" * 50)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"💰 Total Cost: ${total_cost:.6f}")
    print(f"🔢 Total Tokens: {total_tokens:,}")
    print(f"📝 Total Sessions: {len(sessions)}")
    
    print(f"\n📈 Usage by Use Case:")
    for use_case, stats in usage_by_case.items():
        avg_cost = stats["cost"] / stats["sessions"]
        avg_tokens = stats["tokens"] / stats["sessions"]
        
        print(f"\n  {use_case.replace('_', ' ').title()}:")
        print(f"    Sessions: {stats['sessions']}")
        print(f"    Total Cost: ${stats['cost']:.6f}")
        print(f"    Avg Cost/Session: ${avg_cost:.6f}")
        print(f"    Avg Tokens/Session: {avg_tokens:.0f}")
    
    # Cost optimization insights
    print(f"\n💡 Optimization Insights:")
    highest_cost = max(usage_by_case.items(), key=lambda x: x[1]["cost"])
    highest_volume = max(usage_by_case.items(), key=lambda x: x[1]["sessions"])
    
    print(f"  🔥 Highest cost use case: {highest_cost[0]} (${highest_cost[1]['cost']:.6f})")
    print(f"  📊 Highest volume use case: {highest_volume[0]} ({highest_volume[1]['sessions']} sessions)")
    print(f"  💰 Consider cheaper models for high-volume, low-complexity tasks")

if __name__ == "__main__":
    asyncio.run(main())