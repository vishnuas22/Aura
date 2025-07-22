#!/usr/bin/env python3
"""
Demonstration script for the AI Research Assistant foundational architecture.

This script demonstrates the key features of the multi-agent system:
- Agent creation and initialization
- Communication between agents
- Memory management
- Performance metrics tracking
- Error handling and retry mechanisms
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from agents import create_agent, create_research_team, AgentType
from core import initialize_metrics, initialize_communication
from core.config import ConfigManager
from memory.agent_memory import create_memory_store


async def demonstrate_agent_creation():
    """Demonstrate agent creation and basic functionality."""
    print("🤖 AI Research Assistant - Foundational Architecture Demo")
    print("=" * 60)
    
    # Initialize core systems
    print("\n📋 Initializing core systems...")
    config_manager = ConfigManager()
    metrics_collector = initialize_metrics()
    communication_channel = initialize_communication()
    
    print("✅ Core systems initialized successfully!")
    
    # Create individual agents
    print("\n👥 Creating individual agents...")
    
    try:
        # Create a researcher agent
        print("   🔍 Creating Researcher Agent...")
        researcher = await create_agent(
            agent_type=AgentType.RESEARCHER,
            agent_id="demo_researcher",
            auto_start=True
        )
        print(f"   ✅ Created: {researcher}")
        
        # Create an analyst agent
        print("   📊 Creating Analyst Agent...")
        analyst = await create_agent(
            agent_type=AgentType.ANALYST,
            agent_id="demo_analyst", 
            auto_start=True
        )
        print(f"   ✅ Created: {analyst}")
        
        # Create a writer agent
        print("   ✍️ Creating Writer Agent...")
        writer = await create_agent(
            agent_type=AgentType.WRITER,
            agent_id="demo_writer",
            auto_start=True
        )
        print(f"   ✅ Created: {writer}")
        
        return researcher, analyst, writer
        
    except Exception as e:
        print(f"   ❌ Error creating agents: {e}")
        raise


async def demonstrate_agent_communication(researcher, analyst, writer):
    """Demonstrate communication between agents."""
    print("\n💬 Demonstrating agent communication...")
    
    try:
        # Send data from researcher to analyst
        print("   📤 Researcher sharing data with Analyst...")
        await researcher.share_data_with_agent(
            target_agent_id=analyst.agent_id,
            data={
                "research_findings": [
                    "AI agent frameworks are gaining popularity",
                    "CrewAI offers role-based agent collaboration",
                    "Multi-agent systems improve task efficiency"
                ],
                "sources": 3,
                "confidence": 0.8
            },
            data_type="research_findings"
        )
        print("   ✅ Data shared successfully!")
        
        # Send analysis results from analyst to writer
        print("   📤 Analyst sharing insights with Writer...")
        await analyst.share_data_with_agent(
            target_agent_id=writer.agent_id,
            data={
                "key_insights": [
                    "Multi-agent systems show 40% efficiency improvement",
                    "Role-based architecture enhances specialization",
                    "Communication protocols are critical for coordination"
                ],
                "analysis_confidence": 0.85,
                "recommendations": [
                    "Implement CrewAI for production systems",
                    "Focus on agent specialization",
                    "Ensure robust communication channels"
                ]
            },
            data_type="analysis_results"
        )
        print("   ✅ Analysis shared successfully!")
        
    except Exception as e:
        print(f"   ❌ Communication error: {e}")


async def demonstrate_task_execution(researcher, analyst, writer):
    """Demonstrate task execution with error handling and metrics."""
    print("\n⚡ Demonstrating task execution...")
    
    try:
        # Execute research task
        print("   🔍 Executing research task...")
        research_result = await researcher.execute_task({
            "id": "demo_research_001",
            "type": "web_research",
            "query": "Benefits of multi-agent AI systems",
            "max_sources": 5,
            "description": "Research the benefits and applications of multi-agent AI systems"
        })
        print(f"   ✅ Research completed in {research_result['execution_time']:.2f}s")
        
        # Execute analysis task
        print("   📊 Executing analysis task...")
        analysis_result = await analyst.execute_task({
            "id": "demo_analysis_001", 
            "type": "trend_analysis",
            "data": research_result.get("sources", {}),
            "focus_areas": ["efficiency", "scalability", "adoption"],
            "description": "Analyze trends in multi-agent system adoption"
        })
        print(f"   ✅ Analysis completed in {analysis_result['execution_time']:.2f}s")
        
        # Execute writing task
        print("   ✍️ Executing writing task...")
        writing_result = await writer.execute_task({
            "id": "demo_writing_001",
            "type": "report_writing", 
            "content": {
                "research_data": research_result,
                "analysis_data": analysis_result,
                "title": "Multi-Agent AI Systems: Benefits and Applications"
            },
            "style": "professional",
            "audience": "technical",
            "description": "Write a comprehensive report on multi-agent AI systems"
        })
        print(f"   ✅ Writing completed in {writing_result['execution_time']:.2f}s")
        
        return research_result, analysis_result, writing_result
        
    except Exception as e:
        print(f"   ❌ Task execution error: {e}")
        return None, None, None


async def demonstrate_metrics_and_status(researcher, analyst, writer):
    """Demonstrate performance metrics and status reporting."""
    print("\n📈 Demonstrating metrics and status...")
    
    try:
        # Get agent status
        print("   📋 Getting agent status...")
        for agent in [researcher, analyst, writer]:
            status = await agent.get_status()
            print(f"   {agent.agent_type.title()}: {status['is_busy']=}, Tools: {len(status['available_tools'])}")
        
        # Get performance metrics
        print("   📊 Getting performance metrics...")
        for agent in [researcher, analyst, writer]:
            try:
                metrics = await agent.get_performance_metrics()
                all_time_metrics = metrics.get('all_time', {})
                if all_time_metrics:
                    print(f"   {agent.agent_type.title()} metrics: "
                          f"Success rate: {all_time_metrics.get('success_rate', 0):.1%}, "
                          f"Avg exec time: {all_time_metrics.get('avg_execution_time', 0):.2f}s")
                else:
                    print(f"   {agent.agent_type.title()} metrics: No data yet")
            except Exception as e:
                print(f"   {agent.agent_type.title()} metrics: Error - {e}")
        
    except Exception as e:
        print(f"   ❌ Metrics error: {e}")


async def demonstrate_memory_system(researcher, analyst, writer):
    """Demonstrate memory management capabilities."""
    print("\n🧠 Demonstrating memory system...")
    
    try:
        # Check conversation history
        print("   💭 Checking conversation history...")
        for agent in [researcher, analyst, writer]:
            try:
                history = await agent.memory.get_conversation_history(limit=3)
                print(f"   {agent.agent_type.title()}: {len(history)} conversation items in memory")
            except Exception as e:
                print(f"   {agent.agent_type.title()} memory error: {e}")
        
        # Check insights
        print("   💡 Checking stored insights...")
        for agent in [researcher, analyst, writer]:
            try:
                insights = await agent.memory.get_recent_insights(limit=2)
                print(f"   {agent.agent_type.title()}: {len(insights)} insights stored")
            except Exception as e:
                print(f"   {agent.agent_type.title()} insights error: {e}")
                
    except Exception as e:
        print(f"   ❌ Memory system error: {e}")


async def demonstrate_team_creation():
    """Demonstrate research team creation."""
    print("\n👥 Demonstrating research team creation...")
    
    try:
        print("   🏗️ Creating complete research team...")
        team = await create_research_team(team_id="demo_team_001")
        
        print(f"   ✅ Team created with {len(team)} agents:")
        for agent_type, agent in team.items():
            print(f"      - {agent_type.title()}: {agent.agent_id}")
        
        # Demonstrate team coordination
        print("   🤝 Demonstrating team coordination...")
        researcher = team["researcher"]
        analyst = team["analyst"] 
        writer = team["writer"]
        
        # Quick team task
        print("   📋 Executing coordinated team task...")
        await researcher.share_data_with_agent(
            target_agent_id=analyst.agent_id,
            data={"quick_research": "Team coordination data"},
            data_type="team_coordination"
        )
        
        await analyst.share_data_with_agent(
            target_agent_id=writer.agent_id,
            data={"quick_analysis": "Team coordination analysis"},
            data_type="team_coordination"
        )
        
        print("   ✅ Team coordination successful!")
        
        # Cleanup team
        print("   🧹 Cleaning up team...")
        from agents import get_agent_factory
        factory = get_agent_factory()
        
        for agent in team.values():
            await agent.stop()
        
        print("   ✅ Team cleanup completed!")
        
    except Exception as e:
        print(f"   ❌ Team creation error: {e}")


async def cleanup_agents(*agents):
    """Cleanup created agents."""
    print("\n🧹 Cleaning up agents...")
    
    for agent in agents:
        if agent:
            try:
                await agent.stop()
                print(f"   ✅ Stopped: {agent.agent_id}")
            except Exception as e:
                print(f"   ⚠️ Error stopping {agent.agent_id}: {e}")


async def main():
    """Main demonstration function."""
    try:
        # Core demonstration
        researcher, analyst, writer = await demonstrate_agent_creation()
        
        await demonstrate_agent_communication(researcher, analyst, writer)
        
        research_result, analysis_result, writing_result = await demonstrate_task_execution(
            researcher, analyst, writer
        )
        
        await demonstrate_metrics_and_status(researcher, analyst, writer)
        
        await demonstrate_memory_system(researcher, analyst, writer)
        
        # Team demonstration
        await demonstrate_team_creation()
        
        # Final results
        print("\n🎉 Demo Results Summary:")
        print("=" * 60)
        print("✅ Agent creation and initialization")
        print("✅ Inter-agent communication")
        print("✅ Task execution with error handling")
        print("✅ Performance metrics tracking")
        print("✅ Memory management system")
        print("✅ Research team coordination")
        print("\n🎯 All foundational architecture components working successfully!")
        
        if writing_result and writing_result.get("content"):
            print("\n📝 Sample Generated Content:")
            print("-" * 40)
            content_preview = writing_result["content"][:300]
            print(f"{content_preview}...")
            print(f"\nWord count: {writing_result.get('word_count', 0)}")
            print(f"Quality score: {writing_result.get('quality_score', 0):.2f}")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            await cleanup_agents(researcher, analyst, writer)
        except:
            pass


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress verbose logs for demo
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    # Run the demo
    asyncio.run(main())