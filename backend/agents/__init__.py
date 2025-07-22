"""
AI Agents package for the Research Assistant.

This package contains all agent implementations and utilities for
the multi-agent AI research assistant system.
"""

from .base_agent import BaseAIAgent
from .researcher_agent import ResearcherAgent
from .analyst_agent import AnalystAgent
from .writer_agent import WriterAgent
from .agent_factory import (
    AgentFactory, 
    AgentType, 
    get_agent_factory,
    initialize_agent_factory,
    create_agent,
    create_research_team,
    get_active_agents
)

__all__ = [
    # Base classes
    "BaseAIAgent",
    
    # Specific agents
    "ResearcherAgent",
    "AnalystAgent", 
    "WriterAgent",
    
    # Factory and management
    "AgentFactory",
    "AgentType",
    "get_agent_factory",
    "initialize_agent_factory",
    "create_agent",
    "create_research_team",
    "get_active_agents"
]

# Version information
__version__ = "1.0.0"
__author__ = "AI Research Team"
__description__ = "Multi-agent AI research assistant system"