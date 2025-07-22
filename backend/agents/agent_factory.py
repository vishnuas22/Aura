"""
Factory pattern for creating and managing AI agents.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Type, Union
from enum import Enum

from .base_agent import BaseAIAgent
from .researcher_agent import ResearcherAgent
from .analyst_agent import AnalystAgent
from .writer_agent import WriterAgent
from ..core.config import ConfigManager, AgentConfig
from ..core.exceptions import ConfigurationException, AgentException
from ..memory.agent_memory import create_memory_store
from ..tools.agent_tools import get_tools_for_agent


class AgentType(Enum):
    """Available agent types."""
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"


class AgentFactory:
    """
    Factory class for creating and managing AI agents.
    
    Provides:
    - Agent creation with proper configuration
    - Agent lifecycle management
    - Tool assignment
    - Memory store configuration
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize agent factory.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.logger = logging.getLogger(__name__)
        
        # Registry of agent classes
        self._agent_classes: Dict[AgentType, Type[BaseAIAgent]] = {
            AgentType.RESEARCHER: ResearcherAgent,
            AgentType.ANALYST: AnalystAgent,
            AgentType.WRITER: WriterAgent
        }
        
        # Active agents registry
        self._active_agents: Dict[str, BaseAIAgent] = {}
        
        self.logger.info("Agent factory initialized")
    
    async def create_agent(
        self,
        agent_type: Union[AgentType, str],
        agent_id: Optional[str] = None,
        config_override: Optional[Dict[str, Any]] = None,
        memory_store_type: str = "memory",
        auto_start: bool = True
    ) -> BaseAIAgent:
        """
        Create a new agent instance.
        
        Args:
            agent_type: Type of agent to create
            agent_id: Optional custom agent ID
            config_override: Override configuration values
            memory_store_type: Type of memory store ("memory" or "mongo")
            auto_start: Whether to start the agent automatically
            
        Returns:
            Created agent instance
            
        Raises:
            AgentException: If agent creation fails
        """
        try:
            # Convert string to enum if needed
            if isinstance(agent_type, str):
                agent_type = AgentType(agent_type.lower())
            
            # Get agent class
            agent_class = self._agent_classes.get(agent_type)
            if not agent_class:
                raise AgentException(
                    f"Unknown agent type: {agent_type}",
                    error_code="UNKNOWN_AGENT_TYPE"
                )
            
            # Load configuration
            config = self._load_agent_config(agent_type.value, config_override)
            
            # Get tools for this agent type
            tools = await get_tools_for_agent(agent_type.value)
            
            # Create agent instance
            self.logger.info(f"Creating {agent_type.value} agent...")
            
            agent = agent_class(
                agent_type=agent_type.value,
                config=config,
                agent_id=agent_id,
                memory_store_type=memory_store_type,
                tools=tools
            )
            
            # Register agent
            self._active_agents[agent.agent_id] = agent
            
            # Start agent if requested
            if auto_start:
                await agent.start()
            
            self.logger.info(f"Agent created successfully: {agent.agent_id}")
            return agent
            
        except Exception as e:
            self.logger.error(f"Failed to create {agent_type} agent: {e}")
            raise AgentException(f"Agent creation failed: {str(e)}")
    
    async def create_research_team(
        self, 
        team_id: Optional[str] = None,
        config_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, BaseAIAgent]:
        """
        Create a complete research team with all agent types.
        
        Args:
            team_id: Optional team identifier
            config_overrides: Configuration overrides for each agent type
            
        Returns:
            Dictionary of created agents by type
        """
        team_id = team_id or f"team_{int(asyncio.get_event_loop().time())}"
        config_overrides = config_overrides or {}
        
        self.logger.info(f"Creating research team: {team_id}")
        
        team = {}
        
        try:
            # Create each agent type
            for agent_type in AgentType:
                agent_config_override = config_overrides.get(agent_type.value, {})
                
                # Add team ID to metadata
                agent_config_override.setdefault('metadata', {})['team_id'] = team_id
                
                agent = await self.create_agent(
                    agent_type=agent_type,
                    agent_id=f"{team_id}_{agent_type.value}",
                    config_override=agent_config_override,
                    auto_start=True
                )
                
                team[agent_type.value] = agent
            
            self.logger.info(f"Research team created successfully: {team_id}")
            return team
            
        except Exception as e:
            # Cleanup any created agents
            for agent in team.values():
                try:
                    await agent.stop()
                    if agent.agent_id in self._active_agents:
                        del self._active_agents[agent.agent_id]
                except Exception as cleanup_error:
                    self.logger.error(f"Error cleaning up agent {agent.agent_id}: {cleanup_error}")
            
            raise AgentException(f"Team creation failed: {str(e)}")
    
    async def get_agent(self, agent_id: str) -> Optional[BaseAIAgent]:
        """
        Get an active agent by ID.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent instance if found, None otherwise
        """
        return self._active_agents.get(agent_id)
    
    async def list_active_agents(self) -> List[Dict[str, Any]]:
        """
        List all active agents.
        
        Returns:
            List of agent status dictionaries
        """
        agents_status = []
        
        for agent in self._active_agents.values():
            try:
                status = await agent.get_status()
                agents_status.append(status)
            except Exception as e:
                self.logger.error(f"Error getting status for agent {agent.agent_id}: {e}")
                agents_status.append({
                    "agent_id": agent.agent_id,
                    "agent_type": agent.agent_type,
                    "status": "error",
                    "error": str(e)
                })
        
        return agents_status
    
    async def stop_agent(self, agent_id: str) -> bool:
        """
        Stop and remove an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            True if agent was stopped, False if not found
        """
        agent = self._active_agents.get(agent_id)
        if not agent:
            return False
        
        try:
            await agent.stop()
            del self._active_agents[agent_id]
            self.logger.info(f"Agent stopped: {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping agent {agent_id}: {e}")
            raise AgentException(f"Failed to stop agent: {str(e)}")
    
    async def stop_all_agents(self) -> int:
        """
        Stop all active agents.
        
        Returns:
            Number of agents stopped
        """
        stopped_count = 0
        agent_ids = list(self._active_agents.keys())
        
        for agent_id in agent_ids:
            try:
                if await self.stop_agent(agent_id):
                    stopped_count += 1
            except Exception as e:
                self.logger.error(f"Error stopping agent {agent_id}: {e}")
        
        self.logger.info(f"Stopped {stopped_count} agents")
        return stopped_count
    
    async def restart_agent(self, agent_id: str) -> Optional[BaseAIAgent]:
        """
        Restart an agent (stop and recreate with same configuration).
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            New agent instance if successful, None if agent not found
        """
        agent = self._active_agents.get(agent_id)
        if not agent:
            return None
        
        # Get current configuration
        agent_type = agent.agent_type
        config = agent.config
        
        try:
            # Stop current agent
            await self.stop_agent(agent_id)
            
            # Create new agent with same configuration
            new_agent = await self.create_agent(
                agent_type=agent_type,
                agent_id=agent_id,
                config_override=config.__dict__,
                auto_start=True
            )
            
            self.logger.info(f"Agent restarted: {agent_id}")
            return new_agent
            
        except Exception as e:
            self.logger.error(f"Error restarting agent {agent_id}: {e}")
            raise AgentException(f"Failed to restart agent: {str(e)}")
    
    async def get_agent_metrics(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get performance metrics for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Metrics dictionary if agent found, None otherwise
        """
        agent = self._active_agents.get(agent_id)
        if not agent:
            return None
        
        try:
            return await agent.get_performance_metrics()
        except Exception as e:
            self.logger.error(f"Error getting metrics for agent {agent_id}: {e}")
            return None
    
    def _load_agent_config(
        self, 
        agent_type: str, 
        config_override: Optional[Dict[str, Any]] = None
    ) -> AgentConfig:
        """
        Load agent configuration with optional overrides.
        
        Args:
            agent_type: Type of agent
            config_override: Configuration overrides
            
        Returns:
            Agent configuration
        """
        try:
            # Load base configuration
            config = self.config_manager.get_agent_config(agent_type)
            
            # Apply overrides if provided
            if config_override:
                for key, value in config_override.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                    else:
                        self.logger.warning(f"Unknown configuration key: {key}")
            
            return config
            
        except Exception as e:
            raise ConfigurationException(f"Failed to load agent configuration: {str(e)}")
    
    def register_agent_class(self, agent_type: AgentType, agent_class: Type[BaseAIAgent]) -> None:
        """
        Register a custom agent class.
        
        Args:
            agent_type: Agent type
            agent_class: Agent class
        """
        self._agent_classes[agent_type] = agent_class
        self.logger.info(f"Registered custom agent class for {agent_type.value}")
    
    def get_supported_agent_types(self) -> List[str]:
        """
        Get list of supported agent types.
        
        Returns:
            List of agent type names
        """
        return [agent_type.value for agent_type in self._agent_classes.keys()]
    
    def __len__(self) -> int:
        """Get number of active agents."""
        return len(self._active_agents)
    
    def __contains__(self, agent_id: str) -> bool:
        """Check if agent is active."""
        return agent_id in self._active_agents


# Global factory instance
_global_factory: Optional[AgentFactory] = None


def get_agent_factory() -> AgentFactory:
    """Get the global agent factory instance."""
    global _global_factory
    if _global_factory is None:
        _global_factory = AgentFactory()
    return _global_factory


def initialize_agent_factory(config_manager: Optional[ConfigManager] = None) -> AgentFactory:
    """Initialize the global agent factory."""
    global _global_factory
    _global_factory = AgentFactory(config_manager)
    return _global_factory


# Convenience functions
async def create_agent(
    agent_type: Union[AgentType, str],
    **kwargs
) -> BaseAIAgent:
    """Convenience function to create an agent."""
    factory = get_agent_factory()
    return await factory.create_agent(agent_type, **kwargs)


async def create_research_team(**kwargs) -> Dict[str, BaseAIAgent]:
    """Convenience function to create a research team."""
    factory = get_agent_factory()
    return await factory.create_research_team(**kwargs)


async def get_active_agents() -> List[Dict[str, Any]]:
    """Convenience function to list active agents."""
    factory = get_agent_factory()
    return await factory.list_active_agents()