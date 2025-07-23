"""
Configuration management for the AI Research Assistant.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from pydantic import BaseModel, Field
from .exceptions import ConfigurationException


@dataclass
class ModelConfig:
    """Configuration for LLM models."""
    name: str
    provider: str
    temperature: float = 0.3
    max_tokens: int = 4000
    timeout: int = 300


class LLMConfig(BaseModel):
    """Configuration for LLM integration."""
    enable_caching: bool = Field(default=True)
    cache_ttl: int = Field(default=3600, gt=0)  # seconds
    max_retries: int = Field(default=3, ge=0)
    request_timeout: int = Field(default=300, gt=0)
    max_context_length: int = Field(default=10, gt=0)
    enable_fallback: bool = Field(default=True)


class AgentConfig(BaseModel):
    """Configuration for individual agents."""
    role: str = Field(..., description="Agent role")
    backstory: str = Field(..., description="Agent backstory")
    goal: str = Field(..., description="Agent goal")
    model_settings: ModelConfig
    llm_config: LLMConfig = Field(default_factory=LLMConfig)
    tools: list[str] = Field(default_factory=list)
    max_retries: int = Field(default=3, ge=0)
    timeout: int = Field(default=300, gt=0)
    verbose: bool = Field(default=True)
    memory_enabled: bool = Field(default=True)
    performance_tracking: bool = Field(default=True)


class MemoryConfig(BaseModel):
    """Configuration for memory management."""
    enabled: bool = Field(default=True)
    window_size: int = Field(default=10, gt=0)
    long_term_enabled: bool = Field(default=True)
    cache_ttl: int = Field(default=3600, gt=0)  # seconds
    max_entries: int = Field(default=1000, gt=0)


class SystemConfig(BaseModel):
    """System-wide configuration."""
    groq_api_key: str = Field(..., description="Groq API key")
    default_model: ModelConfig
    fast_model: ModelConfig
    reasoning_model: ModelConfig
    llm_config: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig
    log_level: str = Field(default="INFO")
    enable_metrics: bool = Field(default=True)
    max_concurrent_agents: int = Field(default=5, gt=0)


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or Path("/app/config.yaml")
        self._config: Optional[Dict[str, Any]] = None
        self._system_config: Optional[SystemConfig] = None
        
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
            
        Raises:
            ConfigurationException: If configuration cannot be loaded
        """
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f)
            else:
                self._config = self._get_default_config()
            
            return self._config
            
        except Exception as e:
            raise ConfigurationException(
                f"Failed to load configuration: {str(e)}",
                error_code="CONFIG_LOAD_ERROR"
            )
    
    def get_system_config(self) -> SystemConfig:
        """
        Get validated system configuration.
        
        Returns:
            SystemConfig instance
            
        Raises:
            ConfigurationException: If configuration is invalid
        """
        if self._system_config is None:
            config = self.load_config()
            self._system_config = self._build_system_config(config)
        
        return self._system_config
    
    def get_agent_config(self, agent_type: str) -> AgentConfig:
        """
        Get configuration for specific agent type.
        
        Args:
            agent_type: Type of agent (researcher, analyst, writer)
            
        Returns:
            AgentConfig instance
            
        Raises:
            ConfigurationException: If agent configuration not found
        """
        config = self.load_config()
        
        if agent_type not in config:
            raise ConfigurationException(
                f"Configuration for agent type '{agent_type}' not found",
                error_code="AGENT_CONFIG_NOT_FOUND"
            )
        
        agent_data = config[agent_type]
        global_config = config.get('global', {})
        
        # Build model configuration
        model_settings = ModelConfig(
            name=agent_data.get('llm', {}).get('model', global_config.get('default_model')),
            provider="groq",
            temperature=agent_data.get('llm', {}).get('temperature', global_config.get('analytical_temperature', 0.3)),
            max_tokens=agent_data.get('llm', {}).get('max_tokens', global_config.get('max_tokens', 4000)),
            timeout=agent_data.get('limits', {}).get('timeout_seconds', global_config.get('default_timeout', 300))
        )
        
        # Build LLM configuration
        llm_config = LLMConfig(
            enable_caching=agent_data.get('llm', {}).get('enable_caching', True),
            cache_ttl=agent_data.get('llm', {}).get('cache_ttl', 3600),
            max_retries=agent_data.get('limits', {}).get('max_iterations', 3),
            request_timeout=agent_data.get('limits', {}).get('timeout_seconds', 300),
            max_context_length=global_config.get('memory_window', 10),
            enable_fallback=agent_data.get('llm', {}).get('enable_fallback', True)
        )
        
        return AgentConfig(
            role=agent_data['role'],
            backstory=agent_data['backstory'],
            goal=agent_data['goal'],
            model_settings=model_settings,
            llm_config=llm_config,
            tools=agent_data.get('tools', []),
            max_retries=agent_data.get('limits', {}).get('max_iterations', 3),
            timeout=agent_data.get('limits', {}).get('timeout_seconds', 300),
            verbose=agent_data.get('behavior', {}).get('verbose', True),
            memory_enabled=global_config.get('long_term_memory', True),
            performance_tracking=True
        )
    
    def _build_system_config(self, config: Dict[str, Any]) -> SystemConfig:
        """Build system configuration from loaded config."""
        global_config = config.get('global', {})
        
        # Build model configurations
        default_model = ModelConfig(
            name=global_config.get('default_model', 'llama-3.1-70b-versatile'),
            provider="groq",
            temperature=global_config.get('analytical_temperature', 0.3),
            max_tokens=global_config.get('max_tokens', 4000),
            timeout=global_config.get('default_timeout', 300)
        )
        
        fast_model = ModelConfig(
            name=global_config.get('fast_model', 'llama-3.1-8b-instant'),
            provider="groq",
            temperature=global_config.get('analytical_temperature', 0.3),
            max_tokens=global_config.get('max_tokens', 4000),
            timeout=global_config.get('quick_timeout', 60)
        )
        
        reasoning_model = ModelConfig(
            name=global_config.get('reasoning_model', 'llama-3.1-70b-versatile'),
            provider="groq",
            temperature=global_config.get('analytical_temperature', 0.3),
            max_tokens=global_config.get('max_tokens', 4000),
            timeout=global_config.get('deep_timeout', 600)
        )
        
        # Build LLM configuration
        llm_config = LLMConfig(
            enable_caching=global_config.get('enable_caching', True),
            cache_ttl=global_config.get('cache_ttl', 3600),
            max_retries=global_config.get('max_retries', 3),
            request_timeout=global_config.get('default_timeout', 300),
            max_context_length=global_config.get('memory_window', 10),
            enable_fallback=global_config.get('enable_fallback', True)
        )
        
        memory_config = MemoryConfig(
            enabled=global_config.get('long_term_memory', True),
            window_size=global_config.get('memory_window', 10),
            long_term_enabled=global_config.get('long_term_memory', True)
        )
        
        return SystemConfig(
            groq_api_key=os.getenv('GROQ_API_KEY', ''),
            default_model=default_model,
            fast_model=fast_model,
            reasoning_model=reasoning_model,
            llm_config=llm_config,
            memory=memory_config,
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            enable_metrics=True,
            max_concurrent_agents=5
        )
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file doesn't exist."""
        return {
            'global': {
                'default_model': 'llama-3.1-70b-versatile',
                'fast_model': 'llama-3.1-8b-instant',
                'reasoning_model': 'llama-3.1-70b-versatile',
                'analytical_temperature': 0.3,
                'creative_temperature': 0.7,
                'factual_temperature': 0.1,
                'max_tokens': 4000,
                'default_timeout': 300,
                'quick_timeout': 60,
                'deep_timeout': 600,
                'memory_window': 10,
                'long_term_memory': True,
                'enable_caching': True,
                'cache_ttl': 3600,
                'max_retries': 3,
                'enable_fallback': True
            },
            'researcher': {
                'role': 'Senior Research Specialist',
                'backstory': 'Experienced researcher with expertise in data gathering and analysis.',
                'goal': 'Conduct thorough research and gather comprehensive information.',
                'llm': {
                    'model': 'llama-3.1-70b-versatile',
                    'temperature': 0.3,
                    'max_tokens': 4000,
                    'enable_caching': True,
                    'enable_fallback': True
                },
                'tools': ['web_search', 'document_analysis', 'fact_verification'],
                'limits': {
                    'timeout_seconds': 300,
                    'max_iterations': 5
                },
                'behavior': {
                    'verbose': True
                }
            },
            'analyst': {
                'role': 'Strategic Data Analyst',
                'backstory': 'Expert in analyzing complex data and identifying patterns.',
                'goal': 'Analyze research data and provide strategic insights.',
                'llm': {
                    'model': 'llama-3.1-70b-versatile',
                    'temperature': 0.4,
                    'max_tokens': 4000,
                    'enable_caching': True,
                    'enable_fallback': True
                },
                'tools': ['data_analysis', 'pattern_recognition', 'trend_analysis'],
                'limits': {
                    'timeout_seconds': 180,
                    'max_iterations': 4
                },
                'behavior': {
                    'verbose': True
                }
            },
            'writer': {
                'role': 'Professional Content Writer',
                'backstory': 'Skilled writer with expertise in creating engaging content.',
                'goal': 'Create well-structured and professional content.',
                'llm': {
                    'model': 'llama-3.1-70b-versatile',
                    'temperature': 0.6,
                    'max_tokens': 4000,
                    'enable_caching': False,  # Don't cache creative content
                    'enable_fallback': True
                },
                'tools': ['content_structuring', 'grammar_checking', 'style_optimization'],
                'limits': {
                    'timeout_seconds': 240,
                    'max_iterations': 3
                },
                'behavior': {
                    'verbose': True
                }
            }
        }