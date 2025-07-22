"""
LLM integration package for AI agents.
"""

from .groq_client import (
    GroqLLMClient,
    GroqModel,
    ModelResponse,
    TokenUsage,
    ModelSpec,
    ResponseCache,
    TokenCounter,
    HealthChecker,
    get_llm_client,
    initialize_llm_client,
    MODEL_SPECS
)
from .model_config import (
    AgentModelProfile,
    ModelConfiguration,
    get_model_config_for_agent,
    get_model_for_task,
    get_temperature_for_task,
    optimize_config_for_context,
    AGENT_MODEL_CONFIGS,
    TASK_MODEL_RECOMMENDATIONS
)
from .llm_integration import LLMIntegration

__all__ = [
    # Client and core classes
    "GroqLLMClient",
    "GroqModel",
    "ModelResponse",
    "TokenUsage",
    "ModelSpec",
    "ResponseCache",
    "TokenCounter",
    "HealthChecker",
    
    # Client management
    "get_llm_client",
    "initialize_llm_client",
    
    # Model configurations
    "AgentModelProfile",
    "ModelConfiguration",
    "get_model_config_for_agent",
    "get_model_for_task", 
    "get_temperature_for_task",
    "optimize_config_for_context",
    
    # Integration layer
    "LLMIntegration",
    
    # Constants
    "MODEL_SPECS",
    "AGENT_MODEL_CONFIGS",
    "TASK_MODEL_RECOMMENDATIONS"
]

# Version information
__version__ = "1.0.0"
__author__ = "AI Research Team"
__description__ = "LLM integration with Groq for multi-agent system"