"""
Model configuration for different agent types.
"""

from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum

from .groq_client import GroqModel


class AgentModelProfile(Enum):
    """Model profiles for different agent types."""
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    GENERAL = "general"


@dataclass
class ModelConfiguration:
    """Model configuration for an agent."""
    primary_model: GroqModel
    fallback_model: GroqModel
    temperature: float
    max_tokens: int
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    use_cache: bool = True
    enable_fallback: bool = True


# Agent-specific model configurations
AGENT_MODEL_CONFIGS: Dict[AgentModelProfile, ModelConfiguration] = {
    AgentModelProfile.RESEARCHER: ModelConfiguration(
        primary_model=GroqModel.LLAMA_3_1_70B_VERSATILE,
        fallback_model=GroqModel.LLAMA_3_1_8B_INSTANT,
        temperature=0.3,  # Lower temperature for factual research
        max_tokens=4000,
        top_p=0.9,
        use_cache=True,
        enable_fallback=True
    ),
    
    AgentModelProfile.ANALYST: ModelConfiguration(
        primary_model=GroqModel.LLAMA_3_1_70B_VERSATILE,
        fallback_model=GroqModel.MIXTRAL_8X7B_32768,
        temperature=0.4,  # Balanced for analysis
        max_tokens=4000,
        top_p=0.95,
        use_cache=True,
        enable_fallback=True
    ),
    
    AgentModelProfile.WRITER: ModelConfiguration(
        primary_model=GroqModel.LLAMA_3_1_70B_VERSATILE,
        fallback_model=GroqModel.GEMMA_2_9B_IT,
        temperature=0.6,  # Higher temperature for creative writing
        max_tokens=6000,
        top_p=0.9,
        frequency_penalty=0.1,  # Reduce repetition
        presence_penalty=0.1,   # Encourage diverse topics
        use_cache=False,  # Don't cache creative content
        enable_fallback=True
    ),
    
    AgentModelProfile.GENERAL: ModelConfiguration(
        primary_model=GroqModel.LLAMA_3_1_70B_VERSATILE,
        fallback_model=GroqModel.LLAMA_3_1_8B_INSTANT,
        temperature=0.7,
        max_tokens=2000,
        use_cache=True,
        enable_fallback=True
    )
}


# Task-specific model recommendations
TASK_MODEL_RECOMMENDATIONS: Dict[str, GroqModel] = {
    # Research tasks
    "web_research": GroqModel.LLAMA_3_1_70B_VERSATILE,
    "document_analysis": GroqModel.LLAMA_3_1_70B_VERSATILE,
    "fact_verification": GroqModel.LLAMA_3_1_70B_VERSATILE,
    
    # Analysis tasks
    "trend_analysis": GroqModel.LLAMA_3_1_405B_REASONING,
    "comparative_analysis": GroqModel.LLAMA_3_1_70B_VERSATILE,
    "swot_analysis": GroqModel.MIXTRAL_8X7B_32768,
    "risk_assessment": GroqModel.LLAMA_3_1_70B_VERSATILE,
    "synthesis": GroqModel.LLAMA_3_1_405B_REASONING,
    
    # Writing tasks
    "report_writing": GroqModel.LLAMA_3_1_70B_VERSATILE,
    "article_writing": GroqModel.LLAMA_3_1_70B_VERSATILE,
    "summary_writing": GroqModel.LLAMA_3_1_8B_INSTANT,
    "content_editing": GroqModel.GEMMA_2_9B_IT,
    "content_structuring": GroqModel.MIXTRAL_8X7B_32768,
    
    # Quick tasks
    "quick_response": GroqModel.LLAMA_3_1_8B_INSTANT,
    "simple_analysis": GroqModel.LLAMA_3_1_8B_INSTANT,
    "basic_writing": GroqModel.GEMMA_2_9B_IT,
}


def get_model_config_for_agent(agent_type: str) -> ModelConfiguration:
    """
    Get model configuration for agent type.
    
    Args:
        agent_type: Type of agent (researcher, analyst, writer)
        
    Returns:
        Model configuration
    """
    profile_map = {
        "researcher": AgentModelProfile.RESEARCHER,
        "analyst": AgentModelProfile.ANALYST,
        "writer": AgentModelProfile.WRITER,
    }
    
    profile = profile_map.get(agent_type.lower(), AgentModelProfile.GENERAL)
    return AGENT_MODEL_CONFIGS[profile]


def get_model_for_task(task_type: str, complexity: str = "medium") -> GroqModel:
    """
    Get recommended model for specific task.
    
    Args:
        task_type: Type of task
        complexity: Task complexity (low, medium, high)
        
    Returns:
        Recommended model
    """
    # Check task-specific recommendations first
    if task_type in TASK_MODEL_RECOMMENDATIONS:
        return TASK_MODEL_RECOMMENDATIONS[task_type]
    
    # Fall back to complexity-based recommendations
    complexity_models = {
        "low": GroqModel.LLAMA_3_1_8B_INSTANT,
        "medium": GroqModel.LLAMA_3_1_70B_VERSATILE,
        "high": GroqModel.LLAMA_3_1_405B_REASONING
    }
    
    return complexity_models.get(complexity.lower(), GroqModel.LLAMA_3_1_70B_VERSATILE)


def get_temperature_for_task(task_type: str, agent_type: str = "general") -> float:
    """
    Get recommended temperature for task and agent type.
    
    Args:
        task_type: Type of task
        agent_type: Type of agent
        
    Returns:
        Recommended temperature
    """
    # Task-specific temperatures
    task_temperatures = {
        # Research tasks (factual, lower temperature)
        "web_research": 0.2,
        "document_analysis": 0.3,
        "fact_verification": 0.1,
        
        # Analysis tasks (balanced temperature)
        "trend_analysis": 0.4,
        "comparative_analysis": 0.3,
        "swot_analysis": 0.4,
        "risk_assessment": 0.3,
        "synthesis": 0.5,
        
        # Writing tasks (creative, higher temperature)
        "report_writing": 0.5,
        "article_writing": 0.6,
        "summary_writing": 0.4,
        "content_editing": 0.3,
        "content_structuring": 0.4,
    }
    
    # Agent-specific base temperatures
    agent_base_temps = {
        "researcher": 0.3,
        "analyst": 0.4,
        "writer": 0.6,
        "general": 0.5
    }
    
    # Return task-specific if available, otherwise agent-specific
    return task_temperatures.get(
        task_type,
        agent_base_temps.get(agent_type.lower(), 0.5)
    )


def optimize_config_for_context(
    base_config: ModelConfiguration,
    context_length: int,
    urgency: str = "normal"
) -> ModelConfiguration:
    """
    Optimize model configuration based on context and urgency.
    
    Args:
        base_config: Base configuration
        context_length: Length of context in tokens
        urgency: Task urgency (low, normal, high)
        
    Returns:
        Optimized configuration
    """
    # Create a copy to avoid modifying the original
    optimized = ModelConfiguration(
        primary_model=base_config.primary_model,
        fallback_model=base_config.fallback_model,
        temperature=base_config.temperature,
        max_tokens=base_config.max_tokens,
        top_p=base_config.top_p,
        frequency_penalty=base_config.frequency_penalty,
        presence_penalty=base_config.presence_penalty,
        use_cache=base_config.use_cache,
        enable_fallback=base_config.enable_fallback
    )
    
    # Adjust for urgency
    if urgency == "high":
        # Use faster models for urgent tasks
        if optimized.primary_model == GroqModel.LLAMA_3_1_405B_REASONING:
            optimized.primary_model = GroqModel.LLAMA_3_1_70B_VERSATILE
        elif optimized.primary_model == GroqModel.LLAMA_3_1_70B_VERSATILE:
            optimized.primary_model = GroqModel.LLAMA_3_1_8B_INSTANT
        
        # Reduce max_tokens for faster response
        optimized.max_tokens = min(optimized.max_tokens, 2000)
        
        # Enable aggressive caching
        optimized.use_cache = True
    
    elif urgency == "low":
        # Use more capable models for non-urgent tasks
        if optimized.primary_model == GroqModel.LLAMA_3_1_8B_INSTANT:
            optimized.primary_model = GroqModel.LLAMA_3_1_70B_VERSATILE
        elif optimized.primary_model == GroqModel.LLAMA_3_1_70B_VERSATILE:
            optimized.primary_model = GroqModel.LLAMA_3_1_405B_REASONING
    
    # Adjust for context length
    if context_length > 50000:  # Large context
        # Use models with larger context windows
        if optimized.primary_model == GroqModel.GEMMA_2_9B_IT:
            optimized.primary_model = GroqModel.LLAMA_3_1_70B_VERSATILE
    
    elif context_length < 5000:  # Small context
        # Can use smaller, faster models
        if optimized.primary_model == GroqModel.LLAMA_3_1_405B_REASONING:
            optimized.primary_model = GroqModel.LLAMA_3_1_70B_VERSATILE
    
    return optimized