"""
LLM integration layer for AI agents.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, AsyncIterator
from datetime import datetime

from .groq_client import GroqLLMClient, GroqModel, ModelResponse
from .model_config import (
    get_model_config_for_agent, 
    get_model_for_task, 
    get_temperature_for_task,
    optimize_config_for_context,
    ModelConfiguration
)
from core.exceptions import ModelException
from core.metrics import get_metrics_collector, MetricType


class LLMIntegration:
    """
    High-level LLM integration for AI agents.
    
    Provides:
    - Agent-aware model selection
    - Automatic configuration optimization  
    - Performance tracking
    - Error handling with graceful degradation
    - Context-aware processing
    """
    
    def __init__(self, client: GroqLLMClient):
        """
        Initialize LLM integration.
        
        Args:
            client: Groq LLM client instance
        """
        self.client = client
        self.logger = logging.getLogger(__name__)
        self.metrics = get_metrics_collector()
        
        # Cache for model configurations
        self._config_cache: Dict[str, ModelConfiguration] = {}
        
        self.logger.info("LLM integration initialized")
    
    async def generate_response(
        self,
        agent_id: str,
        agent_type: str,
        messages: List[Dict[str, str]],
        task_type: Optional[str] = None,
        complexity: str = "medium",
        urgency: str = "normal",
        stream: bool = False,
        **kwargs
    ) -> Union[ModelResponse, AsyncIterator[str]]:
        """
        Generate response optimized for agent and task.
        
        Args:
            agent_id: ID of the requesting agent
            agent_type: Type of agent (researcher, analyst, writer)
            messages: Conversation messages
            task_type: Specific task type
            complexity: Task complexity (low, medium, high)
            urgency: Task urgency (low, normal, high)
            stream: Whether to stream response
            **kwargs: Additional parameters
            
        Returns:
            Model response or async iterator for streaming
        """
        start_time = datetime.utcnow()
        
        try:
            # Get optimized configuration
            config = self._get_optimized_config(
                agent_type=agent_type,
                task_type=task_type,
                messages=messages,
                complexity=complexity,
                urgency=urgency
            )
            
            # Log configuration selection
            self.logger.debug(
                f"Agent {agent_id} using model {config.primary_model.value} "
                f"for {task_type or 'general'} task (complexity: {complexity}, urgency: {urgency})"
            )
            
            # Record model selection metric
            self.metrics.record_metric(
                agent_id=agent_id,
                agent_type=agent_type,
                metric_type=MetricType.TOOL_USAGE,
                value=1.0,
                metadata={
                    "tool_name": "llm",
                    "model": config.primary_model.value,
                    "task_type": task_type,
                    "complexity": complexity,
                    "urgency": urgency
                }
            )
            
            # Generate response
            response = await self.client.chat_completion(
                messages=messages,
                model=config.primary_model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
                stream=stream,
                use_cache=config.use_cache,
                fallback=config.enable_fallback,
                **kwargs
            )
            
            # Record token usage metrics (for non-streaming responses)
            if not stream and isinstance(response, ModelResponse):
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                self.metrics.record_metric(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    metric_type=MetricType.TOKEN_USAGE,
                    value=float(response.usage.total_tokens),
                    metadata={
                        "model": response.model,
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "response_time": response.response_time,
                        "cached": response.cached
                    }
                )
                
                self.metrics.record_metric(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    metric_type=MetricType.EXECUTION_TIME,
                    value=execution_time,
                    metadata={
                        "operation": "llm_generation",
                        "model": response.model,
                        "task_type": task_type
                    }
                )
                
                self.logger.info(
                    f"LLM response generated for {agent_id}: "
                    f"{response.usage.total_tokens} tokens in {response.response_time:.2f}s "
                    f"(cached: {response.cached})"
                )
            
            return response
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Record error metric
            self.metrics.record_metric(
                agent_id=agent_id,
                agent_type=agent_type,
                metric_type=MetricType.ERROR_RATE,
                value=1.0,
                metadata={
                    "operation": "llm_generation",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "execution_time": execution_time
                }
            )
            
            self.logger.error(f"LLM generation failed for {agent_id}: {e}")
            raise
    
    async def generate_structured_response(
        self,
        agent_id: str,
        agent_type: str,
        prompt: str,
        response_schema: Dict[str, Any],
        task_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured response following a schema.
        
        Args:
            agent_id: ID of the requesting agent
            agent_type: Type of agent
            prompt: Input prompt
            response_schema: Expected response schema
            task_type: Specific task type
            **kwargs: Additional parameters
            
        Returns:
            Structured response data
        """
        # Format prompt to request structured output
        schema_description = self._format_schema_description(response_schema)
        
        structured_prompt = f"""
{prompt}

Please provide your response in the following JSON format:
{schema_description}

Ensure your response is valid JSON that matches the schema exactly.
"""
        
        messages = [{"role": "user", "content": structured_prompt}]
        
        response = await self.generate_response(
            agent_id=agent_id,
            agent_type=agent_type,
            messages=messages,
            task_type=task_type,
            temperature=0.3,  # Lower temperature for structured output
            **kwargs
        )
        
        # Parse structured response
        if isinstance(response, ModelResponse):
            try:
                import json
                structured_data = json.loads(response.content)
                
                # Validate against schema (basic validation)
                if self._validate_schema(structured_data, response_schema):
                    return structured_data
                else:
                    self.logger.warning(f"Response doesn't match schema for {agent_id}")
                    return {"error": "Response doesn't match expected schema", "raw_content": response.content}
                    
            except json.JSONDecodeError:
                self.logger.warning(f"Invalid JSON response for {agent_id}")
                return {"error": "Invalid JSON response", "raw_content": response.content}
        
        return {"error": "Unexpected response type"}
    
    async def generate_with_context(
        self,
        agent_id: str,
        agent_type: str,
        user_input: str,
        context_data: List[Dict[str, Any]],
        task_type: Optional[str] = None,
        max_context_items: int = 5,
        **kwargs
    ) -> ModelResponse:
        """
        Generate response with relevant context data.
        
        Args:
            agent_id: ID of the requesting agent
            agent_type: Type of agent
            user_input: User input/question
            context_data: List of context items
            task_type: Specific task type
            max_context_items: Maximum context items to include
            **kwargs: Additional parameters
            
        Returns:
            Model response with context
        """
        # Format context
        context_text = self._format_context(context_data[:max_context_items])
        
        # Create context-aware prompt
        messages = []
        
        if context_text:
            messages.append({
                "role": "system",
                "content": f"You are an AI assistant. Use the following context information to help answer questions:\n\n{context_text}"
            })
        
        messages.append({
            "role": "user", 
            "content": user_input
        })
        
        return await self.generate_response(
            agent_id=agent_id,
            agent_type=agent_type,
            messages=messages,
            task_type=task_type,
            **kwargs
        )
    
    async def continue_conversation(
        self,
        agent_id: str,
        agent_type: str,
        conversation_history: List[Dict[str, str]],
        new_message: str,
        task_type: Optional[str] = None,
        max_history: int = 10,
        **kwargs
    ) -> ModelResponse:
        """
        Continue an existing conversation.
        
        Args:
            agent_id: ID of the requesting agent
            agent_type: Type of agent
            conversation_history: Previous conversation messages
            new_message: New message to add
            task_type: Specific task type
            max_history: Maximum history messages to include
            **kwargs: Additional parameters
            
        Returns:
            Model response
        """
        # Prepare messages with history
        messages = conversation_history[-max_history:] if conversation_history else []
        messages.append({"role": "user", "content": new_message})
        
        return await self.generate_response(
            agent_id=agent_id,
            agent_type=agent_type,
            messages=messages,
            task_type=task_type,
            **kwargs
        )
    
    def _get_optimized_config(
        self,
        agent_type: str,
        task_type: Optional[str],
        messages: List[Dict[str, str]],
        complexity: str,
        urgency: str
    ) -> ModelConfiguration:
        """Get optimized model configuration."""
        # Create cache key
        cache_key = f"{agent_type}_{task_type}_{complexity}_{urgency}"
        
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]
        
        # Get base configuration for agent
        base_config = get_model_config_for_agent(agent_type)
        
        # Override model if task-specific recommendation exists
        if task_type:
            recommended_model = get_model_for_task(task_type, complexity)
            base_config.primary_model = recommended_model
            
            # Adjust temperature for task
            task_temperature = get_temperature_for_task(task_type, agent_type)
            base_config.temperature = task_temperature
        
        # Calculate context length
        context_length = self.client.token_counter.count_message_tokens(
            messages, 
            base_config.primary_model.value
        )
        
        # Optimize configuration
        optimized_config = optimize_config_for_context(
            base_config=base_config,
            context_length=context_length,
            urgency=urgency
        )
        
        # Cache the configuration
        self._config_cache[cache_key] = optimized_config
        
        return optimized_config
    
    def _format_schema_description(self, schema: Dict[str, Any]) -> str:
        """Format schema description for prompt."""
        import json
        return json.dumps(schema, indent=2)
    
    def _validate_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Basic schema validation."""
        # Simple validation - check if required keys exist
        if "required" in schema:
            required_fields = schema["required"]
            for field in required_fields:
                if field not in data:
                    return False
        
        # Check if data has unexpected keys (basic check)
        if "properties" in schema:
            allowed_keys = set(schema["properties"].keys())
            data_keys = set(data.keys())
            
            # Allow extra keys but warn
            unexpected_keys = data_keys - allowed_keys
            if unexpected_keys:
                self.logger.debug(f"Unexpected keys in response: {unexpected_keys}")
        
        return True
    
    def _format_context(self, context_data: List[Dict[str, Any]]) -> str:
        """Format context data for inclusion in prompt."""
        if not context_data:
            return ""
        
        formatted_items = []
        for i, item in enumerate(context_data, 1):
            if isinstance(item, dict):
                # Format dict items nicely
                item_text = f"Context {i}:\n"
                for key, value in item.items():
                    if key in ["content", "text", "summary", "insight"]:
                        item_text += f"{value}\n"
                    else:
                        item_text += f"{key}: {value}\n"
                formatted_items.append(item_text)
            else:
                formatted_items.append(f"Context {i}: {str(item)}")
        
        return "\n".join(formatted_items)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check LLM integration health."""
        try:
            client_healthy = await self.client.health_check()
            usage_stats = self.client.get_usage_stats()
            
            return {
                "status": "healthy" if client_healthy else "unhealthy",
                "client_healthy": client_healthy,
                "usage_stats": usage_stats,
                "available_models": self.client.get_available_models(),
                "config_cache_size": len(self._config_cache)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "client_healthy": False
            }
    
    def get_model_recommendation(self, agent_type: str, task_type: str, complexity: str = "medium") -> str:
        """Get model recommendation for agent and task."""
        if task_type:
            model = get_model_for_task(task_type, complexity)
        else:
            config = get_model_config_for_agent(agent_type)
            model = config.primary_model
        
        return model.value
    
    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self._config_cache.clear()
        self.logger.info("Configuration cache cleared")