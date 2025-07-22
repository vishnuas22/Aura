"""
Enhanced base agent class with LLM integration.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import uuid

from crewai import Agent, Task, Crew

from .base_agent import BaseAIAgent
from ..core.config import AgentConfig, ConfigManager
from ..core.exceptions import AgentException, ModelException
from ..llm import LLMIntegration, get_llm_client, initialize_llm_client, GroqModel
from ..utils.logging_utils import log_agent_performance, log_agent_task


class LLMEnabledAgent(BaseAIAgent):
    """
    Enhanced base agent with integrated LLM capabilities.
    
    Extends BaseAIAgent with:
    - Direct LLM integration
    - Model selection optimization
    - Context-aware generation
    - Structured response handling
    - Token usage tracking
    - Performance optimization
    """
    
    def __init__(
        self,
        agent_type: str,
        config: Optional[AgentConfig] = None,
        agent_id: Optional[str] = None,
        memory_store_type: str = "memory",
        **kwargs
    ):
        """
        Initialize LLM-enabled agent.
        
        Args:
            agent_type: Type of agent (researcher, analyst, writer)
            config: Agent configuration
            agent_id: Unique agent identifier
            memory_store_type: Type of memory store
            **kwargs: Additional arguments
        """
        super().__init__(
            agent_type=agent_type,
            config=config,
            agent_id=agent_id,
            memory_store_type=memory_store_type,
            **kwargs
        )
        
        # Initialize LLM integration
        self._init_llm_integration()
        
        # LLM-specific attributes
        self._conversation_context: List[Dict[str, str]] = []
        self._max_context_length = 10
        self._preferred_models: Dict[str, GroqModel] = {}
        
        self.logger.info(f"LLM-enabled agent initialized: {self.agent_id}")
    
    def _init_llm_integration(self) -> None:
        """Initialize LLM integration."""
        try:
            llm_client = get_llm_client()
            
            if not llm_client:
                # Initialize client if not already done
                system_config = ConfigManager().get_system_config()
                llm_client = initialize_llm_client(
                    api_key=system_config.groq_api_key,
                    default_model=GroqModel.LLAMA_3_1_70B_VERSATILE,
                    enable_caching=True,
                    max_retries=3
                )
            
            self.llm = LLMIntegration(llm_client)
            self.logger.debug("LLM integration initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM integration: {e}")
            raise AgentException(f"LLM integration failed: {str(e)}")
    
    async def generate_llm_response(
        self,
        prompt: str,
        task_type: Optional[str] = None,
        complexity: str = "medium",
        urgency: str = "normal",
        use_context: bool = True,
        structured_schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate LLM response for the agent.
        
        Args:
            prompt: Input prompt
            task_type: Specific task type
            complexity: Task complexity (low, medium, high)
            urgency: Task urgency (low, normal, high)
            use_context: Whether to include conversation context
            structured_schema: Schema for structured response
            **kwargs: Additional parameters
            
        Returns:
            Generated response (string or structured data)
        """
        try:
            # Prepare conversation context
            messages = []
            
            # Add system prompt based on agent type
            system_prompt = self._get_system_prompt()
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add conversation context if requested
            if use_context and self._conversation_context:
                messages.extend(self._conversation_context[-self._max_context_length:])
            
            # Add current prompt
            messages.append({"role": "user", "content": prompt})
            
            # Generate response
            if structured_schema:
                # Generate structured response
                response_data = await self.llm.generate_structured_response(
                    agent_id=self.agent_id,
                    agent_type=self.agent_type,
                    prompt=prompt,
                    response_schema=structured_schema,
                    task_type=task_type,
                    complexity=complexity,
                    urgency=urgency,
                    **kwargs
                )
                
                # Update conversation context
                if use_context:
                    self._update_conversation_context(prompt, str(response_data))
                
                return response_data
            else:
                # Generate regular response
                response = await self.llm.generate_response(
                    agent_id=self.agent_id,
                    agent_type=self.agent_type,
                    messages=messages,
                    task_type=task_type,
                    complexity=complexity,
                    urgency=urgency,
                    **kwargs
                )
                
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                # Update conversation context
                if use_context:
                    self._update_conversation_context(prompt, response_text)
                
                # Store in memory
                await self.memory.store_conversation(
                    message=prompt,
                    role="user",
                    metadata={
                        "task_type": task_type,
                        "complexity": complexity,
                        "urgency": urgency
                    }
                )
                
                await self.memory.store_conversation(
                    message=response_text,
                    role="assistant",
                    metadata={
                        "model": response.model if hasattr(response, 'model') else "unknown",
                        "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else 0,
                        "response_time": response.response_time if hasattr(response, 'response_time') else 0,
                        "cached": response.cached if hasattr(response, 'cached') else False
                    }
                )
                
                return response_text
                
        except Exception as e:
            self.logger.error(f"LLM response generation failed: {e}")
            raise ModelException(f"LLM generation failed: {str(e)}")
    
    async def generate_with_memory_context(
        self,
        prompt: str,
        task_type: Optional[str] = None,
        max_context_items: int = 5,
        **kwargs
    ) -> str:
        """
        Generate response using relevant memory context.
        
        Args:
            prompt: Input prompt
            task_type: Specific task type
            max_context_items: Maximum context items to include
            **kwargs: Additional parameters
            
        Returns:
            Generated response with context
        """
        try:
            # Get relevant context from memory
            relevant_context = await self.memory.get_relevant_context(
                query=prompt,
                limit=max_context_items
            )
            
            # Format context data
            context_data = []
            for item in relevant_context:
                context_data.append({
                    "content": item.content.get("insight", item.content.get("message", "")),
                    "type": item.memory_type,
                    "importance": item.importance,
                    "timestamp": item.timestamp.isoformat()
                })
            
            # Generate response with context
            response = await self.llm.generate_with_context(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                user_input=prompt,
                context_data=context_data,
                task_type=task_type,
                max_context_items=max_context_items,
                **kwargs
            )
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Store interaction in memory
            await self.memory.store_conversation(
                message=f"Context-aware query: {prompt}",
                role="user",
                metadata={
                    "context_items_used": len(context_data),
                    "task_type": task_type
                }
            )
            
            await self.memory.store_conversation(
                message=response_text,
                role="assistant",
                metadata={
                    "context_enhanced": True,
                    "model": response.model if hasattr(response, 'model') else "unknown"
                }
            )
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"Context-aware generation failed: {e}")
            raise ModelException(f"Context-aware generation failed: {str(e)}")
    
    async def continue_conversation(
        self,
        message: str,
        task_type: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Continue existing conversation.
        
        Args:
            message: New message
            task_type: Specific task type
            **kwargs: Additional parameters
            
        Returns:
            Response message
        """
        response = await self.llm.continue_conversation(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            conversation_history=self._conversation_context,
            new_message=message,
            task_type=task_type,
            max_history=self._max_context_length,
            **kwargs
        )
        
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Update context
        self._update_conversation_context(message, response_text)
        
        return response_text
    
    def _update_conversation_context(self, user_message: str, assistant_response: str) -> None:
        """Update conversation context."""
        # Add user message
        self._conversation_context.append({
            "role": "user",
            "content": user_message
        })
        
        # Add assistant response
        self._conversation_context.append({
            "role": "assistant", 
            "content": assistant_response
        })
        
        # Trim context if too long
        max_messages = self._max_context_length * 2  # user + assistant pairs
        if len(self._conversation_context) > max_messages:
            self._conversation_context = self._conversation_context[-max_messages:]
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for the agent."""
        base_prompt = f"""You are a {self.config.role}. {self.config.backstory}

Your goal is: {self.config.goal}

Please provide helpful, accurate, and professional responses that align with your role and expertise."""
        
        # Add agent-specific instructions
        if self.agent_type == "researcher":
            base_prompt += """

Focus on:
- Gathering accurate and credible information
- Verifying facts and sources
- Providing comprehensive research findings
- Maintaining objectivity and citing sources when possible"""
            
        elif self.agent_type == "analyst":
            base_prompt += """

Focus on:
- Analyzing data and identifying patterns
- Providing strategic insights and recommendations
- Using analytical frameworks (SWOT, PEST, etc.)
- Presenting balanced and evidence-based conclusions"""
            
        elif self.agent_type == "writer":
            base_prompt += """

Focus on:
- Creating well-structured and engaging content
- Adapting tone and style to the target audience
- Ensuring clarity and readability
- Following proper grammar and formatting conventions"""
        
        return base_prompt
    
    async def get_llm_health_status(self) -> Dict[str, Any]:
        """Get LLM integration health status."""
        try:
            return await self.llm.health_check()
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "agent_id": self.agent_id
            }
    
    def get_model_recommendation(self, task_type: str, complexity: str = "medium") -> str:
        """Get model recommendation for task."""
        return self.llm.get_model_recommendation(
            agent_type=self.agent_type,
            task_type=task_type,
            complexity=complexity
        )
    
    def clear_conversation_context(self) -> None:
        """Clear conversation context."""
        self._conversation_context.clear()
        self.logger.debug(f"Conversation context cleared for {self.agent_id}")
    
    def set_context_length(self, max_length: int) -> None:
        """Set maximum conversation context length."""
        self._max_context_length = max(1, min(max_length, 20))  # Reasonable limits
        self.logger.debug(f"Context length set to {self._max_context_length} for {self.agent_id}")
    
    async def optimize_for_task(self, task_type: str, complexity: str = "medium", urgency: str = "normal") -> None:
        """
        Optimize agent configuration for specific task.
        
        Args:
            task_type: Type of task
            complexity: Task complexity
            urgency: Task urgency
        """
        try:
            # Get recommended model
            recommended_model = self.get_model_recommendation(task_type, complexity)
            self._preferred_models[task_type] = GroqModel(recommended_model)
            
            # Adjust context length based on urgency
            if urgency == "high":
                self.set_context_length(5)  # Shorter context for faster processing
            elif urgency == "low":
                self.set_context_length(15)  # Longer context for thorough processing
            
            self.logger.debug(f"Agent {self.agent_id} optimized for {task_type} task")
            
        except Exception as e:
            self.logger.warning(f"Task optimization failed for {self.agent_id}: {e}")
    
    async def execute_task_with_llm(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute task with enhanced LLM integration.
        
        This method should be overridden by subclasses to implement
        specific task execution logic with LLM integration.
        
        Args:
            task_data: Task data and parameters
            
        Returns:
            Task execution results
        """
        # Default implementation - subclasses should override
        task_type = task_data.get("type", "general")
        task_description = task_data.get("description", "")
        
        if not task_description:
            raise AgentException("Task description is required", error_code="MISSING_DESCRIPTION")
        
        # Optimize for task
        await self.optimize_for_task(
            task_type=task_type,
            complexity=task_data.get("complexity", "medium"),
            urgency=task_data.get("urgency", "normal")
        )
        
        # Generate response using LLM
        response = await self.generate_llm_response(
            prompt=task_description,
            task_type=task_type,
            complexity=task_data.get("complexity", "medium"),
            urgency=task_data.get("urgency", "normal"),
            use_context=task_data.get("use_context", True)
        )
        
        return {
            "result": response,
            "task_type": task_type,
            "agent_type": self.agent_type,
            "llm_enhanced": True
        }


# Update the factory to use LLM-enabled agents by default
def create_llm_enabled_agent(agent_class, **kwargs):
    """Create an LLM-enabled version of an agent class."""
    
    class LLMEnhancedAgent(agent_class, LLMEnabledAgent):
        """Agent class with LLM enhancement."""
        
        def __init__(self, **init_kwargs):
            # Initialize both parent classes
            agent_class.__init__(self, **init_kwargs)
            # LLMEnabledAgent initialization is handled by agent_class.__init__
            # since agent_class inherits from BaseAIAgent which handles LLM init
    
    return LLMEnhancedAgent(**kwargs)