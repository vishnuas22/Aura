"""
Base agent class with common functionality for all AI agents.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime
import uuid

# from crewai import Agent, LLM
# from langchain_groq import ChatGroq

from core.config import AgentConfig, ConfigManager
from core.exceptions import (
    AgentException, AgentTimeoutException, AgentRetryExhaustedException, ModelException
)
from core.metrics import get_metrics_collector, MetricsCollector
from core.communication import (
    AgentCommunicator, MessageHandler, AgentMessage, MessageType,
    get_communication_channel, DefaultMessageHandler
)
from memory.agent_memory import AgentMemoryManager, create_memory_store
from utils.retry import RetryHandler
from utils.logging_utils import get_agent_logger
from llm import LLMIntegration, get_llm_client, initialize_llm_client, GroqModel


class BaseAIAgent(ABC):
    """
    Base class for all AI agents with common functionality.
    
    Provides:
    - Configuration management
    - Memory management
    - Communication protocols
    - Performance metrics
    - Error handling and retries
    - Verbose logging
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
        Initialize base agent.
        
        Args:
            agent_type: Type of agent (researcher, analyst, writer)
            config: Agent configuration (loaded from config if None)
            agent_id: Unique agent identifier (generated if None)
            memory_store_type: Type of memory store to use
            **kwargs: Additional arguments
        """
        # Basic identification
        self.agent_id = agent_id or f"{agent_type}_{uuid.uuid4().hex[:8]}"
        self.agent_type = agent_type
        
        # Load configuration
        if config is None:
            config_manager = ConfigManager()
            config = config_manager.get_agent_config(agent_type)
        self.config = config
        
        # Initialize logging
        self.logger = get_agent_logger(self.agent_id, verbose=config.verbose)
        self.logger.info(f"Initializing {agent_type} agent: {self.agent_id}")
        
        # Initialize metrics collector
        self.metrics = get_metrics_collector()
        
        # Initialize memory management
        self._init_memory(memory_store_type)
        
        # Initialize communication
        self._init_communication()
        
        # Initialize retry handler
        self.retry_handler = RetryHandler(
            max_retries=config.max_retries,
            backoff_factor=2.0,
            max_wait_time=60.0
        )

        # Initialize LLM integration
        self._init_llm_integration()
        
        # CrewAI agent (initialized in subclass)
        # self._crew_agent: Optional[Agent] = None
        # self._llm: Optional[LLM] = None
        self._crew_agent = None
        self._llm = None
        
        # State tracking
        self._current_task_id: Optional[str] = None
        self._execution_id: Optional[str] = None
        self._is_busy = False
        self._tools: List[Any] = []
        
        # LLM-specific attributes
        self._conversation_context: List[Dict[str, str]] = []
        self._max_context_length = 10
        
        self.logger.info(f"Agent {self.agent_id} initialized successfully")
    
    def _init_memory(self, store_type: str) -> None:
        """Initialize memory management."""
        try:
            # For now, use in-memory store (MongoDB integration will be added later)
            memory_store = create_memory_store(store_type)
            self.memory = AgentMemoryManager(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                store=memory_store,
                config=ConfigManager().get_system_config().memory
            )
            self.logger.debug("Memory management initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory: {e}")
            raise AgentException(f"Memory initialization failed: {str(e)}")
    
    def _init_communication(self) -> None:
        """Initialize communication system."""
        try:
            channel = get_communication_channel()
            
            # Create custom message handler
            handler = self._create_message_handler()
            
            self.communicator = AgentCommunicator(
                agent_id=self.agent_id,
                channel=channel,
                handler=handler
            )
            self.logger.debug("Communication system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize communication: {e}")
            raise AgentException(f"Communication initialization failed: {str(e)}")
    
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
    
    def _create_message_handler(self) -> MessageHandler:
        """Create a custom message handler for this agent."""
        return AgentMessageHandler(self)
    
    def _init_crew_agent(self, tools: Optional[List[Any]] = None) -> None:
        """Initialize the agent (simplified version without CrewAI)."""
        try:
            # Store tools
            self._tools = tools or []
            
            self.logger.info(f"Agent initialized with {len(self._tools)} tools")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agent: {e}")
            raise AgentException(f"Agent initialization failed: {str(e)}")
    
    @abstractmethod
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task - must be implemented by subclasses.
        
        Args:
            task_data: Task data and parameters
            
        Returns:
            Task results
        """
        pass
    
    @abstractmethod
    def get_available_tools(self) -> List[str]:
        """
        Get list of available tools - must be implemented by subclasses.
        
        Returns:
            List of tool names
        """
        pass
    
    async def execute_task(
        self, 
        task_data: Dict[str, Any],
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a task with full error handling, retries, and metrics.
        
        Args:
            task_data: Task data and parameters
            task_id: Optional task identifier
            
        Returns:
            Task execution results
            
        Raises:
            AgentException: If task execution fails
            AgentTimeoutException: If task times out
            AgentRetryExhaustedException: If all retries are exhausted
        """
        task_id = task_id or f"task_{int(time.time())}"
        self._current_task_id = task_id
        
        # Check if agent is busy
        if self._is_busy:
            raise AgentException(
                f"Agent {self.agent_id} is busy with another task",
                error_code="AGENT_BUSY"
            )
        
        self._is_busy = True
        
        # Start metrics tracking
        self._execution_id = self.metrics.start_execution(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            task_id=task_id
        )
        
        self.logger.info(f"Starting task execution: {task_id}")
        
        # Store task in memory
        await self.memory.store_conversation(
            message=f"Starting task: {task_data.get('description', 'No description')}",
            role="system",
            metadata={
                "task_id": task_id,
                "task_type": "execution_start"
            }
        )
        
        try:
            # Execute with retry logic
            result = await self.retry_handler.execute_with_retry(
                func=self._execute_with_timeout,
                task_data=task_data,
                task_id=task_id,
                on_retry=lambda attempt, error: self._on_retry(attempt, error, task_id)
            )
            
            # Store successful result
            await self.memory.store_insight(
                insight=f"Task completed successfully: {result.get('summary', 'No summary')}",
                category="task_completion",
                confidence=0.9,
                metadata={
                    "task_id": task_id,
                    "execution_time": result.get('execution_time', 0)
                }
            )
            
            self.logger.info(f"Task completed successfully: {task_id}")
            
            # End metrics tracking
            self.metrics.end_execution(
                execution_id=self._execution_id,
                success=True,
                tokens_used=result.get('tokens_used', 0),
                tools_used=result.get('tools_used', []),
                memory_peak=result.get('memory_peak', 0.0)
            )
            
            return result
            
        except Exception as e:
            # Store failed result
            await self.memory.store_conversation(
                message=f"Task failed with error: {str(e)}",
                role="system",
                metadata={
                    "task_id": task_id,
                    "error_type": type(e).__name__,
                    "task_type": "execution_error"
                }
            )
            
            self.logger.error(f"Task failed: {task_id} - {str(e)}")
            
            # End metrics tracking
            if self._execution_id:
                self.metrics.end_execution(
                    execution_id=self._execution_id,
                    success=False,
                    error_message=str(e)
                )
            
            raise
            
        finally:
            self._is_busy = False
            self._current_task_id = None
            self._execution_id = None
    
    async def _execute_with_timeout(self, task_data: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        """Execute task with timeout protection."""
        try:
            result = await asyncio.wait_for(
                self.process_task(task_data),
                timeout=self.config.timeout
            )
            return result
            
        except asyncio.TimeoutError:
            raise AgentTimeoutException(
                f"Task execution timed out after {self.config.timeout} seconds",
                error_code="AGENT_TIMEOUT"
            )
    
    def _on_retry(self, attempt: int, error: Exception, task_id: str) -> None:
        """Handle retry attempts."""
        self.logger.warning(f"Retry attempt {attempt} for task {task_id}: {str(error)}")
        
        # Record retry in metrics
        if self._execution_id:
            self.metrics.record_retry(self._execution_id)
    
    async def start(self) -> None:
        """Start the agent and subscribe to communication."""
        try:
            await self.communicator.subscribe()
            self.logger.info(f"Agent {self.agent_id} started and ready for tasks")
            
        except Exception as e:
            self.logger.error(f"Failed to start agent: {e}")
            raise AgentException(f"Agent startup failed: {str(e)}")
    
    async def stop(self) -> None:
        """Stop the agent and cleanup resources."""
        try:
            # Wait for current task to complete if busy
            if self._is_busy and self._current_task_id:
                self.logger.info(f"Waiting for current task to complete: {self._current_task_id}")
                # Add logic to gracefully stop current task if needed
            
            # Unsubscribe from communication
            await self.communicator.unsubscribe()
            
            # Cleanup old memories
            await self.memory.cleanup_old_memories(days=7)
            
            self.logger.info(f"Agent {self.agent_id} stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping agent: {e}")
            raise AgentException(f"Agent shutdown failed: {str(e)}")
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status.
        
        Returns:
            Status information dictionary
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "is_busy": self._is_busy,
            "current_task_id": self._current_task_id,
            "available_tools": self.get_available_tools(),
            "memory_enabled": self.config.memory_enabled,
            "performance_tracking": self.config.performance_tracking,
            "uptime": time.time(),  # Simple uptime tracking
            "last_active": datetime.utcnow().isoformat()
        }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for this agent.
        
        Returns:
            Performance metrics dictionary
        """
        return self.metrics.get_performance_summary(self.agent_type)
    
    async def share_data_with_agent(
        self, 
        target_agent_id: str, 
        data: Dict[str, Any],
        data_type: str = "general"
    ) -> None:
        """
        Share data with another agent.
        
        Args:
            target_agent_id: ID of target agent
            data: Data to share
            data_type: Type of data
        """
        try:
            await self.communicator.send_data_share(
                receiver_id=target_agent_id,
                data=data,
                data_type=data_type
            )
            
            self.logger.info(f"Shared {data_type} data with agent {target_agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to share data with {target_agent_id}: {e}")
            raise AgentException(f"Data sharing failed: {str(e)}")
    
    async def handoff_task_to_agent(
        self, 
        target_agent_id: str,
        task_context: Dict[str, Any],
        instructions: str = ""
    ) -> Dict[str, Any]:
        """
        Hand off current task to another agent.
        
        Args:
            target_agent_id: ID of target agent
            task_context: Context and data for the task
            instructions: Special instructions
            
        Returns:
            Response from target agent
        """
        try:
            response = await self.communicator.send_handoff(
                receiver_id=target_agent_id,
                task_context=task_context,
                instructions=instructions
            )
            
            self.logger.info(f"Task handed off to agent {target_agent_id}")
            
            return response.content if response else {}
            
        except Exception as e:
            self.logger.error(f"Failed to handoff task to {target_agent_id}: {e}")
            raise AgentException(f"Task handoff failed: {str(e)}")
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(id={self.agent_id}, type={self.agent_type})"
    
    # LLM Integration Methods
    
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
    
    def clear_conversation_context(self) -> None:
        """Clear conversation context."""
        self._conversation_context.clear()
        self.logger.debug(f"Conversation context cleared for {self.agent_id}")
    
    def set_context_length(self, max_length: int) -> None:
        """Set maximum conversation context length."""
        self._max_context_length = max(1, min(max_length, 20))  # Reasonable limits
        self.logger.debug(f"Context length set to {self._max_context_length} for {self.agent_id}")


class AgentMessageHandler(MessageHandler):
    """Custom message handler for BaseAIAgent."""
    
    def __init__(self, agent: BaseAIAgent):
        """
        Initialize message handler.
        
        Args:
            agent: The agent instance
        """
        self.agent = agent
        self.logger = agent.logger
    
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Handle incoming messages for the agent.
        
        Args:
            message: Incoming message
            
        Returns:
            Optional response message
        """
        self.logger.debug(f"Handling {message.message_type.value} from {message.sender_id}")
        
        # Store incoming message in memory
        await self.agent.memory.store_conversation(
            message=f"Received {message.message_type.value}: {message.content}",
            role="system",
            metadata={
                "sender_id": message.sender_id,
                "message_type": message.message_type.value,
                "message_id": message.id
            }
        )
        
        try:
            if message.message_type == MessageType.TASK_REQUEST:
                return await self._handle_task_request(message)
            elif message.message_type == MessageType.DATA_SHARE:
                return await self._handle_data_share(message)
            elif message.message_type == MessageType.HANDOFF:
                return await self._handle_handoff(message)
            elif message.message_type == MessageType.TASK_RESPONSE:
                return await self._handle_task_response(message)
            else:
                # Use default handler for other message types
                default_handler = DefaultMessageHandler(self.agent.agent_id)
                return await default_handler.handle_message(message)
                
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            
            # Return error response if response is required
            if message.requires_response:
                return AgentMessage(
                    sender_id=self.agent.agent_id,
                    receiver_id=message.sender_id,
                    message_type=MessageType.ERROR,
                    correlation_id=message.id,
                    content={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "original_message_id": message.id
                    }
                )
            
            return None
    
    async def _handle_task_request(self, message: AgentMessage) -> AgentMessage:
        """Handle task request messages."""
        task_data = message.content.get("task", {})
        
        # Check if agent is available
        if self.agent._is_busy:
            return AgentMessage(
                sender_id=self.agent.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.TASK_RESPONSE,
                correlation_id=message.id,
                content={
                    "status": "busy",
                    "message": "Agent is currently busy with another task",
                    "retry_after": 60  # seconds
                }
            )
        
        # Accept the task
        self.logger.info(f"Accepting task request from {message.sender_id}")
        
        # Execute task asynchronously
        asyncio.create_task(self._execute_requested_task(task_data, message))
        
        return AgentMessage(
            sender_id=self.agent.agent_id,
            receiver_id=message.sender_id,
            message_type=MessageType.TASK_RESPONSE,
            correlation_id=message.id,
            content={
                "status": "accepted",
                "message": "Task accepted and will be processed",
                "estimated_completion": self.agent.config.timeout
            }
        )
    
    async def _execute_requested_task(self, task_data: Dict[str, Any], original_message: AgentMessage) -> None:
        """Execute a task requested by another agent."""
        try:
            # Execute the task
            result = await self.agent.execute_task(task_data)
            
            # Send completion notification
            completion_message = AgentMessage(
                sender_id=self.agent.agent_id,
                receiver_id=original_message.sender_id,
                message_type=MessageType.TASK_UPDATE,
                content={
                    "status": "completed",
                    "result": result,
                    "original_message_id": original_message.id
                }
            )
            
            await self.agent.communicator.channel.send_message(completion_message)
            
        except Exception as e:
            # Send error notification
            error_message = AgentMessage(
                sender_id=self.agent.agent_id,
                receiver_id=original_message.sender_id,
                message_type=MessageType.ERROR,
                content={
                    "status": "failed",
                    "error": str(e),
                    "original_message_id": original_message.id
                }
            )
            
            await self.agent.communicator.channel.send_message(error_message)
    
    async def _handle_data_share(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle data sharing messages."""
        data = message.content.get("data", {})
        data_type = message.content.get("data_type", "general")
        
        # Store shared data in memory
        await self.agent.memory.store_conversation(
            message=f"Received {data_type} data from {message.sender_id}",
            role="system",
            metadata={
                "data_type": data_type,
                "sender_id": message.sender_id,
                "data_keys": list(data.keys()) if isinstance(data, dict) else []
            }
        )
        
        self.logger.info(f"Received {data_type} data from {message.sender_id}")
        return None  # No response needed for data sharing
    
    async def _handle_handoff(self, message: AgentMessage) -> AgentMessage:
        """Handle task handoff messages."""
        task_context = message.content.get("task_context", {})
        instructions = message.content.get("instructions", "")
        
        # Check if agent is available
        if self.agent._is_busy:
            return AgentMessage(
                sender_id=self.agent.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.TASK_RESPONSE,
                correlation_id=message.id,
                content={
                    "status": "handoff_rejected",
                    "message": "Agent is currently busy",
                    "retry_after": 60
                }
            )
        
        # Accept the handoff
        self.logger.info(f"Accepting task handoff from {message.sender_id}")
        
        # Store handoff information
        await self.agent.memory.store_insight(
            insight=f"Received task handoff from {message.sender_id}: {instructions}",
            category="task_handoff",
            confidence=0.8,
            metadata={
                "sender_id": message.sender_id,
                "context_keys": list(task_context.keys()) if isinstance(task_context, dict) else []
            }
        )
        
        # Execute handoff task asynchronously
        asyncio.create_task(self._execute_requested_task(task_context, message))
        
        return AgentMessage(
            sender_id=self.agent.agent_id,
            receiver_id=message.sender_id,
            message_type=MessageType.TASK_RESPONSE,
            correlation_id=message.id,
            content={
                "status": "handoff_accepted",
                "message": "Task handoff accepted and will be processed",
                "estimated_completion": self.agent.config.timeout
            }
        )
    
    async def _handle_task_response(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle task response messages."""
        # This is typically handled by the communicator for request/response correlation
        # But we can store it for memory purposes
        status = message.content.get("status", "unknown")
        
        await self.agent.memory.store_conversation(
            message=f"Received task response from {message.sender_id}: {status}",
            role="system",
            metadata={
                "sender_id": message.sender_id,
                "response_status": status,
                "correlation_id": message.correlation_id
            }
        )
        
        return None