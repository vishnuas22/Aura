"""
Communication protocol for multi-agent interactions.
"""

import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import uuid

from .exceptions import CommunicationException


class MessageType(Enum):
    """Types of messages between agents."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response" 
    TASK_UPDATE = "task_update"
    DATA_SHARE = "data_share"
    FEEDBACK = "feedback"
    ERROR = "error"
    HANDOFF = "handoff"
    SYSTEM = "system"


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class AgentMessage:
    """Message structure for agent communication."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: str = ""
    message_type: MessageType = MessageType.TASK_REQUEST
    priority: MessagePriority = MessagePriority.NORMAL
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None  # For request/response correlation
    requires_response: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        data = asdict(self)
        data['message_type'] = self.message_type.value
        data['priority'] = self.priority.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create message from dictionary."""
        data['message_type'] = MessageType(data['message_type'])
        data['priority'] = MessagePriority(data['priority'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class MessageHandler(ABC):
    """Abstract base class for message handlers."""
    
    @abstractmethod
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Handle an incoming message.
        
        Args:
            message: Incoming message
            
        Returns:
            Optional response message
        """
        pass


class CommunicationChannel(ABC):
    """Abstract base class for communication channels."""
    
    @abstractmethod
    async def send_message(self, message: AgentMessage) -> None:
        """Send a message through the channel."""
        pass
    
    @abstractmethod
    async def receive_message(self, timeout: Optional[float] = None) -> Optional[AgentMessage]:
        """Receive a message from the channel."""
        pass
    
    @abstractmethod
    async def subscribe(self, agent_id: str, handler: MessageHandler) -> None:
        """Subscribe an agent to receive messages."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, agent_id: str) -> None:
        """Unsubscribe an agent from receiving messages."""
        pass


class InMemoryChannel(CommunicationChannel):
    """In-memory communication channel for local agent communication."""
    
    def __init__(self):
        """Initialize in-memory channel."""
        self._queues: Dict[str, asyncio.Queue] = {}
        self._handlers: Dict[str, MessageHandler] = {}
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)
    
    async def send_message(self, message: AgentMessage) -> None:
        """Send a message to the target agent's queue."""
        async with self._lock:
            if message.receiver_id not in self._queues:
                self._queues[message.receiver_id] = asyncio.Queue()
            
            await self._queues[message.receiver_id].put(message)
            self._logger.debug(f"Message sent from {message.sender_id} to {message.receiver_id}")
    
    async def receive_message(self, timeout: Optional[float] = None) -> Optional[AgentMessage]:
        """Receive a message (not used in this implementation)."""
        # This method is not used in the in-memory implementation
        # Messages are delivered directly to agent queues
        return None
    
    async def subscribe(self, agent_id: str, handler: MessageHandler) -> None:
        """Subscribe an agent to receive messages."""
        async with self._lock:
            if agent_id not in self._queues:
                self._queues[agent_id] = asyncio.Queue()
            
            self._handlers[agent_id] = handler
            self._logger.debug(f"Agent {agent_id} subscribed to communication channel")
            
            # Start message processing task for this agent
            asyncio.create_task(self._process_messages(agent_id))
    
    async def unsubscribe(self, agent_id: str) -> None:
        """Unsubscribe an agent from receiving messages."""
        async with self._lock:
            if agent_id in self._handlers:
                del self._handlers[agent_id]
            if agent_id in self._queues:
                # Clear the queue
                while not self._queues[agent_id].empty():
                    try:
                        self._queues[agent_id].get_nowait()
                    except asyncio.QueueEmpty:
                        break
                del self._queues[agent_id]
            
            self._logger.debug(f"Agent {agent_id} unsubscribed from communication channel")
    
    async def _process_messages(self, agent_id: str) -> None:
        """Process messages for a specific agent."""
        while agent_id in self._handlers:
            try:
                queue = self._queues.get(agent_id)
                if not queue:
                    break
                
                # Wait for message with timeout
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                handler = self._handlers.get(agent_id)
                if handler:
                    try:
                        response = await handler.handle_message(message)
                        if response:
                            await self.send_message(response)
                    except Exception as e:
                        self._logger.error(f"Error handling message for {agent_id}: {e}")
                        
                        # Send error response if required
                        if message.requires_response:
                            error_response = AgentMessage(
                                sender_id=agent_id,
                                receiver_id=message.sender_id,
                                message_type=MessageType.ERROR,
                                correlation_id=message.id,
                                content={
                                    "error": str(e),
                                    "original_message_id": message.id
                                }
                            )
                            await self.send_message(error_response)
                
            except Exception as e:
                self._logger.error(f"Error in message processing for {agent_id}: {e}")
                await asyncio.sleep(1)  # Brief pause before retrying


class AgentCommunicator:
    """Handles communication for individual agents."""
    
    def __init__(
        self, 
        agent_id: str, 
        channel: CommunicationChannel,
        handler: Optional[MessageHandler] = None
    ):
        """
        Initialize agent communicator.
        
        Args:
            agent_id: Unique agent identifier
            channel: Communication channel to use
            handler: Optional message handler
        """
        self.agent_id = agent_id
        self.channel = channel
        self.handler = handler
        self._logger = logging.getLogger(__name__)
        self._response_futures: Dict[str, asyncio.Future] = {}
        self._subscribed = False
    
    async def subscribe(self, handler: Optional[MessageHandler] = None) -> None:
        """Subscribe to receive messages."""
        if handler:
            self.handler = handler
        
        if not self.handler:
            raise CommunicationException(
                "Message handler is required for subscription",
                error_code="COMMUNICATION_NO_HANDLER"
            )
        
        await self.channel.subscribe(self.agent_id, self.handler)
        self._subscribed = True
        self._logger.debug(f"Agent {self.agent_id} subscribed to communication")
    
    async def unsubscribe(self) -> None:
        """Unsubscribe from receiving messages."""
        if self._subscribed:
            await self.channel.unsubscribe(self.agent_id)
            self._subscribed = False
            self._logger.debug(f"Agent {self.agent_id} unsubscribed from communication")
    
    async def send_message(
        self, 
        receiver_id: str,
        message_type: MessageType,
        content: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        requires_response: bool = False,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[AgentMessage]:
        """
        Send a message to another agent.
        
        Args:
            receiver_id: Target agent ID
            message_type: Type of message
            content: Message content
            priority: Message priority
            requires_response: Whether response is required
            correlation_id: Correlation ID for request/response
            metadata: Additional metadata
            
        Returns:
            Response message if requires_response is True
        """
        message = AgentMessage(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            priority=priority,
            content=content,
            requires_response=requires_response,
            correlation_id=correlation_id,
            metadata=metadata or {}
        )
        
        # If response is required, set up future for response
        response_future = None
        if requires_response:
            response_future = asyncio.Future()
            self._response_futures[message.id] = response_future
        
        try:
            await self.channel.send_message(message)
            self._logger.debug(f"Message sent from {self.agent_id} to {receiver_id}")
            
            if requires_response and response_future:
                # Wait for response with timeout
                try:
                    response = await asyncio.wait_for(response_future, timeout=30.0)
                    return response
                except asyncio.TimeoutError:
                    raise CommunicationException(
                        f"Timeout waiting for response from {receiver_id}",
                        error_code="COMMUNICATION_TIMEOUT"
                    )
                finally:
                    # Clean up future
                    self._response_futures.pop(message.id, None)
            
            return None
            
        except Exception as e:
            if requires_response:
                self._response_futures.pop(message.id, None)
            raise CommunicationException(
                f"Failed to send message: {str(e)}",
                error_code="COMMUNICATION_SEND_ERROR"
            )
    
    async def send_task_request(
        self, 
        receiver_id: str,
        task_data: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> AgentMessage:
        """
        Send a task request to another agent.
        
        Args:
            receiver_id: Target agent ID
            task_data: Task data
            priority: Message priority
            
        Returns:
            Response message from the receiving agent
        """
        return await self.send_message(
            receiver_id=receiver_id,
            message_type=MessageType.TASK_REQUEST,
            content={"task": task_data},
            priority=priority,
            requires_response=True
        )
    
    async def send_data_share(
        self, 
        receiver_id: str,
        data: Dict[str, Any],
        data_type: str = "general"
    ) -> None:
        """
        Share data with another agent.
        
        Args:
            receiver_id: Target agent ID
            data: Data to share
            data_type: Type of data being shared
        """
        await self.send_message(
            receiver_id=receiver_id,
            message_type=MessageType.DATA_SHARE,
            content={
                "data": data,
                "data_type": data_type
            }
        )
    
    async def send_handoff(
        self, 
        receiver_id: str,
        task_context: Dict[str, Any],
        instructions: str = ""
    ) -> AgentMessage:
        """
        Hand off a task to another agent.
        
        Args:
            receiver_id: Target agent ID
            task_context: Context and data for the task
            instructions: Special instructions for the receiving agent
            
        Returns:
            Acknowledgment message from the receiving agent
        """
        return await self.send_message(
            receiver_id=receiver_id,
            message_type=MessageType.HANDOFF,
            content={
                "task_context": task_context,
                "instructions": instructions
            },
            priority=MessagePriority.HIGH,
            requires_response=True
        )
    
    def handle_response(self, message: AgentMessage) -> None:
        """Handle a response message."""
        if message.correlation_id and message.correlation_id in self._response_futures:
            future = self._response_futures[message.correlation_id]
            if not future.done():
                future.set_result(message)


class DefaultMessageHandler(MessageHandler):
    """Default message handler implementation."""
    
    def __init__(self, agent_id: str):
        """
        Initialize default message handler.
        
        Args:
            agent_id: Agent identifier
        """
        self.agent_id = agent_id
        self._logger = logging.getLogger(__name__)
    
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle incoming messages with basic responses."""
        self._logger.info(f"Agent {self.agent_id} received {message.message_type.value} from {message.sender_id}")
        
        # Handle different message types
        if message.message_type == MessageType.TASK_REQUEST:
            return await self._handle_task_request(message)
        elif message.message_type == MessageType.DATA_SHARE:
            return await self._handle_data_share(message)
        elif message.message_type == MessageType.HANDOFF:
            return await self._handle_handoff(message)
        elif message.message_type == MessageType.FEEDBACK:
            return await self._handle_feedback(message)
        
        # Default: acknowledge receipt if response is required
        if message.requires_response:
            return AgentMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.TASK_RESPONSE,
                correlation_id=message.id,
                content={"status": "acknowledged", "message": "Message received"}
            )
        
        return None
    
    async def _handle_task_request(self, message: AgentMessage) -> AgentMessage:
        """Handle task request messages."""
        return AgentMessage(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type=MessageType.TASK_RESPONSE,
            correlation_id=message.id,
            content={
                "status": "accepted",
                "message": "Task request received and will be processed"
            }
        )
    
    async def _handle_data_share(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle data sharing messages."""
        # Log the received data
        data_type = message.content.get("data_type", "unknown")
        self._logger.info(f"Received {data_type} data from {message.sender_id}")
        return None  # No response needed for data sharing
    
    async def _handle_handoff(self, message: AgentMessage) -> AgentMessage:
        """Handle task handoff messages."""
        return AgentMessage(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type=MessageType.TASK_RESPONSE,
            correlation_id=message.id,
            content={
                "status": "handoff_accepted",
                "message": "Task handoff accepted and will begin processing"
            }
        )
    
    async def _handle_feedback(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle feedback messages."""
        feedback_type = message.content.get("type", "general")
        self._logger.info(f"Received {feedback_type} feedback from {message.sender_id}")
        return None  # No response needed for feedback


# Global communication channel instance
_global_channel: Optional[CommunicationChannel] = None


def get_communication_channel() -> CommunicationChannel:
    """Get the global communication channel."""
    global _global_channel
    if _global_channel is None:
        _global_channel = InMemoryChannel()
    return _global_channel


def initialize_communication(channel: Optional[CommunicationChannel] = None) -> CommunicationChannel:
    """Initialize the global communication channel."""
    global _global_channel
    _global_channel = channel or InMemoryChannel()
    return _global_channel