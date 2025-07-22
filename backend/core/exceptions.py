"""
Custom exceptions for the AI Research Assistant.
"""

from typing import Optional, Dict, Any


class AIResearchException(Exception):
    """Base exception for AI Research Assistant."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        super().__init__(self.message)


class AgentException(AIResearchException):
    """Exception raised by agents during execution."""
    pass


class AgentTimeoutException(AgentException):
    """Exception raised when agent execution times out."""
    pass


class AgentRetryExhaustedException(AgentException):
    """Exception raised when agent retry attempts are exhausted."""
    pass


class ModelException(AIResearchException):
    """Exception raised by LLM models."""
    pass


class MemoryException(AIResearchException):
    """Exception raised by memory management system."""
    pass


class CommunicationException(AIResearchException):
    """Exception raised during agent communication."""
    pass


class ToolException(AIResearchException):
    """Exception raised by agent tools."""
    pass


class ConfigurationException(AIResearchException):
    """Exception raised for configuration errors."""
    pass


class ValidationException(AIResearchException):
    """Exception raised for input validation errors."""
    pass