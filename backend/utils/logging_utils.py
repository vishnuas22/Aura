"""
Logging utilities for AI agents.
"""

import logging
import sys
from typing import Optional, Dict, Any
from datetime import datetime
import json
import os
from pathlib import Path


class AgentLogFormatter(logging.Formatter):
    """Custom log formatter for agent logs."""
    
    def __init__(self, agent_id: str, include_agent_id: bool = True):
        """
        Initialize agent log formatter.
        
        Args:
            agent_id: Agent identifier
            include_agent_id: Whether to include agent ID in log messages
        """
        self.agent_id = agent_id
        self.include_agent_id = include_agent_id
        
        # Define format based on environment
        log_format = os.getenv('LOG_FORMAT', 'text')
        
        if log_format.lower() == 'json':
            super().__init__()
        else:
            if include_agent_id:
                format_string = (
                    '%(asctime)s - %(name)s - [%(agent_id)s] - '
                    '%(levelname)s - %(message)s'
                )
            else:
                format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            
            super().__init__(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    
    def format(self, record):
        """Format log record."""
        # Add agent ID to record
        if self.include_agent_id:
            record.agent_id = self.agent_id
        
        # Check if JSON format is requested
        log_format = os.getenv('LOG_FORMAT', 'text')
        
        if log_format.lower() == 'json':
            return self._format_json(record)
        else:
            return super().format(record)
    
    def _format_json(self, record) -> str:
        """Format record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add agent ID if enabled
        if self.include_agent_id:
            log_data['agent_id'] = self.agent_id
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 
                          'pathname', 'filename', 'module', 'lineno', 
                          'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process',
                          'getMessage', 'exc_info', 'exc_text', 'stack_info'):
                log_data[key] = value
        
        return json.dumps(log_data)


def get_agent_logger(
    agent_id: str,
    logger_name: Optional[str] = None,
    verbose: bool = True,
    file_logging: bool = True
) -> logging.Logger:
    """
    Get a configured logger for an agent.
    
    Args:
        agent_id: Agent identifier
        logger_name: Logger name (defaults to agent_id)
        verbose: Whether to enable verbose logging
        file_logging: Whether to enable file logging
        
    Returns:
        Configured logger instance
    """
    logger_name = logger_name or f"agent.{agent_id}"
    logger = logging.getLogger(logger_name)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Set log level based on environment and verbose flag
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    if verbose and log_level == 'INFO':
        log_level = 'DEBUG'
    
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Create formatter
    formatter = AgentLogFormatter(agent_id)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if file_logging:
        try:
            log_file_path = os.getenv('LOG_FILE_PATH', './logs/app.log')
            log_dir = Path(log_file_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        except Exception as e:
            # If file logging fails, log to console
            logger.warning(f"Failed to setup file logging: {e}")
    
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False
    
    return logger


def log_agent_performance(
    logger: logging.Logger,
    agent_id: str,
    operation: str,
    duration: float,
    success: bool = True,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log agent performance metrics.
    
    Args:
        logger: Logger instance
        agent_id: Agent identifier
        operation: Operation name
        duration: Operation duration in seconds
        success: Whether operation was successful
        metadata: Additional metadata
    """
    log_data = {
        'performance_metric': True,
        'agent_id': agent_id,
        'operation': operation,
        'duration_seconds': duration,
        'success': success
    }
    
    if metadata:
        log_data.update(metadata)
    
    # Log as structured data
    if success:
        logger.info(f"Performance: {operation} completed in {duration:.2f}s", extra=log_data)
    else:
        logger.warning(f"Performance: {operation} failed after {duration:.2f}s", extra=log_data)


def log_agent_communication(
    logger: logging.Logger,
    agent_id: str,
    action: str,  # 'sent' or 'received'
    message_type: str,
    target_agent_id: str,
    message_id: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log agent communication events.
    
    Args:
        logger: Logger instance
        agent_id: Agent identifier
        action: Communication action ('sent' or 'received')
        message_type: Type of message
        target_agent_id: Target agent identifier
        message_id: Message identifier
        metadata: Additional metadata
    """
    log_data = {
        'communication_event': True,
        'agent_id': agent_id,
        'action': action,
        'message_type': message_type,
        'target_agent_id': target_agent_id,
        'message_id': message_id
    }
    
    if metadata:
        log_data.update(metadata)
    
    logger.info(
        f"Communication: {action} {message_type} {action[:-1]} {target_agent_id}",
        extra=log_data
    )


def log_agent_task(
    logger: logging.Logger,
    agent_id: str,
    task_id: str,
    action: str,  # 'started', 'completed', 'failed'
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log agent task events.
    
    Args:
        logger: Logger instance
        agent_id: Agent identifier
        task_id: Task identifier
        action: Task action
        metadata: Additional metadata
    """
    log_data = {
        'task_event': True,
        'agent_id': agent_id,
        'task_id': task_id,
        'action': action
    }
    
    if metadata:
        log_data.update(metadata)
    
    log_level = logging.INFO
    if action == 'failed':
        log_level = logging.ERROR
    elif action == 'completed':
        log_level = logging.INFO
    
    logger.log(log_level, f"Task {action}: {task_id}", extra=log_data)


def log_agent_memory(
    logger: logging.Logger,
    agent_id: str,
    action: str,  # 'stored', 'retrieved', 'searched'
    memory_type: str,
    count: int = 1,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log agent memory operations.
    
    Args:
        logger: Logger instance
        agent_id: Agent identifier
        action: Memory action
        memory_type: Type of memory operation
        count: Number of items affected
        metadata: Additional metadata
    """
    log_data = {
        'memory_event': True,
        'agent_id': agent_id,
        'action': action,
        'memory_type': memory_type,
        'count': count
    }
    
    if metadata:
        log_data.update(metadata)
    
    logger.debug(
        f"Memory: {action} {count} {memory_type} item(s)",
        extra=log_data
    )


class ContextLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds context information to all log records."""
    
    def __init__(self, logger: logging.Logger, context: Dict[str, Any]):
        """
        Initialize context logger adapter.
        
        Args:
            logger: Base logger
            context: Context information to add to all records
        """
        super().__init__(logger, context)
    
    def process(self, msg, kwargs):
        """Process log record with context."""
        # Add context to extra
        extra = kwargs.setdefault('extra', {})
        extra.update(self.extra)
        
        return msg, kwargs


def get_context_logger(
    base_logger: logging.Logger,
    context: Dict[str, Any]
) -> ContextLoggerAdapter:
    """
    Get a context-aware logger adapter.
    
    Args:
        base_logger: Base logger instance
        context: Context information
        
    Returns:
        Context logger adapter
    """
    return ContextLoggerAdapter(base_logger, context)


# Setup root logger configuration
def setup_logging():
    """Setup basic logging configuration."""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_format = os.getenv('LOG_FORMAT', 'text').lower()
    
    # Basic configuration
    if log_format == 'json':
        formatter = logging.Formatter('%(message)s')
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    root_logger.addHandler(console_handler)
    
    # Suppress noisy loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)


# Initialize logging on import
setup_logging()