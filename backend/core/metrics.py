"""
Performance metrics tracking for agents.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque


class MetricType(Enum):
    """Types of metrics to track."""
    EXECUTION_TIME = "execution_time"
    SUCCESS_RATE = "success_rate"
    ERROR_RATE = "error_rate"
    RETRY_COUNT = "retry_count"
    TOKEN_USAGE = "token_usage"
    MEMORY_USAGE = "memory_usage"
    TOOL_USAGE = "tool_usage"


@dataclass
class AgentMetric:
    """Individual agent metric data."""
    agent_id: str
    agent_type: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionMetrics:
    """Metrics for a single agent execution."""
    agent_id: str
    agent_type: str
    task_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    error_message: Optional[str] = None
    retry_count: int = 0
    tokens_used: int = 0
    tools_used: List[str] = field(default_factory=list)
    memory_peak: float = 0.0  # MB
    
    @property
    def execution_time(self) -> float:
        """Get execution time in seconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()


class MetricsCollector:
    """Collects and manages performance metrics for agents."""
    
    def __init__(self, max_history: int = 10000):
        """
        Initialize metrics collector.
        
        Args:
            max_history: Maximum number of metrics to keep in memory
        """
        self.max_history = max_history
        self._metrics: deque = deque(maxlen=max_history)
        self._executions: Dict[str, ExecutionMetrics] = {}
        self._aggregated: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._lock = threading.Lock()
        self._logger = logging.getLogger(__name__)
        
    def start_execution(
        self, 
        agent_id: str, 
        agent_type: str, 
        task_id: str
    ) -> str:
        """
        Start tracking an agent execution.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (researcher, analyst, writer)
            task_id: Task identifier
            
        Returns:
            Execution tracking ID
        """
        execution_id = f"{agent_id}_{task_id}_{int(time.time())}"
        
        with self._lock:
            self._executions[execution_id] = ExecutionMetrics(
                agent_id=agent_id,
                agent_type=agent_type,
                task_id=task_id,
                start_time=datetime.utcnow()
            )
        
        self._logger.debug(f"Started tracking execution: {execution_id}")
        return execution_id
    
    def end_execution(
        self, 
        execution_id: str, 
        success: bool = True,
        error_message: Optional[str] = None,
        tokens_used: int = 0,
        tools_used: Optional[List[str]] = None,
        memory_peak: float = 0.0
    ) -> Optional[ExecutionMetrics]:
        """
        End tracking an agent execution.
        
        Args:
            execution_id: Execution tracking ID
            success: Whether execution was successful
            error_message: Error message if execution failed
            tokens_used: Number of tokens used
            tools_used: List of tools used
            memory_peak: Peak memory usage in MB
            
        Returns:
            ExecutionMetrics if found, None otherwise
        """
        with self._lock:
            if execution_id not in self._executions:
                self._logger.warning(f"Execution not found: {execution_id}")
                return None
            
            execution = self._executions[execution_id]
            execution.end_time = datetime.utcnow()
            execution.success = success
            execution.error_message = error_message
            execution.tokens_used = tokens_used
            execution.tools_used = tools_used or []
            execution.memory_peak = memory_peak
            
            # Create individual metrics
            self._create_execution_metrics(execution)
            
            # Remove from active executions
            del self._executions[execution_id]
            
            self._logger.debug(f"Ended tracking execution: {execution_id}")
            return execution
    
    def record_retry(self, execution_id: str) -> None:
        """
        Record a retry attempt.
        
        Args:
            execution_id: Execution tracking ID
        """
        with self._lock:
            if execution_id in self._executions:
                self._executions[execution_id].retry_count += 1
                self._logger.debug(f"Recorded retry for execution: {execution_id}")
    
    def record_metric(
        self, 
        agent_id: str, 
        agent_type: str,
        metric_type: MetricType,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a custom metric.
        
        Args:
            agent_id: Agent identifier
            agent_type: Agent type
            metric_type: Type of metric
            value: Metric value
            metadata: Additional metadata
        """
        metric = AgentMetric(
            agent_id=agent_id,
            agent_type=agent_type,
            metric_type=metric_type,
            value=value,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        with self._lock:
            self._metrics.append(metric)
        
        self._logger.debug(f"Recorded metric: {metric_type.value} = {value} for {agent_id}")
    
    def get_agent_metrics(
        self, 
        agent_id: str, 
        time_window: Optional[timedelta] = None
    ) -> List[AgentMetric]:
        """
        Get metrics for a specific agent.
        
        Args:
            agent_id: Agent identifier
            time_window: Time window to filter metrics (default: all time)
            
        Returns:
            List of metrics for the agent
        """
        cutoff_time = datetime.utcnow() - time_window if time_window else None
        
        with self._lock:
            metrics = [
                m for m in self._metrics 
                if m.agent_id == agent_id and (
                    cutoff_time is None or m.timestamp >= cutoff_time
                )
            ]
        
        return metrics
    
    def get_aggregated_metrics(
        self, 
        agent_type: Optional[str] = None,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Get aggregated metrics.
        
        Args:
            agent_type: Filter by agent type
            time_window: Time window to aggregate over
            
        Returns:
            Aggregated metrics dictionary
        """
        cutoff_time = datetime.utcnow() - time_window if time_window else None
        
        with self._lock:
            relevant_metrics = [
                m for m in self._metrics 
                if (agent_type is None or m.agent_type == agent_type) and (
                    cutoff_time is None or m.timestamp >= cutoff_time
                )
            ]
        
        return self._aggregate_metrics(relevant_metrics)
    
    def get_performance_summary(self, agent_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance summary for agents.
        
        Args:
            agent_type: Filter by agent type
            
        Returns:
            Performance summary dictionary
        """
        time_windows = {
            'last_hour': timedelta(hours=1),
            'last_day': timedelta(days=1),
            'last_week': timedelta(weeks=1)
        }
        
        summary = {}
        for window_name, window in time_windows.items():
            summary[window_name] = self.get_aggregated_metrics(agent_type, window)
        
        summary['all_time'] = self.get_aggregated_metrics(agent_type)
        
        return summary
    
    def _create_execution_metrics(self, execution: ExecutionMetrics) -> None:
        """Create individual metrics from execution data."""
        base_metadata = {
            'task_id': execution.task_id,
            'retry_count': execution.retry_count
        }
        
        # Execution time metric
        self._metrics.append(AgentMetric(
            agent_id=execution.agent_id,
            agent_type=execution.agent_type,
            metric_type=MetricType.EXECUTION_TIME,
            value=execution.execution_time,
            timestamp=execution.end_time,
            metadata=base_metadata
        ))
        
        # Success/failure metric
        self._metrics.append(AgentMetric(
            agent_id=execution.agent_id,
            agent_type=execution.agent_type,
            metric_type=MetricType.SUCCESS_RATE,
            value=1.0 if execution.success else 0.0,
            timestamp=execution.end_time,
            metadata={**base_metadata, 'error_message': execution.error_message}
        ))
        
        # Retry count metric
        if execution.retry_count > 0:
            self._metrics.append(AgentMetric(
                agent_id=execution.agent_id,
                agent_type=execution.agent_type,
                metric_type=MetricType.RETRY_COUNT,
                value=float(execution.retry_count),
                timestamp=execution.end_time,
                metadata=base_metadata
            ))
        
        # Token usage metric
        if execution.tokens_used > 0:
            self._metrics.append(AgentMetric(
                agent_id=execution.agent_id,
                agent_type=execution.agent_type,
                metric_type=MetricType.TOKEN_USAGE,
                value=float(execution.tokens_used),
                timestamp=execution.end_time,
                metadata=base_metadata
            ))
        
        # Memory usage metric
        if execution.memory_peak > 0:
            self._metrics.append(AgentMetric(
                agent_id=execution.agent_id,
                agent_type=execution.agent_type,
                metric_type=MetricType.MEMORY_USAGE,
                value=execution.memory_peak,
                timestamp=execution.end_time,
                metadata=base_metadata
            ))
        
        # Tool usage metrics
        for tool in execution.tools_used:
            self._metrics.append(AgentMetric(
                agent_id=execution.agent_id,
                agent_type=execution.agent_type,
                metric_type=MetricType.TOOL_USAGE,
                value=1.0,
                timestamp=execution.end_time,
                metadata={**base_metadata, 'tool_name': tool}
            ))
    
    def _aggregate_metrics(self, metrics: List[AgentMetric]) -> Dict[str, Any]:
        """Aggregate a list of metrics."""
        if not metrics:
            return {}
        
        # Group by metric type
        grouped = defaultdict(list)
        for metric in metrics:
            grouped[metric.metric_type].append(metric.value)
        
        aggregated = {}
        
        for metric_type, values in grouped.items():
            if metric_type == MetricType.EXECUTION_TIME:
                aggregated['avg_execution_time'] = sum(values) / len(values)
                aggregated['min_execution_time'] = min(values)
                aggregated['max_execution_time'] = max(values)
                aggregated['total_executions'] = len(values)
                
            elif metric_type == MetricType.SUCCESS_RATE:
                aggregated['success_rate'] = sum(values) / len(values)
                aggregated['success_count'] = int(sum(values))
                aggregated['failure_count'] = len(values) - int(sum(values))
                
            elif metric_type == MetricType.RETRY_COUNT:
                aggregated['avg_retries'] = sum(values) / len(values)
                aggregated['total_retries'] = int(sum(values))
                
            elif metric_type == MetricType.TOKEN_USAGE:
                aggregated['avg_tokens'] = sum(values) / len(values)
                aggregated['total_tokens'] = int(sum(values))
                
            elif metric_type == MetricType.MEMORY_USAGE:
                aggregated['avg_memory'] = sum(values) / len(values)
                aggregated['peak_memory'] = max(values)
        
        return aggregated


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def initialize_metrics(max_history: int = 10000) -> MetricsCollector:
    """Initialize the global metrics collector."""
    global _metrics_collector
    _metrics_collector = MetricsCollector(max_history)
    return _metrics_collector