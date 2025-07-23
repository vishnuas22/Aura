"""
Retry logic implementation with exponential backoff.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Optional, Type, Union, List
from functools import wraps

from core.exceptions import AgentRetryExhaustedException


class RetryHandler:
    """
    Handles retry logic with exponential backoff.
    
    Features:
    - Configurable max retries
    - Exponential backoff with jitter
    - Selective exception handling
    - Retry callbacks
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        max_wait_time: float = 60.0,
        initial_wait_time: float = 1.0,
        jitter: bool = True,
        retryable_exceptions: Optional[List[Type[Exception]]] = None
    ):
        """
        Initialize retry handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            backoff_factor: Multiplier for wait time between retries
            max_wait_time: Maximum wait time between retries
            initial_wait_time: Initial wait time before first retry
            jitter: Whether to add random jitter to wait times
            retryable_exceptions: List of exceptions that should trigger retries
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_wait_time = max_wait_time
        self.initial_wait_time = initial_wait_time
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions
        self.logger = logging.getLogger(__name__)
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        on_retry: Optional[Callable[[int, Exception], None]] = None,
        **kwargs
    ) -> Any:
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute (can be sync or async)
            *args: Function arguments
            on_retry: Callback function called on each retry
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            AgentRetryExhaustedException: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):  # +1 for initial attempt
            try:
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success - return result
                if attempt > 0:
                    self.logger.info(f"Function succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if this exception should trigger a retry
                if not self._should_retry(e):
                    self.logger.debug(f"Exception {type(e).__name__} is not retryable")
                    raise e
                
                # Check if we have retries left
                if attempt >= self.max_retries:
                    break
                
                # Calculate wait time
                wait_time = self._calculate_wait_time(attempt)
                
                self.logger.warning(
                    f"Attempt {attempt + 1} failed with {type(e).__name__}: {str(e)}. "
                    f"Retrying in {wait_time:.2f} seconds..."
                )
                
                # Call retry callback if provided
                if on_retry:
                    try:
                        on_retry(attempt + 1, e)
                    except Exception as callback_error:
                        self.logger.error(f"Retry callback failed: {callback_error}")
                
                # Wait before retry
                await asyncio.sleep(wait_time)
        
        # All retries exhausted
        raise AgentRetryExhaustedException(
            f"Function failed after {self.max_retries + 1} attempts. "
            f"Last error: {str(last_exception)}",
            error_code="RETRY_EXHAUSTED",
            context={
                "max_retries": self.max_retries,
                "last_exception_type": type(last_exception).__name__,
                "last_exception_message": str(last_exception)
            }
        )
    
    def _should_retry(self, exception: Exception) -> bool:
        """
        Check if an exception should trigger a retry.
        
        Args:
            exception: Exception to check
            
        Returns:
            True if should retry, False otherwise
        """
        if self.retryable_exceptions is None:
            # By default, retry on most exceptions except critical ones
            non_retryable = (KeyboardInterrupt, SystemExit, MemoryError)
            return not isinstance(exception, non_retryable)
        
        # Check if exception type is in retryable list
        return any(isinstance(exception, exc_type) for exc_type in self.retryable_exceptions)
    
    def _calculate_wait_time(self, attempt: int) -> float:
        """
        Calculate wait time for retry attempt.
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            Wait time in seconds
        """
        # Exponential backoff
        wait_time = self.initial_wait_time * (self.backoff_factor ** attempt)
        
        # Apply maximum wait time
        wait_time = min(wait_time, self.max_wait_time)
        
        # Add jitter to avoid thundering herd
        if self.jitter:
            import random
            jitter_amount = wait_time * 0.1  # 10% jitter
            wait_time += random.uniform(-jitter_amount, jitter_amount)
            wait_time = max(0.1, wait_time)  # Minimum 0.1 seconds
        
        return wait_time


def retry(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    max_wait_time: float = 60.0,
    initial_wait_time: float = 1.0,
    retryable_exceptions: Optional[List[Type[Exception]]] = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """
    Decorator for adding retry logic to functions.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for wait time between retries
        max_wait_time: Maximum wait time between retries
        initial_wait_time: Initial wait time before first retry
        retryable_exceptions: List of exceptions that should trigger retries
        on_retry: Callback function called on each retry
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            retry_handler = RetryHandler(
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                max_wait_time=max_wait_time,
                initial_wait_time=initial_wait_time,
                retryable_exceptions=retryable_exceptions
            )
            
            return await retry_handler.execute_with_retry(
                func, *args, on_retry=on_retry, **kwargs
            )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we need to run in an event loop
            async def async_func():
                return func(*args, **kwargs)
            
            retry_handler = RetryHandler(
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                max_wait_time=max_wait_time,
                initial_wait_time=initial_wait_time,
                retryable_exceptions=retryable_exceptions
            )
            
            return asyncio.run(retry_handler.execute_with_retry(
                async_func, on_retry=on_retry
            ))
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Convenience retry decorators with common configurations
def retry_on_failure(max_retries: int = 3):
    """Simple retry decorator for general failures."""
    return retry(max_retries=max_retries, initial_wait_time=0.5, backoff_factor=1.5)


def retry_on_timeout(max_retries: int = 2):
    """Retry decorator specifically for timeout errors."""
    return retry(
        max_retries=max_retries,
        initial_wait_time=2.0,
        backoff_factor=2.0,
        retryable_exceptions=[asyncio.TimeoutError, TimeoutError]
    )


def retry_on_network_error(max_retries: int = 5):
    """Retry decorator for network-related errors."""
    import aiohttp
    import requests
    
    network_exceptions = [
        aiohttp.ClientError,
        requests.RequestException,
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError
    ]
    
    return retry(
        max_retries=max_retries,
        initial_wait_time=1.0,
        backoff_factor=2.0,
        max_wait_time=30.0,
        retryable_exceptions=network_exceptions
    )