"""
Groq LLM client with advanced features for production use.
"""

import asyncio
import logging
import time
import hashlib
import json
from typing import Dict, Any, List, Optional, Union, AsyncIterator, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import tiktoken
from groq import AsyncGroq
from groq.types.chat import ChatCompletion

from ..core.exceptions import ModelException
from ..utils.retry import RetryHandler


class GroqModel(Enum):
    """Available Groq models with their specifications."""
    
    # Fast models for quick tasks
    LLAMA_3_1_8B_INSTANT = "llama-3.1-8b-instant"
    GEMMA_2_9B_IT = "gemma2-9b-it"
    
    # Versatile models for complex tasks
    LLAMA_3_1_70B_VERSATILE = "llama-3.1-70b-versatile"
    MIXTRAL_8X7B_32768 = "mixtral-8x7b-32768"
    
    # Reasoning model for complex analysis (limited access)
    LLAMA_3_1_405B_REASONING = "llama-3.1-405b-reasoning"


@dataclass
class ModelSpec:
    """Model specifications and capabilities."""
    name: str
    context_window: int
    daily_token_limit: int
    cost_per_token: float  # Placeholder for future cost tracking
    recommended_use: List[str]
    temperature_range: Tuple[float, float] = (0.0, 2.0)
    max_tokens_limit: int = 4096


# Model specifications
MODEL_SPECS: Dict[GroqModel, ModelSpec] = {
    GroqModel.LLAMA_3_1_8B_INSTANT: ModelSpec(
        name="Llama 3.1 8B Instant",
        context_window=131072,
        daily_token_limit=500000,
        cost_per_token=0.0,  # Free
        recommended_use=["quick_tasks", "simple_analysis", "fast_responses"],
        max_tokens_limit=8192
    ),
    GroqModel.LLAMA_3_1_70B_VERSATILE: ModelSpec(
        name="Llama 3.1 70B Versatile", 
        context_window=131072,
        daily_token_limit=200000,
        cost_per_token=0.0,  # Free
        recommended_use=["complex_reasoning", "detailed_analysis", "comprehensive_research"],
        max_tokens_limit=8192
    ),
    GroqModel.MIXTRAL_8X7B_32768: ModelSpec(
        name="Mixtral 8x7B",
        context_window=32768,
        daily_token_limit=500000,
        cost_per_token=0.0,  # Free
        recommended_use=["balanced_performance", "code_generation", "structured_output"],
        max_tokens_limit=4096
    ),
    GroqModel.GEMMA_2_9B_IT: ModelSpec(
        name="Gemma 2 9B IT",
        context_window=8192,
        daily_token_limit=500000,
        cost_per_token=0.0,  # Free
        recommended_use=["instruction_following", "chat", "simple_tasks"],
        max_tokens_limit=2048
    ),
    GroqModel.LLAMA_3_1_405B_REASONING: ModelSpec(
        name="Llama 3.1 405B Reasoning",
        context_window=131072,
        daily_token_limit=10000,  # Limited access
        cost_per_token=0.0,  # Free but limited
        recommended_use=["complex_reasoning", "advanced_analysis", "research"],
        max_tokens_limit=8192
    )
}


@dataclass
class TokenUsage:
    """Token usage tracking."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    
    def add_usage(self, other: 'TokenUsage') -> None:
        """Add another token usage to this one."""
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.estimated_cost += other.estimated_cost


@dataclass
class ModelResponse:
    """Standardized model response."""
    content: str
    model: str
    usage: TokenUsage
    response_time: float
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResponseCache:
    """Simple in-memory response cache with TTL."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize response cache.
        
        Args:
            max_size: Maximum number of cached responses
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self.logger = logging.getLogger(__name__)
    
    def _generate_key(self, messages: List[Dict[str, str]], model: str, **kwargs) -> str:
        """Generate cache key from request parameters."""
        # Include relevant parameters in cache key
        cache_data = {
            "messages": messages,
            "model": model,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000),
            "top_p": kwargs.get("top_p", 1.0)
        }
        
        # Create hash of parameters
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def get(self, messages: List[Dict[str, str]], model: str, **kwargs) -> Optional[ModelResponse]:
        """Get cached response if available and not expired."""
        key = self._generate_key(messages, model, **kwargs)
        
        if key not in self._cache:
            return None
        
        cached_item = self._cache[key]
        
        # Check if expired
        if time.time() > cached_item["expires_at"]:
            self._remove(key)
            return None
        
        # Update access time
        self._access_times[key] = time.time()
        
        # Return cached response
        response = cached_item["response"]
        response.cached = True
        
        self.logger.debug(f"Cache hit for key: {key[:8]}...")
        return response
    
    def put(self, messages: List[Dict[str, str]], model: str, response: ModelResponse, ttl: Optional[int] = None, **kwargs) -> None:
        """Cache response with TTL."""
        key = self._generate_key(messages, model, **kwargs)
        ttl = ttl or self.default_ttl
        
        # Clean cache if full
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        
        # Store response
        self._cache[key] = {
            "response": response,
            "expires_at": time.time() + ttl,
            "created_at": time.time()
        }
        
        self._access_times[key] = time.time()
        self.logger.debug(f"Cached response for key: {key[:8]}...")
    
    def _remove(self, key: str) -> None:
        """Remove item from cache."""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
    
    def _evict_oldest(self) -> None:
        """Evict least recently used item."""
        if not self._access_times:
            return
        
        oldest_key = min(self._access_times.items(), key=lambda x: x[1])[0]
        self._remove(oldest_key)
    
    def clear(self) -> None:
        """Clear all cached responses."""
        self._cache.clear()
        self._access_times.clear()
        self.logger.info("Response cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = time.time()
        expired_count = sum(
            1 for item in self._cache.values() 
            if now > item["expires_at"]
        )
        
        return {
            "total_items": len(self._cache),
            "expired_items": expired_count,
            "active_items": len(self._cache) - expired_count,
            "max_size": self.max_size,
            "cache_utilization": len(self._cache) / self.max_size
        }


class TokenCounter:
    """Token counting utilities."""
    
    def __init__(self):
        """Initialize token counter."""
        self.logger = logging.getLogger(__name__)
        self._encodings = {}
    
    def _get_encoding(self, model: str) -> tiktoken.Encoding:
        """Get encoding for model (with caching)."""
        if model not in self._encodings:
            try:
                # Use cl100k_base encoding for most models
                self._encodings[model] = tiktoken.get_encoding("cl100k_base")
            except Exception:
                # Fallback encoding
                self._encodings[model] = tiktoken.get_encoding("gpt2")
        
        return self._encodings[model]
    
    def count_tokens(self, text: str, model: str = "llama-3.1-8b-instant") -> int:
        """Count tokens in text for given model."""
        try:
            encoding = self._get_encoding(model)
            return len(encoding.encode(text))
        except Exception as e:
            self.logger.warning(f"Token counting failed: {e}, using word-based estimate")
            # Fallback: rough estimate (1 token ≈ 0.75 words)
            return int(len(text.split()) * 1.33)
    
    def count_message_tokens(self, messages: List[Dict[str, str]], model: str = "llama-3.1-8b-instant") -> int:
        """Count tokens in a list of messages."""
        total_tokens = 0
        
        for message in messages:
            # Add tokens for message content
            total_tokens += self.count_tokens(message.get("content", ""), model)
            
            # Add overhead tokens for message structure
            total_tokens += 4  # Approximate overhead per message
        
        # Add overhead for conversation structure
        total_tokens += 2
        
        return total_tokens


class HealthChecker:
    """Health checking for Groq API."""
    
    def __init__(self, client: AsyncGroq):
        """Initialize health checker."""
        self.client = client
        self.logger = logging.getLogger(__name__)
        self._last_health_check: Optional[datetime] = None
        self._health_status = True
        self._health_check_interval = 300  # 5 minutes
    
    async def check_health(self, force: bool = False) -> bool:
        """
        Check API health.
        
        Args:
            force: Force health check even if recently checked
            
        Returns:
            True if healthy, False otherwise
        """
        now = datetime.utcnow()
        
        # Skip if recently checked
        if (not force and 
            self._last_health_check and 
            (now - self._last_health_check).total_seconds() < self._health_check_interval):
            return self._health_status
        
        try:
            # Simple test request
            test_messages = [{"role": "user", "content": "ping"}]
            
            response = await self.client.chat.completions.create(
                model=GroqModel.LLAMA_3_1_8B_INSTANT.value,
                messages=test_messages,
                max_tokens=5,
                temperature=0
            )
            
            self._health_status = bool(response and response.choices)
            self._last_health_check = now
            
            if self._health_status:
                self.logger.debug("Health check passed")
            else:
                self.logger.warning("Health check failed: No response content")
            
        except Exception as e:
            self._health_status = False
            self._last_health_check = now
            self.logger.error(f"Health check failed: {e}")
        
        return self._health_status
    
    def is_healthy(self) -> bool:
        """Get last known health status."""
        return self._health_status


class GroqLLMClient:
    """
    Production-ready Groq LLM client with advanced features.
    
    Features:
    - Connection management with health checks
    - Model switching capability
    - Token counting and cost estimation
    - Response caching for efficiency
    - Fallback mechanisms
    - Async processing
    - Retry logic with exponential backoff
    """
    
    def __init__(
        self,
        api_key: str,
        default_model: GroqModel = GroqModel.LLAMA_3_1_70B_VERSATILE,
        enable_caching: bool = True,
        cache_ttl: int = 3600,
        max_retries: int = 3,
        request_timeout: int = 300
    ):
        """
        Initialize Groq LLM client.
        
        Args:
            api_key: Groq API key
            default_model: Default model to use
            enable_caching: Whether to enable response caching
            cache_ttl: Cache time-to-live in seconds
            max_retries: Maximum retry attempts
            request_timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.default_model = default_model
        self.enable_caching = enable_caching
        self.request_timeout = request_timeout
        
        # Initialize client
        self.client = AsyncGroq(api_key=api_key)
        
        # Initialize components
        self.cache = ResponseCache(default_ttl=cache_ttl) if enable_caching else None
        self.token_counter = TokenCounter()
        self.health_checker = HealthChecker(self.client)
        
        # Retry handler
        self.retry_handler = RetryHandler(
            max_retries=max_retries,
            backoff_factor=2.0,
            max_wait_time=60.0,
            initial_wait_time=1.0
        )
        
        # Usage tracking
        self._daily_usage: Dict[str, TokenUsage] = {}
        self._usage_reset_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        
        # Model fallback chain
        self._fallback_models = [
            GroqModel.LLAMA_3_1_70B_VERSATILE,
            GroqModel.LLAMA_3_1_8B_INSTANT,
            GroqModel.MIXTRAL_8X7B_32768,
            GroqModel.GEMMA_2_9B_IT
        ]
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Groq LLM client initialized with default model: {default_model.value}")
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[GroqModel] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False,
        use_cache: bool = True,
        fallback: bool = True,
        **kwargs
    ) -> Union[ModelResponse, AsyncIterator[str]]:
        """
        Generate chat completion.
        
        Args:
            messages: List of messages
            model: Model to use (defaults to default_model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response
            use_cache: Whether to use cached responses
            fallback: Whether to use fallback models on failure
            **kwargs: Additional parameters
            
        Returns:
            ModelResponse or async iterator for streaming
        """
        model = model or self.default_model
        
        # Validate model parameters
        self._validate_parameters(model, temperature, max_tokens)
        
        # Check cache first (if not streaming and caching enabled)
        if not stream and use_cache and self.cache:
            cached_response = self.cache.get(messages, model.value, temperature=temperature, max_tokens=max_tokens, **kwargs)
            if cached_response:
                return cached_response
        
        # Check daily usage limits
        if not self._check_usage_limits(model, messages):
            if fallback:
                return await self._try_fallback_models(messages, temperature, max_tokens, stream, use_cache, **kwargs)
            else:
                raise ModelException(
                    f"Daily usage limit exceeded for {model.value}",
                    error_code="USAGE_LIMIT_EXCEEDED"
                )
        
        # Execute with retry logic
        try:
            if stream:
                return await self._stream_completion(model, messages, temperature, max_tokens, **kwargs)
            else:
                return await self.retry_handler.execute_with_retry(
                    self._single_completion,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    use_cache=use_cache,
                    fallback=fallback,
                    **kwargs
                )
        except Exception as e:
            if fallback and not stream:
                self.logger.warning(f"Model {model.value} failed, trying fallbacks: {e}")
                return await self._try_fallback_models(messages, temperature, max_tokens, stream, use_cache, **kwargs)
            raise
    
    async def _single_completion(
        self,
        model: GroqModel,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        use_cache: bool = True,
        fallback: bool = True,
        **kwargs
    ) -> ModelResponse:
        """Execute single completion request."""
        start_time = time.time()
        
        try:
            # Make API request
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=model.value,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                ),
                timeout=self.request_timeout
            )
            
            response_time = time.time() - start_time
            
            # Process response
            content = response.choices[0].message.content or ""
            
            # Track token usage
            usage = TokenUsage(
                prompt_tokens=getattr(response.usage, 'prompt_tokens', 0),
                completion_tokens=getattr(response.usage, 'completion_tokens', 0),
                total_tokens=getattr(response.usage, 'total_tokens', 0),
                estimated_cost=0.0  # Free for now
            )
            
            # Update daily usage
            self._update_usage(model, usage)
            
            # Create response object
            model_response = ModelResponse(
                content=content,
                model=model.value,
                usage=usage,
                response_time=response_time,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "response_id": response.id,
                    "created": response.created
                }
            )
            
            # Cache response
            if use_cache and self.cache:
                self.cache.put(messages, model.value, model_response, temperature=temperature, max_tokens=max_tokens, **kwargs)
            
            self.logger.debug(f"Completion successful: {model.value}, tokens: {usage.total_tokens}, time: {response_time:.2f}s")
            
            return model_response
            
        except asyncio.TimeoutError:
            raise ModelException(
                f"Request timeout for {model.value}",
                error_code="REQUEST_TIMEOUT"
            )
        except Exception as e:
            raise ModelException(
                f"Model request failed for {model.value}: {str(e)}",
                error_code="MODEL_REQUEST_FAILED"
            )
    
    async def _stream_completion(
        self,
        model: GroqModel,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream completion response."""
        try:
            stream = await self.client.chat.completions.create(
                model=model.value,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            raise ModelException(
                f"Streaming failed for {model.value}: {str(e)}",
                error_code="STREAMING_FAILED"
            )
    
    async def _try_fallback_models(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        stream: bool,
        use_cache: bool,
        **kwargs
    ) -> Union[ModelResponse, AsyncIterator[str]]:
        """Try fallback models in order."""
        last_error = None
        
        for fallback_model in self._fallback_models:
            try:
                self.logger.info(f"Trying fallback model: {fallback_model.value}")
                
                return await self.chat_completion(
                    messages=messages,
                    model=fallback_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                    use_cache=use_cache,
                    fallback=False,  # Prevent infinite recursion
                    **kwargs
                )
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Fallback model {fallback_model.value} failed: {e}")
                continue
        
        # All fallbacks failed
        raise ModelException(
            f"All fallback models failed. Last error: {str(last_error)}",
            error_code="ALL_FALLBACKS_FAILED"
        )
    
    def _validate_parameters(self, model: GroqModel, temperature: float, max_tokens: int) -> None:
        """Validate request parameters."""
        model_spec = MODEL_SPECS[model]
        
        # Validate temperature
        temp_min, temp_max = model_spec.temperature_range
        if not (temp_min <= temperature <= temp_max):
            raise ModelException(
                f"Temperature {temperature} out of range [{temp_min}, {temp_max}] for {model.value}",
                error_code="INVALID_TEMPERATURE"
            )
        
        # Validate max_tokens
        if max_tokens > model_spec.max_tokens_limit:
            raise ModelException(
                f"Max tokens {max_tokens} exceeds limit {model_spec.max_tokens_limit} for {model.value}",
                error_code="INVALID_MAX_TOKENS"
            )
        
        if max_tokens < 1:
            raise ModelException(
                "Max tokens must be positive",
                error_code="INVALID_MAX_TOKENS"
            )
    
    def _check_usage_limits(self, model: GroqModel, messages: List[Dict[str, str]]) -> bool:
        """Check if request would exceed daily usage limits."""
        # Reset usage if new day
        now = datetime.utcnow()
        if now >= self._usage_reset_time:
            self._daily_usage.clear()
            self._usage_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        
        # Get current usage for model
        current_usage = self._daily_usage.get(model.value, TokenUsage())
        
        # Estimate tokens for this request
        estimated_tokens = self.token_counter.count_message_tokens(messages, model.value)
        
        # Check against daily limit
        model_spec = MODEL_SPECS[model]
        if current_usage.total_tokens + estimated_tokens > model_spec.daily_token_limit:
            return False
        
        return True
    
    def _update_usage(self, model: GroqModel, usage: TokenUsage) -> None:
        """Update daily usage tracking."""
        if model.value not in self._daily_usage:
            self._daily_usage[model.value] = TokenUsage()
        
        self._daily_usage[model.value].add_usage(usage)
    
    def get_model_recommendation(self, task_type: str, complexity: str = "medium") -> GroqModel:
        """
        Get recommended model for task type and complexity.
        
        Args:
            task_type: Type of task (research, analysis, writing, etc.)
            complexity: Task complexity (low, medium, high)
            
        Returns:
            Recommended model
        """
        recommendations = {
            ("research", "low"): GroqModel.LLAMA_3_1_8B_INSTANT,
            ("research", "medium"): GroqModel.LLAMA_3_1_70B_VERSATILE,
            ("research", "high"): GroqModel.LLAMA_3_1_405B_REASONING,
            
            ("analysis", "low"): GroqModel.MIXTRAL_8X7B_32768,
            ("analysis", "medium"): GroqModel.LLAMA_3_1_70B_VERSATILE,
            ("analysis", "high"): GroqModel.LLAMA_3_1_405B_REASONING,
            
            ("writing", "low"): GroqModel.GEMMA_2_9B_IT,
            ("writing", "medium"): GroqModel.LLAMA_3_1_70B_VERSATILE,
            ("writing", "high"): GroqModel.LLAMA_3_1_70B_VERSATILE,
            
            ("quick", "any"): GroqModel.LLAMA_3_1_8B_INSTANT,
            ("balanced", "any"): GroqModel.LLAMA_3_1_70B_VERSATILE,
            ("reasoning", "any"): GroqModel.LLAMA_3_1_405B_REASONING,
        }
        
        return recommendations.get(
            (task_type.lower(), complexity.lower()),
            recommendations.get((task_type.lower(), "medium"), self.default_model)
        )
    
    async def health_check(self, force: bool = False) -> bool:
        """Check API health."""
        return await self.health_checker.check_health(force=force)
    
    def is_healthy(self) -> bool:
        """Get last known health status."""
        return self.health_checker.is_healthy()
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return {
            "daily_usage": {
                model: {
                    "total_tokens": usage.total_tokens,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "estimated_cost": usage.estimated_cost
                }
                for model, usage in self._daily_usage.items()
            },
            "usage_reset_time": self._usage_reset_time.isoformat(),
            "cache_stats": self.cache.get_stats() if self.cache else None,
            "health_status": self.is_healthy()
        }
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available models."""
        return {
            model.value: {
                "name": spec.name,
                "context_window": spec.context_window,
                "daily_token_limit": spec.daily_token_limit,
                "recommended_use": spec.recommended_use,
                "max_tokens_limit": spec.max_tokens_limit
            }
            for model, spec in MODEL_SPECS.items()
        }
    
    async def close(self) -> None:
        """Close the client and cleanup resources."""
        if hasattr(self.client, 'close'):
            await self.client.close()
        
        if self.cache:
            self.cache.clear()
        
        self.logger.info("Groq LLM client closed")


# Global client instance
_global_client: Optional[GroqLLMClient] = None


def get_llm_client() -> Optional[GroqLLMClient]:
    """Get the global LLM client instance."""
    return _global_client


def initialize_llm_client(
    api_key: str,
    default_model: GroqModel = GroqModel.LLAMA_3_1_70B_VERSATILE,
    **kwargs
) -> GroqLLMClient:
    """Initialize the global LLM client."""
    global _global_client
    _global_client = GroqLLMClient(api_key=api_key, default_model=default_model, **kwargs)
    return _global_client