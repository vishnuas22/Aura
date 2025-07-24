"""
Base research tool class with common functionality.
"""

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from urllib.parse import urlparse

from pydantic import BaseModel, Field


@dataclass
class SourceMetadata:
    """Metadata about a research source."""
    
    url: str
    title: str
    source_type: str  # 'web', 'academic', 'news', 'social', 'reference'
    domain: str
    author: Optional[str] = None
    publish_date: Optional[datetime] = None
    accessed_date: Optional[datetime] = None
    language: str = 'en'
    credibility_score: float = 0.0
    relevance_score: float = 0.0
    
    def __post_init__(self):
        """Set accessed date if not provided."""
        if self.accessed_date is None:
            self.accessed_date = datetime.now(timezone.utc)
        
        # Extract domain from URL if not provided
        if not self.domain and self.url:
            parsed_url = urlparse(self.url)
            self.domain = parsed_url.netloc.lower()


@dataclass 
class ResearchResult:
    """Result from a research tool."""
    
    content: str
    metadata: SourceMetadata
    raw_data: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None
    key_points: List[str] = None
    confidence: float = 0.0
    
    def __post_init__(self):
        """Initialize empty lists if None."""
        if self.key_points is None:
            self.key_points = []


class SearchQuery(BaseModel):
    """Structured search query."""
    
    query: str = Field(..., description="The search query")
    max_results: int = Field(default=10, ge=1, le=100)
    language: str = Field(default='en')
    date_range: Optional[str] = Field(default=None, description="Date range filter")
    sort_by: str = Field(default='relevance', description="Sort criteria")
    filters: Dict[str, Any] = Field(default_factory=dict)


class BaseResearchTool(ABC):
    """
    Abstract base class for all research tools.
    
    Provides common functionality including:
    - Rate limiting
    - Caching 
    - Error handling
    - Content validation
    - Relevance scoring
    """
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 source_type: str,
                 max_requests_per_minute: int = 30):
        """
        Initialize base research tool.
        
        Args:
            name: Tool name
            description: Tool description
            source_type: Type of source this tool searches
            max_requests_per_minute: Rate limit for requests
        """
        self.name = name
        self.description = description
        self.source_type = source_type
        self.max_requests_per_minute = max_requests_per_minute
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Initialize rate limiting
        self._request_times: List[datetime] = []
        self._rate_limit_lock = asyncio.Lock()
        
        # Cache for results
        self._cache: Dict[str, List[ResearchResult]] = {}
        self._cache_ttl = 3600  # 1 hour
        self._cache_timestamps: Dict[str, datetime] = {}
        
        self.logger.info(f"Initialized {name} research tool")
    
    async def search(self, 
                    query: Union[str, SearchQuery], 
                    use_cache: bool = True) -> List[ResearchResult]:
        """
        Main search method with rate limiting and caching.
        
        Args:
            query: Search query (string or SearchQuery object)
            use_cache: Whether to use cached results
            
        Returns:
            List of research results
            
        Raises:
            Exception: If search fails after retries
        """
        # Convert string to SearchQuery if needed
        if isinstance(query, str):
            search_query = SearchQuery(query=query)
        else:
            search_query = query
        
        # Generate cache key
        cache_key = self._generate_cache_key(search_query)
        
        # Check cache first
        if use_cache and self._is_cached(cache_key):
            self.logger.debug(f"Returning cached results for: {search_query.query}")
            return self._get_cached_results(cache_key)
        
        # Apply rate limiting
        await self._apply_rate_limit()
        
        try:
            # Perform actual search
            results = await self._perform_search(search_query)
            
            # Validate and process results
            validated_results = []
            for result in results:
                if self._validate_result(result):
                    # Calculate relevance score
                    result.metadata.relevance_score = self._calculate_relevance_score(
                        search_query.query, result
                    )
                    validated_results.append(result)
            
            # Cache results
            if use_cache:
                self._cache_results(cache_key, validated_results)
            
            self.logger.info(f"Found {len(validated_results)} results for: {search_query.query}")
            return validated_results
            
        except Exception as e:
            self.logger.error(f"Search failed for {search_query.query}: {str(e)}")
            raise
    
    @abstractmethod
    async def _perform_search(self, query: SearchQuery) -> List[ResearchResult]:
        """
        Perform the actual search. Must be implemented by subclasses.
        
        Args:
            query: The search query
            
        Returns:
            List of research results
        """
        pass
    
    async def _apply_rate_limit(self):
        """Apply rate limiting to prevent API abuse."""
        async with self._rate_limit_lock:
            now = datetime.now()
            
            # Remove old timestamps (older than 1 minute)
            cutoff = now.timestamp() - 60
            self._request_times = [
                req_time for req_time in self._request_times 
                if req_time.timestamp() > cutoff
            ]
            
            # Check if we've exceeded rate limit
            if len(self._request_times) >= self.max_requests_per_minute:
                sleep_time = 60 - (now.timestamp() - self._request_times[0].timestamp())
                if sleep_time > 0:
                    self.logger.warning(f"Rate limit reached, sleeping for {sleep_time:.1f} seconds")
                    await asyncio.sleep(sleep_time)
            
            # Record this request
            self._request_times.append(now)
    
    def _generate_cache_key(self, query: SearchQuery) -> str:
        """Generate a cache key for the query."""
        query_str = f"{query.query}_{query.max_results}_{query.language}_{query.sort_by}"
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if results are cached and still valid."""
        if cache_key not in self._cache:
            return False
        
        # Check if cache has expired
        cache_time = self._cache_timestamps.get(cache_key)
        if cache_time is None:
            return False
        
        age_seconds = (datetime.now() - cache_time).total_seconds()
        return age_seconds < self._cache_ttl
    
    def _get_cached_results(self, cache_key: str) -> List[ResearchResult]:
        """Get cached results."""
        return self._cache.get(cache_key, [])
    
    def _cache_results(self, cache_key: str, results: List[ResearchResult]):
        """Cache search results."""
        self._cache[cache_key] = results
        self._cache_timestamps[cache_key] = datetime.now()
        
        # Cleanup old cache entries (keep cache size reasonable)
        if len(self._cache) > 100:  # Max 100 cached queries
            oldest_key = min(self._cache_timestamps.keys(), 
                           key=lambda k: self._cache_timestamps[k])
            del self._cache[oldest_key]
            del self._cache_timestamps[oldest_key]
    
    def _validate_result(self, result: ResearchResult) -> bool:
        """
        Validate a research result.
        
        Args:
            result: The result to validate
            
        Returns:
            True if result is valid
        """
        if not result.content or not result.content.strip():
            return False
        
        if not result.metadata.url:
            return False
        
        # Basic content length check
        if len(result.content.strip()) < 50:
            return False
        
        return True
    
    def _calculate_relevance_score(self, query: str, result: ResearchResult) -> float:
        """
        Calculate relevance score for a result.
        
        Args:
            query: Original search query
            result: The research result
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        score = 0.0
        query_terms = query.lower().split()
        
        # Check title relevance (weight: 0.4)
        title_lower = result.metadata.title.lower()
        title_matches = sum(1 for term in query_terms if term in title_lower)
        title_score = min(title_matches / len(query_terms), 1.0) * 0.4
        score += title_score
        
        # Check content relevance (weight: 0.4)
        content_lower = result.content.lower()
        content_matches = sum(1 for term in query_terms if term in content_lower)
        content_score = min(content_matches / len(query_terms), 1.0) * 0.4
        score += content_score
        
        # Check source credibility (weight: 0.2)
        credibility_score = result.metadata.credibility_score * 0.2
        score += credibility_score
        
        return min(score, 1.0)
    
    def clear_cache(self):
        """Clear the results cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
        self.logger.info(f"Cache cleared for {self.name}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self._cache),
            'cache_ttl': self._cache_ttl,
            'oldest_entry': min(self._cache_timestamps.values()) if self._cache_timestamps else None,
            'newest_entry': max(self._cache_timestamps.values()) if self._cache_timestamps else None
        }
    
    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        now = datetime.now()
        recent_requests = [
            req_time for req_time in self._request_times 
            if (now - req_time).total_seconds() < 60
        ]
        
        return {
            'max_requests_per_minute': self.max_requests_per_minute,
            'recent_requests': len(recent_requests),
            'remaining_requests': max(0, self.max_requests_per_minute - len(recent_requests)),
            'reset_time': max(self._request_times).timestamp() + 60 if self._request_times else now.timestamp()
        }