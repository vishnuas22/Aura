"""
Research cache management for improved performance.

Provides intelligent caching of research results with:
- TTL-based expiration
- Memory usage optimization
- Cache hit/miss tracking
- Query similarity detection
- Persistent storage options
"""

import asyncio
import hashlib
import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict
from pathlib import Path

from cachetools import TTLCache, LRUCache
from .base_research_tool import ResearchResult, SearchQuery


class ResearchCache:
    """
    Intelligent cache for research results.
    
    Features:
    - TTL-based expiration
    - LRU eviction policy
    - Query similarity detection
    - Persistent storage
    - Performance metrics
    """
    
    def __init__(self,
                 max_size: int = 1000,
                 default_ttl: int = 3600,
                 persistent_storage: bool = False,
                 storage_path: Optional[Path] = None):
        """
        Initialize research cache.
        
        Args:
            max_size: Maximum number of cached entries
            default_ttl: Default TTL in seconds
            persistent_storage: Whether to use persistent storage
            storage_path: Path for persistent storage
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.persistent_storage = persistent_storage
        self.storage_path = storage_path or Path("/tmp/research_cache")
        
        # Initialize cache
        self._cache = TTLCache(maxsize=max_size, ttl=default_ttl)
        self._query_cache = LRUCache(maxsize=max_size // 10)  # For query similarity
        
        # Performance metrics
        self._hits = 0
        self._misses = 0
        self._total_queries = 0
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        self.logger = logging.getLogger(__name__)
        
        # Load persistent cache if enabled
        if self.persistent_storage:
            self._load_persistent_cache()
    
    async def get(self, 
                  query: Union[str, SearchQuery], 
                  tool_name: str,
                  similarity_threshold: float = 0.8) -> Optional[List[ResearchResult]]:
        """
        Get cached results for a query.
        
        Args:
            query: Search query
            tool_name: Name of the research tool
            similarity_threshold: Threshold for query similarity matching
            
        Returns:
            Cached results if found, None otherwise
        """
        async with self._lock:
            self._total_queries += 1
            
            # Generate cache key
            cache_key = self._generate_cache_key(query, tool_name)
            
            # Check exact match first
            if cache_key in self._cache:
                self._hits += 1
                self.logger.debug(f"Cache hit for key: {cache_key}")
                return self._deserialize_results(self._cache[cache_key])
            
            # Check for similar queries
            similar_key = await self._find_similar_query(query, tool_name, similarity_threshold)
            if similar_key and similar_key in self._cache:
                self._hits += 1
                self.logger.debug(f"Cache hit for similar query: {similar_key}")
                return self._deserialize_results(self._cache[similar_key])
            
            self._misses += 1
            return None
    
    async def set(self,
                  query: Union[str, SearchQuery],
                  tool_name: str,
                  results: List[ResearchResult],
                  ttl: Optional[int] = None) -> None:
        """
        Cache results for a query.
        
        Args:
            query: Search query
            tool_name: Name of the research tool
            results: Research results to cache
            ttl: Time to live in seconds (uses default if None)
        """
        async with self._lock:
            # Generate cache key
            cache_key = self._generate_cache_key(query, tool_name)
            
            # Serialize results
            serialized_results = self._serialize_results(results)
            
            # Set TTL
            cache_ttl = ttl or self.default_ttl
            
            # Store in cache
            self._cache[cache_key] = serialized_results
            
            # Store query for similarity matching
            query_str = query if isinstance(query, str) else query.query
            self._query_cache[cache_key] = {
                'query': query_str,
                'tool_name': tool_name,
                'timestamp': datetime.now(),
                'result_count': len(results)
            }
            
            self.logger.debug(f"Cached {len(results)} results for key: {cache_key}")
            
            # Save to persistent storage if enabled
            if self.persistent_storage:
                await self._save_to_persistent_storage(cache_key, serialized_results, cache_ttl)
    
    def _generate_cache_key(self, query: Union[str, SearchQuery], tool_name: str) -> str:
        """Generate a unique cache key for the query."""
        if isinstance(query, SearchQuery):
            # Include all relevant query parameters
            key_data = {
                'query': query.query,
                'max_results': query.max_results,
                'language': query.language,
                'date_range': query.date_range,
                'sort_by': query.sort_by,
                'filters': query.filters,
                'tool_name': tool_name
            }
        else:
            key_data = {
                'query': query,
                'tool_name': tool_name
            }
        
        # Create hash from key data
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def _find_similar_query(self, 
                                  query: Union[str, SearchQuery], 
                                  tool_name: str,
                                  threshold: float) -> Optional[str]:
        """Find similar cached query using text similarity."""
        query_str = query if isinstance(query, str) else query.query
        query_words = set(query_str.lower().split())
        
        best_similarity = 0.0
        best_key = None
        
        for cache_key, cache_info in self._query_cache.items():
            if cache_info['tool_name'] != tool_name:
                continue
            
            cached_query = cache_info['query']
            cached_words = set(cached_query.lower().split())
            
            # Calculate Jaccard similarity
            if not query_words and not cached_words:
                similarity = 1.0
            elif not query_words or not cached_words:
                similarity = 0.0
            else:
                intersection = len(query_words & cached_words)
                union = len(query_words | cached_words)
                similarity = intersection / union
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_key = cache_key
        
        return best_key
    
    def _serialize_results(self, results: List[ResearchResult]) -> bytes:
        """Serialize research results for caching."""
        try:
            # Convert to serializable format
            serializable_results = []
            for result in results:
                result_dict = {
                    'content': result.content,
                    'metadata': asdict(result.metadata),
                    'summary': result.summary,
                    'key_points': result.key_points,
                    'confidence': result.confidence,
                    'raw_data': result.raw_data
                }
                serializable_results.append(result_dict)
            
            return pickle.dumps(serializable_results)
        
        except Exception as e:
            self.logger.error(f"Failed to serialize results: {str(e)}")
            return b''
    
    def _deserialize_results(self, serialized_data: bytes) -> List[ResearchResult]:
        """Deserialize research results from cache."""
        try:
            result_dicts = pickle.loads(serialized_data)
            
            results = []
            for result_dict in result_dicts:
                # Reconstruct metadata
                from .base_research_tool import SourceMetadata
                metadata = SourceMetadata(**result_dict['metadata'])
                
                # Reconstruct result
                result = ResearchResult(
                    content=result_dict['content'],
                    metadata=metadata,
                    summary=result_dict['summary'],
                    key_points=result_dict['key_points'],
                    confidence=result_dict['confidence'],
                    raw_data=result_dict['raw_data']
                )
                results.append(result)
            
            return results
        
        except Exception as e:
            self.logger.error(f"Failed to deserialize results: {str(e)}")
            return []
    
    async def _save_to_persistent_storage(self, cache_key: str, data: bytes, ttl: int):
        """Save cache entry to persistent storage."""
        if not self.persistent_storage:
            return
        
        try:
            # Create storage directory if it doesn't exist
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # Prepare cache entry
            cache_entry = {
                'data': data,
                'expires_at': (datetime.now() + timedelta(seconds=ttl)).isoformat(),
                'created_at': datetime.now().isoformat()
            }
            
            # Save to file
            cache_file = self.storage_path / f"{cache_key}.cache"
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_entry, f)
        
        except Exception as e:
            self.logger.error(f"Failed to save to persistent storage: {str(e)}")
    
    def _load_persistent_cache(self):
        """Load cache entries from persistent storage."""
        if not self.persistent_storage or not self.storage_path.exists():
            return
        
        try:
            now = datetime.now()
            loaded_count = 0
            
            for cache_file in self.storage_path.glob("*.cache"):
                try:
                    with open(cache_file, 'rb') as f:
                        cache_entry = pickle.load(f)
                    
                    # Check if entry has expired
                    expires_at = datetime.fromisoformat(cache_entry['expires_at'])
                    if expires_at > now:
                        cache_key = cache_file.stem
                        self._cache[cache_key] = cache_entry['data']
                        loaded_count += 1
                    else:
                        # Remove expired file
                        cache_file.unlink()
                
                except Exception as e:
                    self.logger.warning(f"Failed to load cache file {cache_file}: {str(e)}")
                    continue
            
            if loaded_count > 0:
                self.logger.info(f"Loaded {loaded_count} entries from persistent cache")
        
        except Exception as e:
            self.logger.error(f"Failed to load persistent cache: {str(e)}")
    
    async def invalidate(self, query: Union[str, SearchQuery], tool_name: str) -> bool:
        """
        Invalidate cached results for a query.
        
        Args:
            query: Search query
            tool_name: Name of the research tool
            
        Returns:
            True if entry was found and invalidated
        """
        async with self._lock:
            cache_key = self._generate_cache_key(query, tool_name)
            
            if cache_key in self._cache:
                del self._cache[cache_key]
                
                if cache_key in self._query_cache:
                    del self._query_cache[cache_key]
                
                # Remove from persistent storage
                if self.persistent_storage:
                    cache_file = self.storage_path / f"{cache_key}.cache"
                    if cache_file.exists():
                        cache_file.unlink()
                
                self.logger.debug(f"Invalidated cache entry: {cache_key}")
                return True
            
            return False
    
    async def clear(self, tool_name: Optional[str] = None):
        """
        Clear cache entries.
        
        Args:
            tool_name: If specified, only clear entries for this tool
        """
        async with self._lock:
            if tool_name is None:
                # Clear all entries
                self._cache.clear()
                self._query_cache.clear()
                
                # Clear persistent storage
                if self.persistent_storage and self.storage_path.exists():
                    for cache_file in self.storage_path.glob("*.cache"):
                        cache_file.unlink()
                
                self.logger.info("Cleared all cache entries")
            else:
                # Clear entries for specific tool
                keys_to_remove = []
                for cache_key, cache_info in self._query_cache.items():
                    if cache_info['tool_name'] == tool_name:
                        keys_to_remove.append(cache_key)
                
                for cache_key in keys_to_remove:
                    if cache_key in self._cache:
                        del self._cache[cache_key]
                    if cache_key in self._query_cache:
                        del self._query_cache[cache_key]
                    
                    # Remove from persistent storage
                    if self.persistent_storage:
                        cache_file = self.storage_path / f"{cache_key}.cache"
                        if cache_file.exists():
                            cache_file.unlink()
                
                self.logger.info(f"Cleared {len(keys_to_remove)} cache entries for {tool_name}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        hit_rate = self._hits / self._total_queries if self._total_queries > 0 else 0.0
        
        return {
            'cache_size': len(self._cache),
            'max_size': self.max_size,
            'total_queries': self._total_queries,
            'cache_hits': self._hits,
            'cache_misses': self._misses,
            'hit_rate': hit_rate,
            'query_cache_size': len(self._query_cache),
            'persistent_storage': self.persistent_storage,
            'storage_path': str(self.storage_path) if self.persistent_storage else None
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self._hits = 0
        self._misses = 0
        self._total_queries = 0
        self.logger.info("Cache statistics reset")


# Global cache instance
_global_cache: Optional[ResearchCache] = None


def get_research_cache() -> ResearchCache:
    """Get the global research cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = ResearchCache()
    return _global_cache


def initialize_research_cache(**kwargs) -> ResearchCache:
    """Initialize the global research cache with custom settings."""
    global _global_cache
    _global_cache = ResearchCache(**kwargs)
    return _global_cache