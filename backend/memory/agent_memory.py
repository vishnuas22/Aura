"""
Memory management system for AI agents.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import asyncio
from motor.motor_asyncio import AsyncIOMotorCollection

from ..core.exceptions import MemoryException
from ..core.config import MemoryConfig


@dataclass
class MemoryItem:
    """Individual memory item."""
    id: str
    agent_id: str
    agent_type: str
    content: Dict[str, Any]
    timestamp: datetime
    memory_type: str  # 'conversation', 'tool_result', 'insight', etc.
    importance: float = 0.5  # 0.0 to 1.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class MemoryStore(ABC):
    """Abstract base class for memory storage backends."""
    
    @abstractmethod
    async def store(self, item: MemoryItem) -> None:
        """Store a memory item."""
        pass
    
    @abstractmethod
    async def retrieve(
        self, 
        agent_id: str, 
        limit: int = 10,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        since: Optional[datetime] = None
    ) -> List[MemoryItem]:
        """Retrieve memory items."""
        pass
    
    @abstractmethod
    async def search(self, agent_id: str, query: str, limit: int = 5) -> List[MemoryItem]:
        """Search memory items by content."""
        pass
    
    @abstractmethod
    async def delete(self, agent_id: str, item_id: str) -> bool:
        """Delete a specific memory item."""
        pass
    
    @abstractmethod
    async def cleanup(self, agent_id: str, before: datetime) -> int:
        """Cleanup old memory items."""
        pass


class MongoMemoryStore(MemoryStore):
    """MongoDB-based memory storage."""
    
    def __init__(self, collection: AsyncIOMotorCollection):
        """
        Initialize MongoDB memory store.
        
        Args:
            collection: MongoDB collection for memory storage
        """
        self.collection = collection
        self._logger = logging.getLogger(__name__)
    
    async def store(self, item: MemoryItem) -> None:
        """Store a memory item in MongoDB."""
        try:
            data = item.to_dict()
            await self.collection.insert_one(data)
            self._logger.debug(f"Stored memory item: {item.id}")
            
        except Exception as e:
            raise MemoryException(
                f"Failed to store memory item: {str(e)}",
                error_code="MEMORY_STORE_ERROR"
            )
    
    async def retrieve(
        self, 
        agent_id: str, 
        limit: int = 10,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        since: Optional[datetime] = None
    ) -> List[MemoryItem]:
        """Retrieve memory items from MongoDB."""
        try:
            query = {"agent_id": agent_id}
            
            if memory_type:
                query["memory_type"] = memory_type
            
            if tags:
                query["tags"] = {"$in": tags}
            
            if since:
                query["timestamp"] = {"$gte": since.isoformat()}
            
            cursor = self.collection.find(query).sort("timestamp", -1).limit(limit)
            documents = await cursor.to_list(length=limit)
            
            items = []
            for doc in documents:
                doc.pop('_id', None)  # Remove MongoDB ObjectId
                items.append(MemoryItem.from_dict(doc))
            
            return items
            
        except Exception as e:
            raise MemoryException(
                f"Failed to retrieve memory items: {str(e)}",
                error_code="MEMORY_RETRIEVE_ERROR"
            )
    
    async def search(self, agent_id: str, query: str, limit: int = 5) -> List[MemoryItem]:
        """Search memory items by content."""
        try:
            # Simple text search in content
            search_query = {
                "agent_id": agent_id,
                "$or": [
                    {"content": {"$regex": query, "$options": "i"}},
                    {"tags": {"$regex": query, "$options": "i"}}
                ]
            }
            
            cursor = self.collection.find(search_query).sort("importance", -1).limit(limit)
            documents = await cursor.to_list(length=limit)
            
            items = []
            for doc in documents:
                doc.pop('_id', None)
                items.append(MemoryItem.from_dict(doc))
            
            return items
            
        except Exception as e:
            raise MemoryException(
                f"Failed to search memory items: {str(e)}",
                error_code="MEMORY_SEARCH_ERROR"
            )
    
    async def delete(self, agent_id: str, item_id: str) -> bool:
        """Delete a specific memory item."""
        try:
            result = await self.collection.delete_one({
                "agent_id": agent_id,
                "id": item_id
            })
            return result.deleted_count > 0
            
        except Exception as e:
            raise MemoryException(
                f"Failed to delete memory item: {str(e)}",
                error_code="MEMORY_DELETE_ERROR"
            )
    
    async def cleanup(self, agent_id: str, before: datetime) -> int:
        """Cleanup old memory items."""
        try:
            result = await self.collection.delete_many({
                "agent_id": agent_id,
                "timestamp": {"$lt": before.isoformat()}
            })
            return result.deleted_count
            
        except Exception as e:
            raise MemoryException(
                f"Failed to cleanup memory items: {str(e)}",
                error_code="MEMORY_CLEANUP_ERROR"
            )


class InMemoryStore(MemoryStore):
    """In-memory storage for testing and development."""
    
    def __init__(self):
        """Initialize in-memory storage."""
        self._storage: Dict[str, List[MemoryItem]] = {}
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)
    
    async def store(self, item: MemoryItem) -> None:
        """Store a memory item in memory."""
        async with self._lock:
            if item.agent_id not in self._storage:
                self._storage[item.agent_id] = []
            self._storage[item.agent_id].append(item)
            self._logger.debug(f"Stored memory item: {item.id}")
    
    async def retrieve(
        self, 
        agent_id: str, 
        limit: int = 10,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        since: Optional[datetime] = None
    ) -> List[MemoryItem]:
        """Retrieve memory items from memory."""
        async with self._lock:
            if agent_id not in self._storage:
                return []
            
            items = self._storage[agent_id]
            
            # Apply filters
            filtered_items = []
            for item in items:
                if memory_type and item.memory_type != memory_type:
                    continue
                if tags and not any(tag in item.tags for tag in tags):
                    continue
                if since and item.timestamp < since:
                    continue
                filtered_items.append(item)
            
            # Sort by timestamp (newest first) and limit
            filtered_items.sort(key=lambda x: x.timestamp, reverse=True)
            return filtered_items[:limit]
    
    async def search(self, agent_id: str, query: str, limit: int = 5) -> List[MemoryItem]:
        """Search memory items by content."""
        async with self._lock:
            if agent_id not in self._storage:
                return []
            
            query_lower = query.lower()
            matching_items = []
            
            for item in self._storage[agent_id]:
                content_str = json.dumps(item.content).lower()
                tags_str = " ".join(item.tags).lower()
                
                if query_lower in content_str or query_lower in tags_str:
                    matching_items.append(item)
            
            # Sort by importance and limit
            matching_items.sort(key=lambda x: x.importance, reverse=True)
            return matching_items[:limit]
    
    async def delete(self, agent_id: str, item_id: str) -> bool:
        """Delete a specific memory item."""
        async with self._lock:
            if agent_id not in self._storage:
                return False
            
            items = self._storage[agent_id]
            for i, item in enumerate(items):
                if item.id == item_id:
                    items.pop(i)
                    return True
            return False
    
    async def cleanup(self, agent_id: str, before: datetime) -> int:
        """Cleanup old memory items."""
        async with self._lock:
            if agent_id not in self._storage:
                return 0
            
            items = self._storage[agent_id]
            original_count = len(items)
            
            self._storage[agent_id] = [
                item for item in items 
                if item.timestamp >= before
            ]
            
            return original_count - len(self._storage[agent_id])


class AgentMemoryManager:
    """Manages memory for individual agents."""
    
    def __init__(
        self, 
        agent_id: str, 
        agent_type: str,
        store: MemoryStore,
        config: MemoryConfig
    ):
        """
        Initialize agent memory manager.
        
        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent
            store: Memory storage backend
            config: Memory configuration
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.store = store
        self.config = config
        self._logger = logging.getLogger(__name__)
        self._cache: List[MemoryItem] = []
        self._cache_last_update: Optional[datetime] = None
    
    async def store_conversation(
        self, 
        message: str, 
        role: str = "user",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store a conversation message.
        
        Args:
            message: Message content
            role: Role (user, assistant, system)
            metadata: Additional metadata
        """
        item = MemoryItem(
            id=f"conv_{int(datetime.utcnow().timestamp() * 1000)}",
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            content={
                "message": message,
                "role": role
            },
            timestamp=datetime.utcnow(),
            memory_type="conversation",
            importance=0.6 if role == "user" else 0.4,
            tags=["conversation", role],
            metadata=metadata or {}
        )
        
        await self.store.store(item)
        self._invalidate_cache()
    
    async def store_tool_result(
        self, 
        tool_name: str, 
        result: Dict[str, Any],
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store a tool execution result.
        
        Args:
            tool_name: Name of the tool used
            result: Tool execution result
            success: Whether tool execution was successful
            metadata: Additional metadata
        """
        item = MemoryItem(
            id=f"tool_{int(datetime.utcnow().timestamp() * 1000)}",
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            content={
                "tool_name": tool_name,
                "result": result,
                "success": success
            },
            timestamp=datetime.utcnow(),
            memory_type="tool_result",
            importance=0.7 if success else 0.8,  # Failed results are more important to remember
            tags=["tool", tool_name, "success" if success else "failure"],
            metadata=metadata or {}
        )
        
        await self.store.store(item)
        self._invalidate_cache()
    
    async def store_insight(
        self, 
        insight: str, 
        category: str,
        confidence: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store an insight or conclusion.
        
        Args:
            insight: The insight content
            category: Category of insight
            confidence: Confidence level (0.0 to 1.0)
            metadata: Additional metadata
        """
        item = MemoryItem(
            id=f"insight_{int(datetime.utcnow().timestamp() * 1000)}",
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            content={
                "insight": insight,
                "category": category,
                "confidence": confidence
            },
            timestamp=datetime.utcnow(),
            memory_type="insight",
            importance=min(1.0, 0.5 + confidence / 2),  # Higher confidence = higher importance
            tags=["insight", category],
            metadata=metadata or {}
        )
        
        await self.store.store(item)
        self._invalidate_cache()
    
    async def get_conversation_history(
        self, 
        limit: int = 10
    ) -> List[MemoryItem]:
        """
        Get recent conversation history.
        
        Args:
            limit: Maximum number of items to retrieve
            
        Returns:
            List of conversation memory items
        """
        return await self.store.retrieve(
            agent_id=self.agent_id,
            limit=limit,
            memory_type="conversation"
        )
    
    async def get_relevant_context(
        self, 
        query: str, 
        limit: int = 5
    ) -> List[MemoryItem]:
        """
        Get relevant context for a query.
        
        Args:
            query: Search query
            limit: Maximum number of items to retrieve
            
        Returns:
            List of relevant memory items
        """
        return await self.store.search(
            agent_id=self.agent_id,
            query=query,
            limit=limit
        )
    
    async def get_recent_insights(
        self, 
        category: Optional[str] = None,
        limit: int = 5
    ) -> List[MemoryItem]:
        """
        Get recent insights.
        
        Args:
            category: Filter by category
            limit: Maximum number of items to retrieve
            
        Returns:
            List of insight memory items
        """
        tags = ["insight"]
        if category:
            tags.append(category)
        
        return await self.store.retrieve(
            agent_id=self.agent_id,
            limit=limit,
            memory_type="insight",
            tags=tags if category else None
        )
    
    async def cleanup_old_memories(self, days: int = 30) -> int:
        """
        Clean up old memories.
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of items deleted
        """
        if not self.config.enabled:
            return 0
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        deleted_count = await self.store.cleanup(self.agent_id, cutoff_date)
        
        self._logger.info(f"Cleaned up {deleted_count} old memories for agent {self.agent_id}")
        self._invalidate_cache()
        
        return deleted_count
    
    def _invalidate_cache(self) -> None:
        """Invalidate the internal cache."""
        self._cache.clear()
        self._cache_last_update = None


def create_memory_store(
    store_type: str = "mongo", 
    **kwargs
) -> MemoryStore:
    """
    Create a memory store instance.
    
    Args:
        store_type: Type of store ("mongo" or "memory")
        **kwargs: Additional arguments for store initialization
        
    Returns:
        MemoryStore instance
    """
    if store_type == "mongo":
        if 'collection' not in kwargs:
            raise MemoryException(
                "MongoDB collection is required for MongoMemoryStore",
                error_code="MEMORY_CONFIG_ERROR"
            )
        return MongoMemoryStore(kwargs['collection'])
    
    elif store_type == "memory":
        return InMemoryStore()
    
    else:
        raise MemoryException(
            f"Unknown memory store type: {store_type}",
            error_code="MEMORY_CONFIG_ERROR"
        )