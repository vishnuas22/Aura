"""
Web Research Tools Package - Comprehensive research capabilities for multi-agent system.

This package provides production-ready research tools including:
- DuckDuckGo search integration
- Wikipedia API wrapper
- arXiv paper search
- Google Scholar scraper
- News API integration
- Reddit search
- Advanced web scraping

All tools include rate limiting, caching, content validation, and relevance scoring.
"""

from .base_research_tool import BaseResearchTool, ResearchResult, SourceMetadata
from .duckduckgo_search import DuckDuckGoSearchTool
from .wikipedia_search import WikipediaSearchTool
from .arxiv_search import ArxivSearchTool
from .google_scholar import GoogleScholarTool
from .news_search import NewsSearchTool
from .reddit_search import RedditSearchTool
from .web_scraper import WebScrapingTool
from .research_cache import ResearchCache
from .content_validator import ContentValidator
from .relevance_scorer import RelevanceScorer

__all__ = [
    'BaseResearchTool',
    'ResearchResult', 
    'SourceMetadata',
    'DuckDuckGoSearchTool',
    'WikipediaSearchTool',
    'ArxivSearchTool',
    'GoogleScholarTool',
    'NewsSearchTool',
    'RedditSearchTool',
    'WebScrapingTool',
    'ResearchCache',
    'ContentValidator',
    'RelevanceScorer'
]

# Tool registry for easy access
RESEARCH_TOOLS = {
    'duckduckgo': DuckDuckGoSearchTool,
    'wikipedia': WikipediaSearchTool,
    'arxiv': ArxivSearchTool,
    'google_scholar': GoogleScholarTool,
    'news': NewsSearchTool,
    'reddit': RedditSearchTool,
    'web_scraper': WebScrapingTool
}

def get_research_tool(tool_name: str) -> BaseResearchTool:
    """
    Get a research tool by name.
    
    Args:
        tool_name: Name of the research tool
        
    Returns:
        Research tool instance
        
    Raises:
        ValueError: If tool name is not found
    """
    if tool_name not in RESEARCH_TOOLS:
        available_tools = list(RESEARCH_TOOLS.keys())
        raise ValueError(f"Unknown research tool: {tool_name}. Available tools: {available_tools}")
    
    return RESEARCH_TOOLS[tool_name]()

def get_all_research_tools() -> dict[str, BaseResearchTool]:
    """
    Get all available research tools.
    
    Returns:
        Dictionary of tool name -> tool instance
    """
    return {name: tool_class() for name, tool_class in RESEARCH_TOOLS.items()}