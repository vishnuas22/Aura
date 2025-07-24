"""
Research Tools Integration Examples and Usage Guide.

This module provides comprehensive examples of how to use the research tools
in various scenarios including:
- Individual tool usage
- Multi-tool research workflows
- Advanced filtering and scoring
- Performance optimization
- Error handling best practices
"""

import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime

# Import all research tools
from research_tools import (
    DuckDuckGoSearchTool,
    WikipediaSearchTool,
    ArxivSearchTool,
    GoogleScholarTool,
    NewsSearchTool,
    RedditSearchTool,
    WebScrapingTool,
    ResearchCache,
    ContentValidator,
    RelevanceScorer,
    SearchQuery
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResearchOrchestrator:
    """
    Orchestrates multiple research tools for comprehensive research.
    
    Provides high-level interface for conducting multi-source research
    with automatic caching, validation, and relevance scoring.
    """
    
    def __init__(self, use_cache: bool = True, validate_content: bool = True):
        """
        Initialize research orchestrator.
        
        Args:
            use_cache: Whether to use result caching
            validate_content: Whether to validate content quality
        """
        self.use_cache = use_cache
        self.validate_content = validate_content
        
        # Initialize tools
        self.tools = {
            'duckduckgo': DuckDuckGoSearchTool(),
            'wikipedia': WikipediaSearchTool(),
            'arxiv': ArxivSearchTool(),
            'google_scholar': GoogleScholarTool(),
            'news': NewsSearchTool(),
            'reddit': RedditSearchTool(),
            'web_scraper': WebScrapingTool()
        }
        
        # Initialize utilities
        self.cache = ResearchCache() if use_cache else None
        self.validator = ContentValidator() if validate_content else None
        self.scorer = RelevanceScorer()
        
        logger.info("Research orchestrator initialized")
    
    async def comprehensive_search(self, 
                                 query: str, 
                                 tools: List[str] = None,
                                 max_results_per_tool: int = 5,
                                 min_relevance: float = 0.3) -> Dict[str, Any]:
        """
        Perform comprehensive research across multiple tools.
        
        Args:
            query: Research query
            tools: List of tool names to use (None for all)
            max_results_per_tool: Maximum results per tool
            min_relevance: Minimum relevance threshold
            
        Returns:
            Dictionary with results from each tool and summary
        """
        if tools is None:
            tools = list(self.tools.keys())
        
        logger.info(f"Starting comprehensive search for: {query}")
        
        search_query = SearchQuery(
            query=query,
            max_results=max_results_per_tool,
            language='en'
        )
        
        # Run searches concurrently
        tasks = []
        for tool_name in tools:
            if tool_name in self.tools:
                task = self._search_with_tool(tool_name, search_query)
                tasks.append((tool_name, task))
        
        # Execute all searches
        results_by_tool = {}
        for tool_name, task in tasks:
            try:
                results = await task
                results_by_tool[tool_name] = results
                logger.info(f"{tool_name}: Found {len(results)} results")
            except Exception as e:
                logger.error(f"{tool_name} search failed: {str(e)}")
                results_by_tool[tool_name] = []
        
        # Combine and process all results
        all_results = []
        for tool_results in results_by_tool.values():
            all_results.extend(tool_results)
        
        # Validate content if enabled
        if self.validate_content and all_results:
            all_results = self.validator.filter_results(all_results, min_quality=0.4)
            logger.info(f"After validation: {len(all_results)} results remain")
        
        # Score and rank results
        if all_results:
            all_results = self.scorer.score_results(all_results, query)
            
            # Filter by relevance
            relevant_results = [
                r for r in all_results 
                if r.metadata.relevance_score >= min_relevance
            ]
            
            logger.info(f"After relevance filtering: {len(relevant_results)} results remain")
        else:
            relevant_results = []
        
        # Generate summary
        summary = self._generate_research_summary(query, results_by_tool, relevant_results)
        
        return {
            'query': query,
            'results_by_tool': results_by_tool,
            'combined_results': relevant_results,
            'summary': summary,
            'total_results': sum(len(results) for results in results_by_tool.values()),
            'relevant_results': len(relevant_results)
        }
    
    async def _search_with_tool(self, tool_name: str, query: SearchQuery) -> List:
        """Search with a specific tool, handling caching."""
        tool = self.tools[tool_name]
        
        # Check cache first
        if self.cache:
            cached_results = await self.cache.get(query, tool_name)
            if cached_results:
                logger.debug(f"Cache hit for {tool_name}")
                return cached_results
        
        # Perform search
        results = await tool.search(query)
        
        # Cache results
        if self.cache and results:
            await self.cache.set(query, tool_name, results)
        
        return results
    
    def _generate_research_summary(self, query: str, results_by_tool: Dict, combined_results: List) -> Dict[str, Any]:
        """Generate research summary."""
        summary = {
            'query': query,
            'search_timestamp': datetime.now().isoformat(),
            'tools_used': list(results_by_tool.keys()),
            'total_sources': len(combined_results),
            'source_breakdown': {},
            'quality_metrics': {},
            'top_sources': []
        }
        
        # Source type breakdown
        source_counts = {}
        for result in combined_results:
            source_type = result.metadata.source_type
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
        
        summary['source_breakdown'] = source_counts
        
        # Quality metrics
        if combined_results:
            relevance_scores = [r.metadata.relevance_score for r in combined_results]
            confidence_scores = [r.confidence for r in combined_results]
            
            summary['quality_metrics'] = {
                'avg_relevance': sum(relevance_scores) / len(relevance_scores),
                'avg_confidence': sum(confidence_scores) / len(confidence_scores),
                'high_quality_count': len([r for r in combined_results if r.confidence > 0.8])
            }
        
        # Top sources
        top_results = sorted(combined_results, key=lambda r: r.metadata.relevance_score, reverse=True)[:5]
        summary['top_sources'] = [
            {
                'title': result.metadata.title,
                'url': result.metadata.url,
                'source_type': result.metadata.source_type,
                'relevance': result.metadata.relevance_score,
                'confidence': result.confidence
            }
            for result in top_results
        ]
        
        return summary
    
    async def academic_research(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Perform academic research using scholarly sources.
        
        Args:
            query: Research query
            max_results: Maximum results to return
            
        Returns:
            Academic research results
        """
        academic_tools = ['wikipedia', 'arxiv', 'google_scholar']
        return await self.comprehensive_search(
            query, 
            tools=academic_tools, 
            max_results_per_tool=max_results//len(academic_tools),
            min_relevance=0.5
        )
    
    async def current_events_research(self, query: str, max_results: int = 15) -> Dict[str, Any]:
        """
        Research current events and news.
        
        Args:
            query: Research query
            max_results: Maximum results to return
            
        Returns:
            Current events research results
        """
        news_tools = ['news', 'reddit', 'duckduckgo']
        
        # Use recent date filter
        search_query = SearchQuery(
            query=query,
            max_results=max_results//len(news_tools),
            date_range='month'  # Recent news
        )
        
        return await self.comprehensive_search(
            query,
            tools=news_tools,
            max_results_per_tool=search_query.max_results,
            min_relevance=0.3
        )
    
    async def deep_web_research(self, urls: List[str]) -> Dict[str, Any]:
        """
        Perform deep research by scraping specific URLs.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            Web scraping results
        """
        scraper = self.tools['web_scraper']
        
        # Scrape URLs concurrently
        results = await scraper.scrape_urls(urls, max_concurrent=3)
        
        if results:
            # Score results
            combined_query = ' '.join(urls)  # Use URLs as pseudo-query
            results = self.scorer.score_results(results, combined_query)
        
        return {
            'urls_scraped': urls,
            'results': results,
            'total_results': len(results),
            'successful_scrapes': len([r for r in results if r.content])
        }
    
    async def close(self):
        """Clean up resources."""
        # Close tool sessions
        for tool in self.tools.values():
            if hasattr(tool, '__aexit__'):
                try:
                    await tool.__aexit__(None, None, None)
                except Exception:
                    pass
        
        logger.info("Research orchestrator closed")


# Usage Examples
async def example_basic_search():
    """Example: Basic search using DuckDuckGo."""
    print("=== Basic DuckDuckGo Search ===")
    
    async with DuckDuckGoSearchTool() as ddg:
        query = SearchQuery(query="Python machine learning", max_results=5)
        results = await ddg.search(query)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.metadata.title}")
            print(f"   URL: {result.metadata.url}")
            print(f"   Relevance: {result.metadata.relevance_score:.2f}")
            print(f"   Summary: {result.summary[:100]}...")
            print()


async def example_wikipedia_research():
    """Example: Wikipedia research with language support."""
    print("=== Wikipedia Research ===")
    
    async with WikipediaSearchTool() as wiki:
        # Search in English
        results = await wiki.search("artificial intelligence")
        
        print(f"Found {len(results)} results")
        
        if results:
            result = results[0]
            print(f"Title: {result.metadata.title}")
            print(f"Summary: {result.summary[:200]}...")
            print(f"Key Points: {result.key_points[:3]}")


async def example_academic_research():
    """Example: Academic research using arXiv."""
    print("=== Academic Research (arXiv) ===")
    
    async with ArxivSearchTool() as arxiv:
        # Search recent papers
        query = SearchQuery(
            query="neural networks deep learning",
            max_results=3,
            date_range="year",
            sort_by="date"
        )
        
        results = await arxiv.search(query)
        
        for result in results:
            print(f"Title: {result.metadata.title}")
            print(f"Authors: {result.metadata.author}")
            print(f"Published: {result.metadata.publish_date}")
            print(f"arXiv ID: {result.raw_data.get('arxiv_id', 'N/A')}")
            print(f"Abstract: {result.content[:200]}...")
            print("-" * 50)


async def example_news_research():
    """Example: News research and analysis."""
    print("=== News Research ===")
    
    async with NewsSearchTool() as news:
        # Search recent tech news
        query = SearchQuery(
            query="artificial intelligence technology",
            max_results=5,
            date_range="week"
        )
        
        results = await news.search(query)
        
        for result in results:
            print(f"Title: {result.metadata.title}")
            print(f"Source: {result.metadata.domain}")
            print(f"Published: {result.metadata.publish_date}")
            print(f"Credibility: {result.metadata.credibility_score:.2f}")
            print(f"Summary: {result.summary}")
            print("-" * 50)


async def example_comprehensive_research():
    """Example: Comprehensive multi-tool research."""
    print("=== Comprehensive Research ===")
    
    orchestrator = ResearchOrchestrator()
    
    try:
        # Perform comprehensive research
        research_results = await orchestrator.comprehensive_search(
            query="climate change artificial intelligence",
            tools=['duckduckgo', 'wikipedia', 'news'],
            max_results_per_tool=3,
            min_relevance=0.4
        )
        
        # Print summary
        summary = research_results['summary']
        print(f"Query: {summary['query']}")
        print(f"Total results: {research_results['total_results']}")
        print(f"Relevant results: {research_results['relevant_results']}")
        print(f"Average relevance: {summary['quality_metrics'].get('avg_relevance', 0):.2f}")
        print()
        
        # Print top sources
        print("Top Sources:")
        for i, source in enumerate(summary['top_sources'], 1):
            print(f"{i}. {source['title']}")
            print(f"   Type: {source['source_type']}")
            print(f"   Relevance: {source['relevance']:.2f}")
            print(f"   URL: {source['url']}")
            print()
    
    finally:
        await orchestrator.close()


async def example_web_scraping():
    """Example: Web scraping specific URLs."""
    print("=== Web Scraping Example ===")
    
    async with WebScrapingTool() as scraper:
        urls = [
            "https://en.wikipedia.org/wiki/Machine_learning",
            "https://en.wikipedia.org/wiki/Artificial_intelligence"
        ]
        
        results = await scraper.scrape_urls(urls, max_concurrent=2)
        
        for result in results:
            print(f"URL: {result.metadata.url}")
            print(f"Title: {result.metadata.title}")
            print(f"Content length: {len(result.content)} characters")
            print(f"Key points: {len(result.key_points)}")
            print("-" * 50)


async def example_caching_and_performance():
    """Example: Caching and performance optimization."""
    print("=== Caching and Performance ===")
    
    # Initialize cache
    cache = ResearchCache(max_size=100, default_ttl=300)
    
    # Initialize validator and scorer
    validator = ContentValidator()
    scorer = RelevanceScorer()
    
    async with DuckDuckGoSearchTool() as ddg:
        query = "machine learning applications"
        
        # First search (will be cached)
        print("First search (no cache)...")
        start_time = asyncio.get_event_loop().time()
        results = await ddg.search(query)
        end_time = asyncio.get_event_loop().time()
        
        print(f"Search took {end_time - start_time:.2f} seconds")
        print(f"Found {len(results)} results")
        
        # Cache the results
        await cache.set(query, "duckduckgo", results)
        
        # Second search (from cache)
        print("\nSecond search (from cache)...")
        start_time = asyncio.get_event_loop().time()
        cached_results = await cache.get(query, "duckduckgo")
        end_time = asyncio.get_event_loop().time()
        
        print(f"Cache retrieval took {end_time - start_time:.4f} seconds")
        print(f"Retrieved {len(cached_results)} results from cache")
        
        # Validate and score results
        if cached_results:
            print("\nValidating and scoring results...")
            
            # Validate content
            validated_results = validator.filter_results(cached_results)
            print(f"After validation: {len(validated_results)} results")
            
            # Score relevance
            scored_results = scorer.score_results(validated_results, query)
            print(f"Scored {len(scored_results)} results")
            
            # Show top result
            if scored_results:
                top_result = scored_results[0]
                print(f"\nTop result:")
                print(f"Title: {top_result.metadata.title}")
                print(f"Relevance: {top_result.metadata.relevance_score:.2f}")
                print(f"Confidence: {top_result.confidence:.2f}")
    
    # Show cache stats
    print(f"\nCache statistics:")
    stats = cache.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


async def main():
    """Run all examples."""
    print("🔍 Research Tools Integration Examples\n")
    
    examples = [
        ("Basic Search", example_basic_search),
        ("Wikipedia Research", example_wikipedia_research),
        ("Academic Research", example_academic_research),
        ("News Research", example_news_research),
        ("Comprehensive Research", example_comprehensive_research),
        ("Web Scraping", example_web_scraping),
        ("Caching & Performance", example_caching_and_performance)
    ]
    
    for name, example_func in examples:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print('='*60)
        
        try:
            await example_func()
        except Exception as e:
            print(f"❌ Example failed: {str(e)}")
            logger.error(f"Example {name} failed", exc_info=True)
        
        print(f"\n✅ Completed: {name}")
        await asyncio.sleep(1)  # Brief pause between examples
    
    print(f"\n{'='*60}")
    print("🎉 All examples completed!")
    print('='*60)


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())