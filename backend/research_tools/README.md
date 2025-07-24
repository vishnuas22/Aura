# 🔍 Comprehensive Web Research Tools

A production-ready suite of web research tools for multi-agent AI systems. Provides intelligent, rate-limited, and cached access to multiple research sources including DuckDuckGo, Wikipedia, arXiv, Google Scholar, news sources, Reddit, and advanced web scraping.

## ✨ Features

### 🛠️ Research Tools
- **DuckDuckGo Search**: Free web search without API keys
- **Wikipedia API**: Intelligent querying with disambiguation handling
- **arXiv Search**: Academic paper research with filtering
- **Google Scholar**: Academic citation search (respectful rate limiting)
- **News API**: Multi-source news aggregation with credibility scoring
- **Reddit Search**: Community insights and discussions
- **Web Scraper**: Advanced content extraction with Beautiful Soup + Selenium

### 🚀 Advanced Features
- **Rate Limiting**: Respectful API usage with configurable limits
- **Intelligent Caching**: TTL-based caching with query similarity detection
- **Content Validation**: Quality assessment and spam detection
- **Relevance Scoring**: TF-IDF based relevance ranking
- **Error Handling**: Comprehensive error recovery and retries
- **Async Support**: Full asyncio support for concurrent operations

### 📊 Quality Assurance
- **Source Metadata**: Comprehensive source information and credibility scores
- **Content Cleaning**: Automated content extraction and cleaning
- **Duplicate Detection**: Intelligent duplicate removal
- **Language Detection**: Multi-language content support

## 📦 Installation

### Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

### Required Packages
```text
# Web Research Tools Dependencies
beautifulsoup4>=4.12.3
selenium>=4.19.0
lxml>=5.1.0
wikipedia>=1.4.0
arxiv>=2.1.0
feedparser>=6.0.11
fake-useragent>=1.4.0
cachetools>=5.3.3
ratelimit>=2.2.1
tenacity>=8.2.3
html2text>=2024.2.26
readability-lxml>=0.8.1
webdriver-manager>=4.0.1
praw>=7.7.1
```

### Optional API Keys
Create a `.env` file in your backend directory:

```env
# Optional: For enhanced news search
NEWS_API_KEY=your_news_api_key_here

# Optional: For Reddit API access (better rate limits)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=research-tool:v1.0 (by /u/your-username)
```

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from research_tools import DuckDuckGoSearchTool, SearchQuery

async def basic_search():
    async with DuckDuckGoSearchTool() as ddg:
        query = SearchQuery(
            query="artificial intelligence applications",
            max_results=5,
            language="en"
        )
        
        results = await ddg.search(query)
        
        for result in results:
            print(f"Title: {result.metadata.title}")
            print(f"URL: {result.metadata.url}")
            print(f"Relevance: {result.metadata.relevance_score:.2f}")
            print(f"Summary: {result.summary}")
            print("-" * 50)

# Run the search
asyncio.run(basic_search())
```

### Multi-Tool Research

```python
from research_tools_examples import ResearchOrchestrator

async def comprehensive_research():
    orchestrator = ResearchOrchestrator()
    
    try:
        results = await orchestrator.comprehensive_search(
            query="climate change solutions",
            tools=['duckduckgo', 'wikipedia', 'news'],
            max_results_per_tool=5,
            min_relevance=0.4
        )
        
        print(f"Found {results['total_results']} total results")
        print(f"Relevant results: {results['relevant_results']}")
        
        # Show top sources
        for source in results['summary']['top_sources']:
            print(f"• {source['title']} (relevance: {source['relevance']:.2f})")
    
    finally:
        await orchestrator.close()

asyncio.run(comprehensive_research())
```

## 🔧 Individual Tool Usage

### DuckDuckGo Search
```python
from research_tools import DuckDuckGoSearchTool

async with DuckDuckGoSearchTool() as ddg:
    # Web search
    results = await ddg.search("Python programming")
    
    # News search
    news_results = await ddg.search_news("latest tech news", max_results=10)
    
    # Instant answers
    answer = await ddg.get_instant_answer("what is machine learning")
```

### Wikipedia Research
```python
from research_tools import WikipediaSearchTool

async with WikipediaSearchTool() as wiki:
    # Basic search
    results = await wiki.search("artificial intelligence")
    
    # Multi-language support
    wiki.set_language('es')  # Spanish
    spanish_results = await wiki.search("inteligencia artificial")
    
    # Category search
    category_results = await wiki.search_by_category("Machine learning", max_results=5)
```

### Academic Research (arXiv)
```python
from research_tools import ArxivSearchTool, SearchQuery

async with ArxivSearchTool() as arxiv:
    # Recent papers
    query = SearchQuery(
        query="deep learning neural networks",
        max_results=10,
        date_range="month",
        sort_by="date"
    )
    results = await arxiv.search(query)
    
    # Search by author
    author_papers = await arxiv.search_by_author("Yoshua Bengio")
    
    # Search by category
    cs_papers = await arxiv.search_by_category("cs", "machine learning")
```

### News Research
```python
from research_tools import NewsSearchTool

async with NewsSearchTool() as news:
    # Recent news
    results = await news.search(SearchQuery(
        query="artificial intelligence regulation",
        date_range="week",
        max_results=10
    ))
    
    # Breaking news
    breaking = await news.search_breaking_news(max_results=5)
    
    # Source-specific search
    bbc_results = await news.search_by_source("bbc", query="technology")
```

### Reddit Community Insights
```python
from research_tools import RedditSearchTool

async with RedditSearchTool() as reddit:
    # General search
    results = await reddit.search("machine learning career advice")
    
    # Subreddit-specific search
    ml_results = await reddit.search_subreddit("MachineLearning", "career")
    
    # Trending topics
    trending = await reddit.get_trending_topics("technology", max_results=10)
```

### Web Scraping
```python
from research_tools import WebScrapingTool

async with WebScrapingTool() as scraper:
    # Scrape specific URLs
    urls = [
        "https://example.com/article1",
        "https://example.com/article2"
    ]
    results = await scraper.scrape_urls(urls, max_concurrent=3)
    
    # Extract links from a page
    links = await scraper.extract_links_from_page(
        "https://example.com",
        filter_domain=True
    )
```

## 🎯 Advanced Features

### Caching System
```python
from research_tools import ResearchCache, get_research_cache

# Initialize cache
cache = ResearchCache(
    max_size=1000,
    default_ttl=3600,  # 1 hour
    persistent_storage=True
)

# Cache is automatically used by tools
async with DuckDuckGoSearchTool() as ddg:
    # First search - fetched from source
    results1 = await ddg.search("AI research")
    
    # Second search - retrieved from cache
    results2 = await ddg.search("AI research")

# Cache statistics
stats = cache.get_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
```

### Content Validation
```python
from research_tools import ContentValidator

validator = ContentValidator()

# Validate individual results
for result in results:
    validation = validator.validate_result(result)
    
    if validation['is_valid']:
        print(f"Quality score: {validation['quality_score']:.2f}")
        print(f"Spam probability: {validation['spam_probability']:.2f}")
    else:
        print(f"Issues: {validation['issues']}")

# Filter results by quality
high_quality_results = validator.filter_results(results, min_quality=0.7)
```

### Relevance Scoring
```python
from research_tools import RelevanceScorer

scorer = RelevanceScorer()

# Score and rank results
scored_results = scorer.score_results(results, "machine learning applications")

# Get top results
top_results = scorer.get_top_results_by_relevance(
    results, 
    query="AI research",
    top_k=10,
    min_relevance=0.5
)

# Explain relevance score
explanation = scorer.explain_relevance_score(results[0], "AI research")
print(f"Relevance factors: {explanation['factors']}")
```

## 🔄 Integration with Multi-Agent Systems

### Agent Tool Integration
```python
# In your agent implementation
from research_tools import get_research_tool, RESEARCH_TOOLS

class ResearchAgent:
    def __init__(self):
        self.tools = {
            name: get_research_tool(name)
            for name in ['duckduckgo', 'wikipedia', 'arxiv']
        }
    
    async def research_topic(self, topic: str, context: dict = None):
        results = []
        
        for tool_name, tool in self.tools.items():
            try:
                tool_results = await tool.search(topic)
                results.extend(tool_results)
            except Exception as e:
                logger.error(f"Tool {tool_name} failed: {e}")
        
        return self._synthesize_results(results, topic)
```

### CrewAI Integration Example
```python
from crewai import Tool
from research_tools import ResearchOrchestrator

def create_research_tool():
    """Create CrewAI tool for research."""
    orchestrator = ResearchOrchestrator()
    
    async def research_function(query: str) -> str:
        results = await orchestrator.comprehensive_search(
            query=query,
            max_results_per_tool=3
        )
        
        # Format results for agent consumption
        formatted_results = []
        for result in results['combined_results'][:5]:
            formatted_results.append(
                f"Title: {result.metadata.title}\n"
                f"Source: {result.metadata.domain}\n"
                f"Summary: {result.summary}\n"
                f"URL: {result.metadata.url}\n"
            )
        
        return "\n---\n".join(formatted_results)
    
    return Tool(
        name="comprehensive_research",
        description="Perform comprehensive research across multiple sources",
        func=research_function
    )
```

## ⚡ Performance Optimization

### Concurrent Searches
```python
import asyncio
from research_tools import DuckDuckGoSearchTool, WikipediaSearchTool

async def concurrent_research():
    tools = {
        'ddg': DuckDuckGoSearchTool(),
        'wiki': WikipediaSearchTool()
    }
    
    query = "artificial intelligence ethics"
    
    # Run searches concurrently
    tasks = [
        tools['ddg'].search(query),
        tools['wiki'].search(query)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    all_results = []
    for result in results:
        if isinstance(result, list):
            all_results.extend(result)
    
    return all_results
```

### Rate Limiting Configuration
```python
# Configure rate limits per tool
tools_config = {
    'duckduckgo': {'max_requests_per_minute': 60},
    'google_scholar': {'max_requests_per_minute': 10},  # Very conservative
    'wikipedia': {'max_requests_per_minute': 100}
}

# Initialize tools with custom limits
ddg = DuckDuckGoSearchTool()
ddg.max_requests_per_minute = tools_config['duckduckgo']['max_requests_per_minute']
```

## 🧪 Testing

### Run Unit Tests
```bash
# Install testing dependencies
pip install pytest pytest-asyncio

# Run all tests
cd backend/research_tools
python -m pytest test_research_tools.py -v --asyncio-mode=auto

# Run specific test class
python -m pytest test_research_tools.py::TestDuckDuckGoSearch -v

# Run with coverage
pip install pytest-cov
python -m pytest test_research_tools.py --cov=. --cov-report=html
```

### Performance Testing
```python
# Run performance tests
python -m pytest test_research_tools.py::TestPerformance -v
```

### Integration Testing
```python
# Test tool integration
python research_tools_examples.py
```

## 🚨 Error Handling

### Best Practices
```python
import logging
from research_tools import DuckDuckGoSearchTool

logger = logging.getLogger(__name__)

async def robust_search(query: str):
    async with DuckDuckGoSearchTool() as tool:
        try:
            results = await tool.search(query)
            return results
        
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            
            # Fallback strategy
            try:
                # Try with simpler query
                simple_query = query.split()[0]  # First word only
                results = await tool.search(simple_query)
                logger.info(f"Fallback search succeeded with '{simple_query}'")
                return results
            
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
                return []
```

### Timeout Handling
```python
import asyncio

async def search_with_timeout(tool, query, timeout=30):
    try:
        return await asyncio.wait_for(
            tool.search(query),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.warning(f"Search timed out after {timeout} seconds")
        return []
```

## 📊 Monitoring and Metrics

### Performance Monitoring
```python
from research_tools import ResearchCache, ContentValidator

# Cache performance
cache = get_research_cache()
cache_stats = cache.get_stats()

print(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")
print(f"Cache size: {cache_stats['cache_size']}")

# Tool performance
tool = DuckDuckGoSearchTool()
rate_stats = tool.get_rate_limit_stats()

print(f"Remaining requests: {rate_stats['remaining_requests']}")
print(f"Reset time: {rate_stats['reset_time']}")

# Content quality metrics
validator = ContentValidator()
quality_report = validator.get_quality_report(results)

print(f"Average quality: {quality_report['average_quality']:.2f}")
print(f"Valid results: {quality_report['valid_results']}/{quality_report['total_results']}")
```

## 🛡️ Security Considerations

### Safe Scraping Practices
- Respectful rate limiting (implemented by default)
- User agent rotation to avoid blocking
- Timeout handling for hung requests
- Error recovery for network issues

### Data Privacy
- No personal data storage in cache by default
- Configurable cache TTL and size limits
- Optional persistent storage with encryption support

### API Key Security
- Environment variable storage
- Optional API key rotation
- Graceful degradation when keys are unavailable

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd research-tools

# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Run tests
pytest
```

### Adding New Tools
1. Inherit from `BaseResearchTool`
2. Implement `_perform_search` method
3. Add comprehensive error handling
4. Include rate limiting
5. Add unit tests
6. Update documentation

### Code Style
- Follow PEP 8
- Use type hints
- Include comprehensive docstrings
- Add logging for debugging
- Handle errors gracefully

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DuckDuckGo** for providing free search API
- **Wikipedia** for open knowledge access
- **arXiv** for academic paper access
- **Beautiful Soup & Selenium** for web scraping capabilities
- **aiohttp** for async HTTP client functionality

## 📞 Support

For issues, questions, or contributions:

1. Check existing issues in the repository
2. Create a new issue with detailed description
3. Include error logs and reproduction steps
4. Tag appropriate labels (bug, enhancement, question)

---

**Happy Researching! 🔍✨**