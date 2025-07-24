"""
News API integration for current events research.

Provides access to news articles with:
- Multiple news sources
- Date range filtering
- Sentiment analysis
- Source credibility scoring
- Real-time news updates
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode

import aiohttp
import feedparser

from .base_research_tool import BaseResearchTool, ResearchResult, SourceMetadata, SearchQuery


class NewsSearchTool(BaseResearchTool):
    """
    News search tool with multiple source integration.
    
    Features:
    - NewsAPI integration (with free tier)
    - RSS feed aggregation
    - Source credibility scoring
    - Sentiment analysis
    - Real-time updates
    - Multiple language support
    """
    
    def __init__(self):
        super().__init__(
            name="news_search",
            description="Search news articles from multiple sources",
            source_type="news",
            max_requests_per_minute=50  # Conservative for free tier
        )
        
        # NewsAPI configuration
        self.newsapi_key = os.getenv('NEWS_API_KEY')
        self.newsapi_url = "https://newsapi.org/v2"
        
        # RSS feeds for backup when no API key
        self.rss_feeds = {
            'bbc': 'http://feeds.bbci.co.uk/news/rss.xml',
            'reuters': 'http://feeds.reuters.com/reuters/topNews',
            'cnn': 'http://rss.cnn.com/rss/edition.rss',
            'npr': 'https://feeds.npr.org/1001/rss.xml',
            'guardian': 'https://www.theguardian.com/international/rss',
            'ap': 'https://rssfeed.today/rss/ap',
            'techcrunch': 'http://feeds.feedburner.com/TechCrunch',
            'ars-technica': 'http://feeds.arstechnica.com/arstechnica/index'
        }
        
        # Source credibility scores
        self.source_credibility = {
            'bbc.com': 0.95,
            'reuters.com': 0.95,
            'ap.org': 0.95,
            'npr.org': 0.9,
            'theguardian.com': 0.9,
            'washingtonpost.com': 0.9,
            'nytimes.com': 0.9,
            'wsj.com': 0.85,
            'cnn.com': 0.8,
            'techcrunch.com': 0.8,
            'arstechnica.com': 0.85,
            'bloomberg.com': 0.85,
            'economist.com': 0.9
        }
        
        # Session for connection pooling
        self._session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def _perform_search(self, query: SearchQuery) -> List[ResearchResult]:
        """
        Perform news search using available sources.
        
        Args:
            query: Search query object
            
        Returns:
            List of research results
        """
        try:
            results = []
            
            # Try NewsAPI first if available
            if self.newsapi_key:
                newsapi_results = await self._search_newsapi(query)
                results.extend(newsapi_results)
            
            # Use RSS feeds as backup or supplement
            if len(results) < query.max_results:
                rss_results = await self._search_rss_feeds(query)
                results.extend(rss_results)
            
            # Remove duplicates based on title similarity
            results = self._remove_duplicate_articles(results)
            
            # Sort by relevance and recency
            results.sort(key=lambda r: (r.metadata.relevance_score, 
                                      r.metadata.publish_date or datetime.min), 
                        reverse=True)
            
            return results[:query.max_results]
            
        except Exception as e:
            self.logger.error(f"News search failed: {str(e)}")
            raise
    
    async def _search_newsapi(self, query: SearchQuery) -> List[ResearchResult]:
        """Search using NewsAPI."""
        try:
            session = await self._get_session()
            
            # Build parameters
            params = {
                'q': query.query,
                'apiKey': self.newsapi_key,
                'language': query.language,
                'sortBy': 'relevancy' if query.sort_by == 'relevance' else 'publishedAt',
                'pageSize': min(query.max_results, 100)
            }
            
            # Add date range
            if query.date_range:
                from_date = self._get_date_from_range(query.date_range)
                if from_date:
                    params['from'] = from_date.isoformat()
            
            # Choose endpoint based on filters
            endpoint = '/everything'  # Most flexible endpoint
            
            url = f"{self.newsapi_url}{endpoint}?{urlencode(params)}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_newsapi_results(data, query.query)
                else:
                    self.logger.warning(f"NewsAPI request failed: {response.status}")
                    return []
        
        except Exception as e:
            self.logger.warning(f"NewsAPI search failed: {str(e)}")
            return []
    
    async def _search_rss_feeds(self, query: SearchQuery) -> List[ResearchResult]:
        """Search RSS feeds for news articles."""
        results = []
        
        # Determine relevant feeds based on query
        relevant_feeds = self._select_relevant_feeds(query.query)
        
        for feed_name, feed_url in relevant_feeds.items():
            try:
                feed_results = await self._parse_rss_feed(feed_url, feed_name, query)
                results.extend(feed_results)
            except Exception as e:
                self.logger.warning(f"Failed to parse RSS feed {feed_name}: {str(e)}")
                continue
        
        return results
    
    def _select_relevant_feeds(self, query: str) -> Dict[str, str]:
        """Select RSS feeds most relevant to the query."""
        query_lower = query.lower()
        
        # Tech-related keywords
        if any(keyword in query_lower for keyword in ['tech', 'ai', 'computer', 'software', 'internet', 'cyber']):
            return {
                'techcrunch': self.rss_feeds['techcrunch'],
                'ars-technica': self.rss_feeds['ars-technica'],
                'bbc': self.rss_feeds['bbc'],
                'reuters': self.rss_feeds['reuters']
            }
        
        # General news feeds
        return {
            'bbc': self.rss_feeds['bbc'],
            'reuters': self.rss_feeds['reuters'],
            'guardian': self.rss_feeds['guardian'],
            'npr': self.rss_feeds['npr']
        }
    
    async def _parse_rss_feed(self, feed_url: str, feed_name: str, query: SearchQuery) -> List[ResearchResult]:
        """Parse RSS feed and extract relevant articles."""
        try:
            session = await self._get_session()
            
            async with session.get(feed_url) as response:
                if response.status != 200:
                    return []
                
                feed_content = await response.text()
                feed = feedparser.parse(feed_content)
                
                results = []
                query_terms = set(query.query.lower().split())
                
                for entry in feed.entries[:20]:  # Limit RSS entries
                    # Check relevance
                    title_lower = entry.get('title', '').lower()
                    summary_lower = entry.get('summary', '').lower()
                    
                    # Simple relevance check
                    relevance = sum(1 for term in query_terms 
                                  if term in title_lower or term in summary_lower)
                    
                    if relevance > 0 or not query_terms:  # Include if relevant or no specific query
                        result = self._process_rss_entry(entry, feed_name, query.query)
                        if result:
                            results.append(result)
                
                return results
        
        except Exception as e:
            self.logger.warning(f"Failed to parse RSS feed: {str(e)}")
            return []
    
    def _process_newsapi_results(self, data: Dict[str, Any], original_query: str) -> List[ResearchResult]:
        """Process NewsAPI response data."""
        results = []
        
        for article in data.get('articles', []):
            try:
                result = self._create_news_result_from_api(article, original_query)
                if result:
                    results.append(result)
            except Exception as e:
                self.logger.warning(f"Failed to process NewsAPI article: {str(e)}")
                continue
        
        return results
    
    def _create_news_result_from_api(self, article: Dict[str, Any], original_query: str) -> Optional[ResearchResult]:
        """Create ResearchResult from NewsAPI article."""
        try:
            title = article.get('title', '')
            description = article.get('description', '')
            content = article.get('content', description)  # Fallback to description
            url = article.get('url', '')
            
            if not title or not url:
                return None
            
            # Parse publish date
            publish_date = None
            if article.get('publishedAt'):
                try:
                    publish_date = datetime.fromisoformat(
                        article['publishedAt'].replace('Z', '+00:00')
                    )
                except ValueError:
                    pass
            
            # Extract domain for credibility scoring
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            # Get source credibility
            credibility = self._get_source_credibility(domain)
            
            # Create metadata
            metadata = SourceMetadata(
                url=url,
                title=title,
                source_type="news",
                domain=domain,
                author=article.get('author'),
                publish_date=publish_date,
                credibility_score=credibility,
                language="en"
            )
            
            # Extract key points
            key_points = self._extract_news_key_points(content or description)
            
            # Calculate confidence
            confidence = self._calculate_news_confidence(title, content, original_query, credibility)
            
            return ResearchResult(
                content=content or description,
                metadata=metadata,
                summary=description[:200] + "..." if len(description) > 200 else description,
                key_points=key_points,
                confidence=confidence,
                raw_data={
                    'source_name': article.get('source', {}).get('name', ''),
                    'author': article.get('author'),
                    'published_at': article.get('publishedAt'),
                    'url_to_image': article.get('urlToImage'),
                    'source': 'newsapi'
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to create news result: {str(e)}")
            return None
    
    def _process_rss_entry(self, entry: Dict[str, Any], feed_name: str, original_query: str) -> Optional[ResearchResult]:
        """Process RSS feed entry."""
        try:
            title = entry.get('title', '')
            summary = entry.get('summary', '')
            link = entry.get('link', '')
            
            if not title or not link:
                return None
            
            # Parse publish date
            publish_date = None
            if entry.get('published_parsed'):
                try:
                    import time
                    timestamp = time.mktime(entry['published_parsed'])
                    publish_date = datetime.fromtimestamp(timestamp)
                except (ValueError, TypeError):
                    pass
            
            # Extract domain
            from urllib.parse import urlparse
            parsed_url = urlparse(link)
            domain = parsed_url.netloc.lower()
            
            # Get source credibility
            credibility = self._get_source_credibility(domain)
            
            # Create metadata
            metadata = SourceMetadata(
                url=link,
                title=title,
                source_type="news",
                domain=domain,
                author=entry.get('author'),
                publish_date=publish_date,
                credibility_score=credibility,
                language="en"
            )
            
            # Extract key points
            key_points = self._extract_news_key_points(summary)
            
            # Calculate confidence
            confidence = self._calculate_news_confidence(title, summary, original_query, credibility)
            
            return ResearchResult(
                content=summary,
                metadata=metadata,
                summary=summary[:200] + "..." if len(summary) > 200 else summary,
                key_points=key_points,
                confidence=confidence,
                raw_data={
                    'feed_name': feed_name,
                    'published': entry.get('published'),
                    'tags': entry.get('tags', []),
                    'source': 'rss'
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to process RSS entry: {str(e)}")
            return None
    
    def _get_source_credibility(self, domain: str) -> float:
        """Get credibility score for news source."""
        # Check exact matches first
        if domain in self.source_credibility:
            return self.source_credibility[domain]
        
        # Check partial matches
        for source_domain, score in self.source_credibility.items():
            if source_domain in domain or domain in source_domain:
                return score
        
        # Default credibility for unknown sources
        return 0.6
    
    def _get_date_from_range(self, date_range: str) -> Optional[datetime]:
        """Convert date range string to datetime."""
        now = datetime.now()
        
        if date_range == 'day':
            return now - timedelta(days=1)
        elif date_range == 'week':
            return now - timedelta(weeks=1)
        elif date_range == 'month':
            return now - timedelta(days=30)
        elif date_range == 'year':
            return now - timedelta(days=365)
        
        return None
    
    def _extract_news_key_points(self, content: str) -> List[str]:
        """Extract key points from news content."""
        if not content or len(content) < 50:
            return []
        
        import re
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        
        # Key indicators for important news information
        key_indicators = [
            'according to', 'officials said', 'reported', 'announced',
            'confirmed', 'revealed', 'disclosed', 'stated', 'breaking',
            'urgent', 'developing', 'update'
        ]
        
        key_points = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            sentence_lower = sentence.lower()
            
            # Check for key indicators
            if any(indicator in sentence_lower for indicator in key_indicators):
                key_points.append(sentence)
            # Or substantial sentences with news-worthy content
            elif len(sentence) > 40 and any(word in sentence_lower for word in ['said', 'will', 'new', 'first', 'major']):
                key_points.append(sentence)
        
        # If no key indicators, take first few substantial sentences
        if not key_points:
            substantial_sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
            key_points = substantial_sentences[:2]
        
        return key_points[:4]  # Limit to 4 key points
    
    def _calculate_news_confidence(self, title: str, content: str, query: str, credibility: float) -> float:
        """Calculate confidence score for news result."""
        score = 0.0
        query_terms = set(query.lower().split())
        
        # Title relevance (40%)
        title_lower = title.lower()
        title_matches = sum(1 for term in query_terms if term in title_lower)
        title_score = min(title_matches / len(query_terms), 1.0) * 0.4
        score += title_score
        
        # Content relevance (35%)
        content_lower = content.lower()
        content_matches = sum(1 for term in query_terms if term in content_lower)
        content_score = min(content_matches / len(query_terms), 1.0) * 0.35
        score += content_score
        
        # Source credibility (25%)
        credibility_score = credibility * 0.25
        score += credibility_score
        
        return min(score, 1.0)
    
    def _remove_duplicate_articles(self, results: List[ResearchResult]) -> List[ResearchResult]:
        """Remove duplicate articles based on title similarity."""
        if not results:
            return results
        
        unique_results = []
        seen_titles = set()
        
        for result in results:
            title_lower = result.metadata.title.lower()
            
            # Simple deduplication based on title words
            title_words = set(title_lower.split())
            
            is_duplicate = False
            for seen_title in seen_titles:
                seen_words = set(seen_title.split())
                
                # Calculate similarity (Jaccard index)
                intersection = len(title_words & seen_words)
                union = len(title_words | seen_words)
                
                if union > 0 and intersection / union > 0.7:  # 70% similarity threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append(result)
                seen_titles.add(title_lower)
        
        return unique_results
    
    async def search_breaking_news(self, max_results: int = 10) -> List[ResearchResult]:
        """
        Search for breaking news stories.
        
        Args:
            max_results: Maximum number of results
            
        Returns:
            List of breaking news results
        """
        search_query = SearchQuery(
            query="breaking OR urgent OR developing",
            max_results=max_results,
            date_range="day",
            sort_by="date"
        )
        
        return await self.search(search_query)
    
    async def search_by_source(self, source: str, query: str = "", max_results: int = 10) -> List[ResearchResult]:
        """
        Search news from a specific source.
        
        Args:
            source: News source name
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of news results from the source
        """
        if source in self.rss_feeds:
            # Use RSS feed for this source
            search_query = SearchQuery(query=query, max_results=max_results)
            return await self._parse_rss_feed(self.rss_feeds[source], source, search_query)
        
        # Fallback to general search
        return await self.search(SearchQuery(query=f"{query} site:{source}", max_results=max_results))
    
    def get_supported_sources(self) -> List[str]:
        """Get list of supported news sources."""
        return list(self.rss_feeds.keys())
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        if self._session and not self._session.closed:
            asyncio.create_task(self._session.close())