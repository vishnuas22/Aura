"""
DuckDuckGo search integration for web research.

Uses DuckDuckGo's free search API without requiring API keys.
Provides web search, news search, and instant answers.
"""

import aiohttp
import asyncio
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus, urlencode

from .base_research_tool import BaseResearchTool, ResearchResult, SourceMetadata, SearchQuery


class DuckDuckGoSearchTool(BaseResearchTool):
    """
    DuckDuckGo search tool for web research.
    
    Features:
    - Web search without API keys
    - News search capability  
    - Instant answers and snippets
    - Safe search filtering
    - Region-specific results
    """
    
    def __init__(self):
        super().__init__(
            name="duckduckgo_search",
            description="Search the web using DuckDuckGo without API keys",
            source_type="web",
            max_requests_per_minute=60  # DuckDuckGo is quite permissive
        )
        
        # DuckDuckGo API endpoints
        self.base_url = "https://api.duckduckgo.com/"
        self.html_url = "https://html.duckduckgo.com/html/"
        
        # Session for connection pooling
        self._session = None
        
        # Headers to mimic browser requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self.headers
            )
        return self._session
    
    async def _perform_search(self, query: SearchQuery) -> List[ResearchResult]:
        """
        Perform DuckDuckGo search.
        
        Args:
            query: Search query object
            
        Returns:
            List of research results
        """
        try:
            # Try instant answer API first
            instant_results = await self._get_instant_answers(query)
            
            # Get web search results
            web_results = await self._get_web_results(query)
            
            # Combine results
            all_results = instant_results + web_results
            
            # Sort by relevance score
            all_results.sort(key=lambda r: r.metadata.relevance_score, reverse=True)
            
            # Return top results
            return all_results[:query.max_results]
            
        except Exception as e:
            self.logger.error(f"DuckDuckGo search failed: {str(e)}")
            raise
    
    async def _get_instant_answers(self, query: SearchQuery) -> List[ResearchResult]:
        """Get instant answers from DuckDuckGo API."""
        results = []
        
        try:
            session = await self._get_session()
            params = {
                'q': query.query,
                'format': 'json',
                'no_redirect': '1',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            url = f"{self.base_url}?{urlencode(params)}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Process instant answer
                    if data.get('Abstract'):
                        result = self._create_instant_answer_result(data, query.query)
                        if result:
                            results.append(result)
                    
                    # Process related topics
                    for topic in data.get('RelatedTopics', [])[:3]:
                        if isinstance(topic, dict) and topic.get('Text'):
                            result = self._create_related_topic_result(topic, query.query)
                            if result:
                                results.append(result)
        
        except Exception as e:
            self.logger.warning(f"Failed to get instant answers: {str(e)}")
        
        return results
    
    async def _get_web_results(self, query: SearchQuery) -> List[ResearchResult]:
        """Get web search results by scraping DuckDuckGo HTML."""
        results = []
        
        try:
            session = await self._get_session()
            
            # Search parameters
            params = {
                'q': query.query,
                'l': query.language,
                's': '0',  # Start from first result
                'dc': str(min(query.max_results, 50)),  # Number of results
                'v': 'l',  # Version
                'o': 'json',
                'api': '/d.js'
            }
            
            # Add date filter if specified
            if query.date_range:
                if query.date_range == 'day':
                    params['df'] = 'd'
                elif query.date_range == 'week':
                    params['df'] = 'w'
                elif query.date_range == 'month':
                    params['df'] = 'm'
                elif query.date_range == 'year':
                    params['df'] = 'y'
            
            url = f"{self.html_url}?{urlencode(params)}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    html_content = await response.text()
                    results = self._parse_html_results(html_content, query.query)
        
        except Exception as e:
            self.logger.warning(f"Failed to get web results: {str(e)}")
        
        return results
    
    def _create_instant_answer_result(self, data: Dict[str, Any], query: str) -> Optional[ResearchResult]:
        """Create result from instant answer data."""
        try:
            abstract = data.get('Abstract', '')
            abstract_source = data.get('AbstractSource', '')
            abstract_url = data.get('AbstractURL', '')
            
            if not abstract:
                return None
            
            # Create metadata
            metadata = SourceMetadata(
                url=abstract_url or f"https://duckduckgo.com/?q={quote_plus(query)}",
                title=f"{abstract_source} - Instant Answer" if abstract_source else "DuckDuckGo Instant Answer",
                source_type="reference",
                domain=abstract_source.lower() if abstract_source else "duckduckgo.com",
                credibility_score=0.9,  # Instant answers are generally high quality
                language="en"
            )
            
            # Extract key points from abstract
            sentences = re.split(r'[.!?]+', abstract)
            key_points = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
            
            return ResearchResult(
                content=abstract,
                metadata=metadata,
                summary=abstract[:200] + "..." if len(abstract) > 200 else abstract,
                key_points=key_points,
                confidence=0.85,
                raw_data=data
            )
        
        except Exception as e:
            self.logger.warning(f"Failed to create instant answer result: {str(e)}")
            return None
    
    def _create_related_topic_result(self, topic: Dict[str, Any], query: str) -> Optional[ResearchResult]:
        """Create result from related topic data."""
        try:
            text = topic.get('Text', '')
            first_url = topic.get('FirstURL', '')
            
            if not text or len(text) < 50:
                return None
            
            # Extract title from text (usually the first part before dash or period)
            title_match = re.match(r'^([^-\.]+)', text)
            title = title_match.group(1).strip() if title_match else "Related Topic"
            
            metadata = SourceMetadata(
                url=first_url or f"https://duckduckgo.com/?q={quote_plus(query)}",
                title=title,
                source_type="reference",
                domain="duckduckgo.com",
                credibility_score=0.7,
                language="en"
            )
            
            return ResearchResult(
                content=text,
                metadata=metadata,
                summary=text[:150] + "..." if len(text) > 150 else text,
                key_points=[text],
                confidence=0.7,
                raw_data=topic
            )
        
        except Exception as e:
            self.logger.warning(f"Failed to create related topic result: {str(e)}")
            return None
    
    def _parse_html_results(self, html_content: str, query: str) -> List[ResearchResult]:
        """Parse HTML search results."""
        results = []
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find result containers
            result_containers = soup.find_all('div', class_='result')
            
            for container in result_containers[:10]:  # Limit to top 10
                try:
                    # Extract title and link
                    title_link = container.find('a', class_='result__a')
                    if not title_link:
                        continue
                    
                    title = title_link.get_text(strip=True)
                    url = title_link.get('href', '')
                    
                    # Extract snippet/description
                    snippet_elem = container.find('a', class_='result__snippet')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    # Skip if no meaningful content
                    if not title or not url or len(snippet) < 20:
                        continue
                    
                    # Clean up URL (DuckDuckGo sometimes uses redirects)
                    if url.startswith('/l/?kh=-1&uddg='):
                        # Extract real URL from redirect
                        import urllib.parse
                        parsed = urllib.parse.parse_qs(url.split('uddg=')[1])
                        if parsed:
                            url = list(parsed.keys())[0]
                    
                    # Create metadata
                    from urllib.parse import urlparse
                    parsed_url = urlparse(url)
                    domain = parsed_url.netloc.lower()
                    
                    metadata = SourceMetadata(
                        url=url,
                        title=title,
                        source_type="web",
                        domain=domain,
                        credibility_score=self._calculate_domain_credibility(domain),
                        language="en"
                    )
                    
                    # Extract key points from snippet
                    sentences = re.split(r'[.!?]+', snippet)
                    key_points = [s.strip() for s in sentences if len(s.strip()) > 10][:2]
                    
                    result = ResearchResult(
                        content=snippet,
                        metadata=metadata,
                        summary=snippet[:150] + "..." if len(snippet) > 150 else snippet,
                        key_points=key_points,
                        confidence=0.8,
                        raw_data={
                            'title': title,
                            'url': url,
                            'snippet': snippet,
                            'source': 'duckduckgo_web'
                        }
                    )
                    
                    results.append(result)
                
                except Exception as e:
                    self.logger.warning(f"Failed to parse result container: {str(e)}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Failed to parse HTML results: {str(e)}")
        
        return results
    
    def _calculate_domain_credibility(self, domain: str) -> float:
        """Calculate credibility score based on domain."""
        # High credibility domains
        high_credibility = [
            'wikipedia.org', 'britannica.com', 'nature.com', 'science.org',
            'nih.gov', 'cdc.gov', 'who.int', 'gov.uk', 'edu',
            'reuters.com', 'bbc.com', 'ap.org', 'npr.org'
        ]
        
        # Medium credibility domains
        medium_credibility = [
            'com', 'org', 'net', 'nytimes.com', 'washingtonpost.com',
            'guardian.com', 'economist.com', 'wsj.com'
        ]
        
        domain_lower = domain.lower()
        
        # Check for high credibility patterns
        for pattern in high_credibility:
            if pattern in domain_lower:
                return 0.9
        
        # Check for medium credibility patterns
        for pattern in medium_credibility:
            if pattern in domain_lower:
                return 0.7
        
        # Default credibility for unknown domains
        return 0.5
    
    async def search_news(self, query: str, max_results: int = 10) -> List[ResearchResult]:
        """
        Search for news articles using DuckDuckGo.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of news research results
        """
        # Create news-specific query
        news_query = SearchQuery(
            query=f"{query} news",
            max_results=max_results,
            date_range="month",  # Recent news
            sort_by="date"
        )
        
        return await self.search(news_query)
    
    async def get_instant_answer(self, query: str) -> Optional[ResearchResult]:
        """
        Get instant answer for a specific query.
        
        Args:
            query: Query for instant answer
            
        Returns:
            Research result with instant answer or None
        """
        search_query = SearchQuery(query=query, max_results=1)
        instant_results = await self._get_instant_answers(search_query)
        
        return instant_results[0] if instant_results else None
    
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