"""
Google Scholar scraper for academic research.

Provides respectful scraping of Google Scholar with:
- Rate limiting and anti-bot measures
- Citation count extraction
- Author profile information
- Related articles discovery
- PDF availability detection
"""

import asyncio
import re
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus, urlencode, urlparse

import aiohttp
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

from .base_research_tool import BaseResearchTool, ResearchResult, SourceMetadata, SearchQuery


class GoogleScholarTool(BaseResearchTool):
    """
    Google Scholar scraper with respectful rate limiting.
    
    Features:
    - Citation count extraction
    - Author information
    - Related articles discovery
    - PDF availability detection
    - Anti-bot measures
    - Respectful rate limiting
    """
    
    def __init__(self):
        super().__init__(
            name="google_scholar",
            description="Search Google Scholar for academic papers and citations",
            source_type="academic",
            max_requests_per_minute=10  # Very conservative to avoid blocking
        )
        
        # Google Scholar configuration
        self.base_url = "https://scholar.google.com"
        self.search_url = f"{self.base_url}/scholar"
        
        # User agent rotation
        self.ua = UserAgent()
        
        # Session management
        self._session = None
        self._last_request_time = None
        
        # Headers rotation
        self.header_templates = [
            {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            },
            {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Cache-Control': 'max-age=0',
            }
        ]
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with rotating headers."""
        if self._session is None or self._session.closed:
            # Rotate headers and user agent
            headers = random.choice(self.header_templates).copy()
            headers['User-Agent'] = self.ua.random
            
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(limit=1, limit_per_host=1)  # Limit connections
            
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers,
                connector=connector
            )
        
        return self._session
    
    async def _perform_search(self, query: SearchQuery) -> List[ResearchResult]:
        """
        Perform Google Scholar search with anti-bot measures.
        
        Args:
            query: Search query object
            
        Returns:
            List of research results
        """
        try:
            # Apply extra delay for Google Scholar
            await self._scholar_rate_limit()
            
            # Build search parameters
            params = self._build_search_params(query)
            
            # Make search request
            html_content = await self._make_scholar_request(params)
            
            if not html_content:
                return []
            
            # Parse results
            results = self._parse_scholar_results(html_content, query.query)
            
            return results[:query.max_results]
            
        except Exception as e:
            self.logger.error(f"Google Scholar search failed: {str(e)}")
            raise
    
    async def _scholar_rate_limit(self):
        """Apply aggressive rate limiting for Google Scholar."""
        # Add random delay between 3-8 seconds
        delay = random.uniform(3.0, 8.0)
        
        if self._last_request_time:
            time_since_last = datetime.now().timestamp() - self._last_request_time
            if time_since_last < delay:
                sleep_time = delay - time_since_last
                self.logger.info(f"Google Scholar rate limiting: sleeping {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
        
        self._last_request_time = datetime.now().timestamp()
    
    def _build_search_params(self, query: SearchQuery) -> Dict[str, str]:
        """Build search parameters for Google Scholar."""
        params = {
            'q': query.query,
            'hl': 'en',
            'as_sdt': '0,5',  # Include patents and citations
            'as_vis': '1',    # Include citations
        }
        
        # Handle filters
        filters = query.filters or {}
        
        # Author filter
        if 'author' in filters:
            params['as_sauthors'] = filters['author']
        
        # Date range filter
        if query.date_range:
            if query.date_range == 'year':
                current_year = datetime.now().year
                params['as_ylo'] = str(current_year - 1)
            elif query.date_range == '5years':
                current_year = datetime.now().year
                params['as_ylo'] = str(current_year - 5)
        
        # Sort by date if requested
        if query.sort_by == 'date':
            params['scisbd'] = '1'
        
        return params
    
    async def _make_scholar_request(self, params: Dict[str, str]) -> Optional[str]:
        """Make request to Google Scholar with error handling."""
        try:
            session = await self._get_session()
            url = f"{self.search_url}?{urlencode(params)}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                elif response.status == 429:
                    self.logger.warning("Google Scholar rate limit exceeded")
                    # Wait longer and retry once
                    await asyncio.sleep(30)
                    async with session.get(url) as retry_response:
                        if retry_response.status == 200:
                            return await retry_response.text()
                else:
                    self.logger.warning(f"Google Scholar request failed: {response.status}")
                
                return None
        
        except Exception as e:
            self.logger.error(f"Google Scholar request error: {str(e)}")
            return None
    
    def _parse_scholar_results(self, html_content: str, original_query: str) -> List[ResearchResult]:
        """Parse Google Scholar search results."""
        results = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find result containers
            result_containers = soup.find_all('div', class_='gs_r gs_or gs_scl')
            
            for container in result_containers:
                try:
                    result = self._parse_scholar_result(container, original_query)
                    if result:
                        results.append(result)
                except Exception as e:
                    self.logger.warning(f"Failed to parse Scholar result: {str(e)}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Failed to parse Scholar HTML: {str(e)}")
        
        return results
    
    def _parse_scholar_result(self, container, original_query: str) -> Optional[ResearchResult]:
        """Parse individual Google Scholar result."""
        try:
            # Extract title and link
            title_elem = container.find('h3', class_='gs_rt')
            if not title_elem:
                return None
            
            title_link = title_elem.find('a')
            title = title_link.get_text(strip=True) if title_link else title_elem.get_text(strip=True)
            url = title_link.get('href', '') if title_link else ''
            
            # Extract snippet/abstract
            snippet_elem = container.find('div', class_='gs_rs')
            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
            
            # Extract authors and publication info
            authors_elem = container.find('div', class_='gs_a')
            authors_text = authors_elem.get_text(strip=True) if authors_elem else ""
            
            # Parse authors and publication details
            authors, publish_date, journal = self._parse_authors_info(authors_text)
            
            # Extract citation count
            citation_count = self._extract_citation_count(container)
            
            # Check for PDF availability
            pdf_link = self._extract_pdf_link(container)
            
            # Create metadata
            parsed_url = urlparse(url) if url else None
            domain = parsed_url.netloc if parsed_url else "scholar.google.com"
            
            metadata = SourceMetadata(
                url=url or f"https://scholar.google.com/scholar?q={quote_plus(original_query)}",
                title=title,
                source_type="academic",
                domain=domain,
                author=authors,
                publish_date=publish_date,
                credibility_score=0.8,  # Google Scholar has good quality control
                language="en"
            )
            
            # Extract key points from snippet
            key_points = self._extract_key_points_from_snippet(snippet)
            
            # Calculate confidence
            confidence = self._calculate_result_confidence(title, snippet, original_query, citation_count)
            
            # Create research result
            return ResearchResult(
                content=snippet,
                metadata=metadata,
                summary=self._create_result_summary(title, authors, journal, citation_count),
                key_points=key_points,
                confidence=confidence,
                raw_data={
                    'citation_count': citation_count,
                    'pdf_link': pdf_link,
                    'journal': journal,
                    'authors_raw': authors_text,
                    'source': 'google_scholar'
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to parse Scholar result: {str(e)}")
            return None
    
    def _parse_authors_info(self, authors_text: str) -> tuple[Optional[str], Optional[datetime], Optional[str]]:
        """Parse authors, date, and journal from authors line."""
        authors = None
        publish_date = None
        journal = None
        
        if not authors_text:
            return authors, publish_date, journal
        
        # Split by commas and dashes
        parts = re.split(r'[,-]', authors_text)
        
        for i, part in enumerate(parts):
            part = part.strip()
            
            # First part is usually authors
            if i == 0 and not re.search(r'\d{4}', part):
                authors = part
            
            # Look for year (4 digits)
            year_match = re.search(r'\b(19|20)\d{2}\b', part)
            if year_match and not publish_date:
                try:
                    year = int(year_match.group())
                    publish_date = datetime(year, 1, 1)
                except ValueError:
                    pass
            
            # Look for journal/venue (usually contains common publication words)
            journal_indicators = ['journal', 'conference', 'proceedings', 'workshop', 'symposium', 'acm', 'ieee']
            if any(indicator in part.lower() for indicator in journal_indicators):
                journal = part.strip()
        
        return authors, publish_date, journal
    
    def _extract_citation_count(self, container) -> int:
        """Extract citation count from result container."""
        try:
            citation_elem = container.find('div', class_='gs_fl')
            if citation_elem:
                citation_link = citation_elem.find('a', string=lambda text: text and 'Cited by' in text)
                if citation_link:
                    citation_text = citation_link.get_text()
                    citation_match = re.search(r'Cited by (\d+)', citation_text)
                    if citation_match:
                        return int(citation_match.group(1))
            return 0
        except Exception:
            return 0
    
    def _extract_pdf_link(self, container) -> Optional[str]:
        """Extract PDF link if available."""
        try:
            pdf_link = container.find('div', class_='gs_or_ggsm')
            if pdf_link:
                link_elem = pdf_link.find('a')
                if link_elem:
                    href = link_elem.get('href', '')
                    if href and ('.pdf' in href.lower() or 'pdf' in link_elem.get_text().lower()):
                        return href
            return None
        except Exception:
            return None
    
    def _extract_key_points_from_snippet(self, snippet: str) -> List[str]:
        """Extract key points from result snippet."""
        if not snippet or len(snippet) < 50:
            return []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', snippet)
        
        # Filter meaningful sentences
        key_points = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and not sentence.startswith('...'):
                key_points.append(sentence)
        
        return key_points[:3]  # Limit to 3 key points
    
    def _calculate_result_confidence(self, title: str, snippet: str, query: str, citation_count: int) -> float:
        """Calculate confidence score for Scholar result."""
        score = 0.0
        query_terms = set(query.lower().split())
        
        # Title relevance (40%)
        title_lower = title.lower()
        title_matches = sum(1 for term in query_terms if term in title_lower)
        title_score = min(title_matches / len(query_terms), 1.0) * 0.4
        score += title_score
        
        # Content relevance (30%)
        snippet_lower = snippet.lower()
        snippet_matches = sum(1 for term in query_terms if term in snippet_lower)
        snippet_score = min(snippet_matches / len(query_terms), 1.0) * 0.3
        score += snippet_score
        
        # Citation count bonus (30%)
        if citation_count > 100:
            citation_score = 0.3
        elif citation_count > 50:
            citation_score = 0.25
        elif citation_count > 10:
            citation_score = 0.2
        elif citation_count > 0:
            citation_score = 0.15
        else:
            citation_score = 0.1
        
        score += citation_score
        
        return min(score, 1.0)
    
    def _create_result_summary(self, title: str, authors: Optional[str], journal: Optional[str], citation_count: int) -> str:
        """Create summary for Scholar result."""
        summary_parts = []
        
        if authors:
            summary_parts.append(f"Authors: {authors}")
        
        if journal:
            summary_parts.append(f"Published in: {journal}")
        
        if citation_count > 0:
            summary_parts.append(f"Citations: {citation_count}")
        
        summary_parts.append(f"Title: {title}")
        
        return " | ".join(summary_parts)
    
    async def search_by_author(self, author: str, max_results: int = 10) -> List[ResearchResult]:
        """
        Search papers by specific author.
        
        Args:
            author: Author name
            max_results: Maximum number of results
            
        Returns:
            List of research results
        """
        search_query = SearchQuery(
            query="",
            max_results=max_results,
            filters={'author': author}
        )
        
        return await self.search(search_query)
    
    async def search_recent_papers(self, query: str, years: int = 5, max_results: int = 10) -> List[ResearchResult]:
        """
        Search for recent papers in the last N years.
        
        Args:
            query: Search query
            years: Number of years back to search
            max_results: Maximum number of results
            
        Returns:
            List of recent research results
        """
        date_filter = '5years' if years >= 5 else 'year'
        
        search_query = SearchQuery(
            query=query,
            max_results=max_results,
            date_range=date_filter,
            sort_by='date'
        )
        
        return await self.search(search_query)
    
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