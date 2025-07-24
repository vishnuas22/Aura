"""
Advanced web scraping tool with Beautiful Soup and Selenium.

Provides comprehensive web scraping capabilities including:
- Static HTML parsing with Beautiful Soup
- Dynamic content scraping with Selenium
- Content extraction and cleaning
- PDF text extraction
- Structured data extraction
- Anti-bot measures and rate limiting
"""

import asyncio
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urljoin, urlparse, quote_plus

import aiohttp
from bs4 import BeautifulSoup, Comment
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from fake_useragent import UserAgent
import html2text
from readability import Document

from .base_research_tool import BaseResearchTool, ResearchResult, SourceMetadata, SearchQuery


class WebScrapingTool(BaseResearchTool):
    """
    Advanced web scraping tool with multiple extraction methods.
    
    Features:
    - Static HTML parsing with Beautiful Soup
    - Dynamic content with Selenium
    - Content extraction and cleaning
    - PDF text extraction
    - Structured data extraction
    - Anti-bot measures
    - Rate limiting and retries
    """
    
    def __init__(self):
        super().__init__(
            name="web_scraper",
            description="Advanced web scraping with content extraction",
            source_type="web",
            max_requests_per_minute=30  # Conservative for web scraping
        )
        
        # User agent rotation
        self.ua = UserAgent()
        
        # HTML to text converter
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        self.html_converter.ignore_emphasis = False
        
        # Chrome driver options
        self.chrome_options = self._setup_chrome_options()
        
        # Session for HTTP requests
        self._session = None
        
        # Selenium driver pool
        self._driver_pool: List[webdriver.Chrome] = []
        self._max_drivers = 2
        
        # Content extraction patterns
        self.content_selectors = [
            'article', 'main', '.content', '#content', '.post-content',
            '.entry-content', '.article-content', '.story-body', 
            '.article-body', '.post-body', '[role="main"]'
        ]
        
        # Elements to remove for clean content
        self.remove_selectors = [
            'script', 'style', 'nav', 'header', 'footer', 'aside',
            '.advertisement', '.ads', '.sidebar', '.navigation',
            '.social-share', '.comments', '.related-posts'
        ]
    
    def _setup_chrome_options(self) -> Options:
        """Setup Chrome options for Selenium."""
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-images')
        options.add_argument('--disable-javascript')  # For most scraping tasks
        options.add_argument('--window-size=1920,1080')
        options.add_argument(f'--user-agent={self.ua.random}')
        
        return options
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                'User-Agent': self.ua.random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self._session
    
    async def _get_driver(self) -> webdriver.Chrome:
        """Get or create Chrome driver."""
        if not self._driver_pool:
            try:
                driver = webdriver.Chrome(
                    service=webdriver.chrome.service.Service(ChromeDriverManager().install()),
                    options=self.chrome_options
                )
                return driver
            except Exception as e:
                self.logger.error(f"Failed to create Chrome driver: {str(e)}")
                raise
        else:
            return self._driver_pool.pop()
    
    def _return_driver(self, driver: webdriver.Chrome):
        """Return driver to pool."""
        if len(self._driver_pool) < self._max_drivers:
            try:
                # Clear cookies and reset state
                driver.delete_all_cookies()
                self._driver_pool.append(driver)
            except Exception:
                try:
                    driver.quit()
                except Exception:
                    pass
        else:
            try:
                driver.quit()
            except Exception:
                pass
    
    async def _perform_search(self, query: SearchQuery) -> List[ResearchResult]:
        """
        Perform web scraping based on query.
        
        For web scraping, the query should contain URLs to scrape.
        
        Args:
            query: Search query object
            
        Returns:
            List of research results
        """
        try:
            # Extract URLs from query
            urls = self._extract_urls_from_query(query.query)
            
            if not urls:
                # If no URLs provided, return empty results
                self.logger.warning("No URLs found in query for web scraping")
                return []
            
            results = []
            
            for url in urls[:query.max_results]:
                try:
                    result = await self._scrape_url(url, query)
                    if result:
                        results.append(result)
                except Exception as e:
                    self.logger.warning(f"Failed to scrape {url}: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            self.logger.error(f"Web scraping failed: {str(e)}")
            raise
    
    def _extract_urls_from_query(self, query: str) -> List[str]:
        """Extract URLs from query string."""
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        urls = url_pattern.findall(query)
        return urls
    
    async def _scrape_url(self, url: str, query: SearchQuery) -> Optional[ResearchResult]:
        """
        Scrape content from a single URL.
        
        Args:
            url: URL to scrape
            query: Original search query
            
        Returns:
            Research result with extracted content
        """
        try:
            # Try static scraping first (faster)
            result = await self._scrape_static(url, query)
            
            # If static scraping fails or returns poor content, try dynamic
            if not result or len(result.content) < 200:
                self.logger.info(f"Static scraping yielded poor results for {url}, trying dynamic scraping")
                dynamic_result = await self._scrape_dynamic(url, query)
                if dynamic_result and len(dynamic_result.content) > len(result.content if result else ""):
                    result = dynamic_result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to scrape {url}: {str(e)}")
            return None
    
    async def _scrape_static(self, url: str, query: SearchQuery) -> Optional[ResearchResult]:
        """Scrape URL using static HTTP request."""
        try:
            session = await self._get_session()
            
            async with session.get(url) as response:
                if response.status != 200:
                    self.logger.warning(f"HTTP {response.status} for {url}")
                    return None
                
                html_content = await response.text()
                
                return self._extract_content_from_html(html_content, url, query.query)
        
        except Exception as e:
            self.logger.warning(f"Static scraping failed for {url}: {str(e)}")
            return None
    
    async def _scrape_dynamic(self, url: str, query: SearchQuery) -> Optional[ResearchResult]:
        """Scrape URL using Selenium for dynamic content."""
        driver = None
        try:
            driver = await self._get_driver()
            driver.get(url)
            
            # Wait for page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Additional wait for dynamic content
            await asyncio.sleep(2)
            
            html_content = driver.page_source
            
            return self._extract_content_from_html(html_content, url, query.query)
        
        except Exception as e:
            self.logger.warning(f"Dynamic scraping failed for {url}: {str(e)}")
            return None
        
        finally:
            if driver:
                self._return_driver(driver)
    
    def _extract_content_from_html(self, html_content: str, url: str, original_query: str) -> Optional[ResearchResult]:
        """Extract and clean content from HTML."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for selector in self.remove_selectors:
                for element in soup.select(selector):
                    element.decompose()
            
            # Remove comments
            for comment in soup.findAll(text=lambda text: isinstance(text, Comment)):
                comment.extract()
            
            # Extract main content
            content_element = self._find_main_content(soup)
            
            if not content_element:
                # Fallback to body
                content_element = soup.find('body') or soup
            
            # Extract text content
            text_content = self._extract_clean_text(content_element)
            
            if len(text_content) < 100:
                self.logger.warning(f"Insufficient content extracted from {url}")
                return None
            
            # Extract metadata
            metadata = self._extract_page_metadata(soup, url)
            
            # Extract key points
            key_points = self._extract_key_points_from_content(text_content)
            
            # Calculate confidence
            confidence = self._calculate_scraping_confidence(text_content, original_query, metadata)
            
            # Create summary
            summary = self._create_content_summary(text_content, metadata)
            
            return ResearchResult(
                content=text_content,
                metadata=metadata,
                summary=summary,
                key_points=key_points,
                confidence=confidence,
                raw_data={
                    'url': url,
                    'content_length': len(text_content),
                    'extraction_method': 'html_parsing',
                    'source': 'web_scraper'
                }
            )
            
        except Exception as e:
            self.logger.error(f"Content extraction failed for {url}: {str(e)}")
            return None
    
    def _find_main_content(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        """Find the main content element in the page."""
        # Try content selectors in order of preference
        for selector in self.content_selectors:
            elements = soup.select(selector)
            if elements:
                # Return the element with most text content
                best_element = max(elements, key=lambda e: len(e.get_text()))
                if len(best_element.get_text().strip()) > 200:
                    return best_element
        
        return None
    
    def _extract_clean_text(self, element: BeautifulSoup) -> str:
        """Extract and clean text from HTML element."""
        # Use readability for better content extraction
        try:
            html_str = str(element)
            doc = Document(html_str)
            clean_html = doc.summary()
            
            # Convert to text
            clean_text = self.html_converter.handle(clean_html)
            
        except Exception:
            # Fallback to simple text extraction
            clean_text = element.get_text()
        
        # Clean up text
        clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text)  # Multiple newlines
        clean_text = re.sub(r'[ \t]+', ' ', clean_text)  # Multiple spaces
        clean_text = clean_text.strip()
        
        return clean_text
    
    def _extract_page_metadata(self, soup: BeautifulSoup, url: str) -> SourceMetadata:
        """Extract metadata from page."""
        # Extract title
        title_element = soup.find('title')
        title = title_element.get_text().strip() if title_element else ""
        
        # Try Open Graph title
        if not title:
            og_title = soup.find('meta', property='og:title')
            title = og_title.get('content', '') if og_title else ""
        
        # Extract description
        description = ""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            description = meta_desc.get('content', '')
        
        # Try Open Graph description
        if not description:
            og_desc = soup.find('meta', property='og:description')
            description = og_desc.get('content', '') if og_desc else ""
        
        # Extract author
        author = None
        author_meta = soup.find('meta', attrs={'name': 'author'})
        if author_meta:
            author = author_meta.get('content')
        
        # Try other author patterns
        if not author:
            author_selectors = [
                '.author', '.byline', '[rel="author"]', '.post-author'
            ]
            for selector in author_selectors:
                author_element = soup.select_one(selector)
                if author_element:
                    author = author_element.get_text().strip()
                    break
        
        # Extract publication date
        publish_date = None
        date_selectors = [
            'meta[property="article:published_time"]',
            'meta[name="pubdate"]',
            'time[datetime]',
            '.published', '.date', '.post-date'
        ]
        
        for selector in date_selectors:
            date_element = soup.select_one(selector)
            if date_element:
                date_str = date_element.get('content') or date_element.get('datetime') or date_element.get_text()
                if date_str:
                    publish_date = self._parse_date(date_str)
                    if publish_date:
                        break
        
        # Extract domain
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Calculate credibility
        credibility = self._calculate_domain_credibility(domain)
        
        return SourceMetadata(
            url=url,
            title=title or f"Content from {domain}",
            source_type="web",
            domain=domain,
            author=author,
            publish_date=publish_date,
            credibility_score=credibility,
            language="en"
        )
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object."""
        try:
            # Common date formats
            import dateutil.parser
            return dateutil.parser.parse(date_str)
        except Exception:
            return None
    
    def _calculate_domain_credibility(self, domain: str) -> float:
        """Calculate credibility score for domain."""
        # High credibility domains
        high_credibility = [
            'wikipedia.org', 'britannica.com', 'nature.com', 'science.org',
            'nih.gov', 'cdc.gov', 'who.int', 'gov.uk', '.edu',
            'reuters.com', 'bbc.com', 'ap.org', 'npr.org'
        ]
        
        # Medium credibility domains
        medium_credibility = [
            'nytimes.com', 'washingtonpost.com', 'guardian.com',
            'economist.com', 'wsj.com', 'bloomberg.com'
        ]
        
        # Low credibility indicators
        low_credibility = [
            'blogspot.com', 'wordpress.com', 'medium.com'
        ]
        
        domain_lower = domain.lower()
        
        for pattern in high_credibility:
            if pattern in domain_lower:
                return 0.9
        
        for pattern in medium_credibility:
            if pattern in domain_lower:
                return 0.8
        
        for pattern in low_credibility:
            if pattern in domain_lower:
                return 0.4
        
        # Default credibility
        return 0.6
    
    def _extract_key_points_from_content(self, content: str) -> List[str]:
        """Extract key points from content."""
        if len(content) < 200:
            return []
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]
        
        # Take first few substantial paragraphs as key points
        key_points = []
        for paragraph in paragraphs[:5]:
            # Extract first sentence of paragraph
            sentences = re.split(r'[.!?]+', paragraph)
            if sentences:
                first_sentence = sentences[0].strip()
                if len(first_sentence) > 30:
                    key_points.append(first_sentence)
        
        return key_points[:4]  # Limit to 4 key points
    
    def _calculate_scraping_confidence(self, content: str, query: str, metadata: SourceMetadata) -> float:
        """Calculate confidence score for scraped content."""
        confidence = 0.0
        
        # Content length score (30%)
        content_length = len(content)
        if content_length > 2000:
            length_score = 0.3
        elif content_length > 1000:
            length_score = 0.25
        elif content_length > 500:
            length_score = 0.2
        else:
            length_score = 0.1
        
        confidence += length_score
        
        # Query relevance (40%)
        if query:
            query_terms = set(query.lower().split())
            content_lower = content.lower()
            title_lower = metadata.title.lower()
            
            # Title matches
            title_matches = sum(1 for term in query_terms if term in title_lower)
            title_score = min(title_matches / len(query_terms), 1.0) * 0.2
            
            # Content matches
            content_matches = sum(1 for term in query_terms if term in content_lower)
            content_score = min(content_matches / len(query_terms), 1.0) * 0.2
            
            confidence += title_score + content_score
        else:
            confidence += 0.3  # Default if no query
        
        # Domain credibility (30%)
        confidence += metadata.credibility_score * 0.3
        
        return min(confidence, 1.0)
    
    def _create_content_summary(self, content: str, metadata: SourceMetadata) -> str:
        """Create summary of scraped content."""
        # Extract first paragraph as summary
        paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 30]
        
        if paragraphs:
            summary = paragraphs[0]
            if len(summary) > 300:
                summary = summary[:300] + "..."
        else:
            summary = content[:300] + "..." if len(content) > 300 else content
        
        return summary
    
    async def scrape_urls(self, urls: List[str], max_concurrent: int = 3) -> List[ResearchResult]:
        """
        Scrape multiple URLs concurrently.
        
        Args:
            urls: List of URLs to scrape
            max_concurrent: Maximum concurrent scraping tasks
            
        Returns:
            List of research results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_with_semaphore(url: str) -> Optional[ResearchResult]:
            async with semaphore:
                query = SearchQuery(query=url)
                return await self._scrape_url(url, query)
        
        # Create tasks for all URLs
        tasks = [scrape_with_semaphore(url) for url in urls]
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and None results
        valid_results = []
        for result in results:
            if isinstance(result, ResearchResult):
                valid_results.append(result)
            elif isinstance(result, Exception):
                self.logger.warning(f"Scraping task failed: {str(result)}")
        
        return valid_results
    
    async def extract_links_from_page(self, url: str, filter_domain: bool = True) -> List[str]:
        """
        Extract all links from a webpage.
        
        Args:
            url: URL to extract links from
            filter_domain: Whether to only return links from same domain
            
        Returns:
            List of URLs found on the page
        """
        try:
            session = await self._get_session()
            
            async with session.get(url) as response:
                if response.status != 200:
                    return []
                
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'html.parser')
                
                links = []
                base_domain = urlparse(url).netloc if filter_domain else None
                
                for link_element in soup.find_all('a', href=True):
                    href = link_element['href']
                    
                    # Convert relative URLs to absolute
                    absolute_url = urljoin(url, href)
                    
                    # Filter by domain if requested
                    if filter_domain and base_domain:
                        link_domain = urlparse(absolute_url).netloc
                        if link_domain != base_domain:
                            continue
                    
                    # Basic URL validation
                    if absolute_url.startswith(('http://', 'https://')):
                        links.append(absolute_url)
                
                return list(set(links))  # Remove duplicates
        
        except Exception as e:
            self.logger.error(f"Failed to extract links from {url}: {str(e)}")
            return []
    
    def cleanup_drivers(self):
        """Clean up all Selenium drivers."""
        for driver in self._driver_pool:
            try:
                driver.quit()
            except Exception:
                pass
        self._driver_pool.clear()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session and not self._session.closed:
            await self._session.close()
        self.cleanup_drivers()
    
    def __del__(self):
        """Cleanup on deletion."""
        if self._session and not self._session.closed:
            asyncio.create_task(self._session.close())
        self.cleanup_drivers()