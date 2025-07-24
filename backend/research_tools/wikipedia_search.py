"""
Wikipedia API wrapper for encyclopedic research.

Provides intelligent querying of Wikipedia with:
- Smart disambiguation handling
- Multi-language support
- Section extraction
- Link following
- Content summarization
"""

import asyncio
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from urllib.parse import quote_plus

import aiohttp
import wikipedia

from .base_research_tool import BaseResearchTool, ResearchResult, SourceMetadata, SearchQuery


class WikipediaSearchTool(BaseResearchTool):
    """
    Wikipedia search tool with intelligent querying capabilities.
    
    Features:
    - Disambiguation handling
    - Section extraction
    - Multi-language support
    - Related articles discovery
    - Content summarization
    - Link following for deep research
    """
    
    def __init__(self):
        super().__init__(
            name="wikipedia_search",
            description="Search Wikipedia for encyclopedic information",
            source_type="reference",
            max_requests_per_minute=100  # Wikipedia is very permissive
        )
        
        # Wikipedia API configuration
        self.api_url = "https://en.wikipedia.org/api/rest_v1"
        self.wiki_url = "https://en.wikipedia.org/wiki"
        
        # Supported languages
        self.supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'zh']
        self.current_language = 'en'
        
        # Session for connection pooling
        self._session = None
        
        # Configure wikipedia library
        wikipedia.set_rate_limiting(True)
        wikipedia.set_lang(self.current_language)
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def _perform_search(self, query: SearchQuery) -> List[ResearchResult]:
        """
        Perform Wikipedia search with intelligent disambiguation.
        
        Args:
            query: Search query object
            
        Returns:
            List of research results
        """
        try:
            # Set language if specified
            if query.language in self.supported_languages:
                self.current_language = query.language
                wikipedia.set_lang(self.current_language)
            
            results = []
            
            # Search for articles
            search_results = await self._search_articles(query.query, query.max_results)
            
            # Process each search result
            for article_title in search_results[:query.max_results]:
                try:
                    article_result = await self._get_article_content(article_title, query.query)
                    if article_result:
                        results.append(article_result)
                        
                        # If we need more results, get related articles
                        if len(results) < query.max_results:
                            related_results = await self._get_related_articles(
                                article_title, 
                                query.query,
                                max_related=min(3, query.max_results - len(results))
                            )
                            results.extend(related_results)
                
                except Exception as e:
                    self.logger.warning(f"Failed to process article '{article_title}': {str(e)}")
                    continue
            
            return results[:query.max_results]
            
        except Exception as e:
            self.logger.error(f"Wikipedia search failed: {str(e)}")
            raise
    
    async def _search_articles(self, query: str, max_results: int = 10) -> List[str]:
        """Search for Wikipedia articles."""
        try:
            # Use wikipedia library for initial search
            search_results = wikipedia.search(query, results=max_results * 2)
            
            # Filter out disambiguation pages and redirect
            filtered_results = []
            for title in search_results:
                if not self._is_disambiguation_page(title):
                    filtered_results.append(title)
                    if len(filtered_results) >= max_results:
                        break
            
            return filtered_results
            
        except Exception as e:
            self.logger.warning(f"Wikipedia search failed: {str(e)}")
            return []
    
    async def _get_article_content(self, title: str, original_query: str) -> Optional[ResearchResult]:
        """Get full content of a Wikipedia article."""
        try:
            # Get page object
            page = wikipedia.page(title)
            
            # Create metadata
            metadata = SourceMetadata(
                url=page.url,
                title=page.title,
                source_type="reference",
                domain="wikipedia.org",
                credibility_score=0.95,  # Wikipedia is highly credible
                language=self.current_language
            )
            
            # Get summary and content
            summary = page.summary
            full_content = page.content
            
            # Extract key sections that are most relevant
            relevant_sections = self._extract_relevant_sections(full_content, original_query)
            
            # Create key points from sections
            key_points = self._extract_key_points(summary, relevant_sections)
            
            # Determine confidence based on content relevance
            confidence = self._calculate_content_relevance(original_query, full_content, page.title)
            
            return ResearchResult(
                content=full_content,
                metadata=metadata,
                summary=summary,
                key_points=key_points,
                confidence=confidence,
                raw_data={
                    'page_id': page.pageid,
                    'categories': getattr(page, 'categories', []),
                    'links': getattr(page, 'links', [])[:20],  # First 20 links
                    'references': getattr(page, 'references', [])[:10],  # First 10 references
                    'sections': relevant_sections,
                    'language': self.current_language
                }
            )
            
        except wikipedia.exceptions.DisambiguationError as e:
            # Handle disambiguation by picking the most relevant option
            self.logger.info(f"Disambiguation for '{title}': {len(e.options)} options found")
            return await self._handle_disambiguation(e.options, original_query)
        
        except wikipedia.exceptions.PageError:
            self.logger.warning(f"Page not found: {title}")
            return None
        
        except Exception as e:
            self.logger.warning(f"Failed to get article content for '{title}': {str(e)}")
            return None
    
    async def _handle_disambiguation(self, options: List[str], original_query: str) -> Optional[ResearchResult]:
        """Handle disambiguation by selecting the most relevant option."""
        try:
            # Score each option based on relevance to original query
            scored_options = []
            query_lower = original_query.lower()
            
            for option in options[:10]:  # Limit to first 10 options
                score = 0
                option_lower = option.lower()
                
                # Exact match gets highest score
                if query_lower == option_lower:
                    score = 100
                # Partial match in title
                elif query_lower in option_lower:
                    score = 80
                # Query terms in title
                else:
                    query_terms = query_lower.split()
                    matches = sum(1 for term in query_terms if term in option_lower)
                    score = (matches / len(query_terms)) * 60
                
                scored_options.append((option, score))
            
            # Sort by score and pick the best match
            scored_options.sort(key=lambda x: x[1], reverse=True)
            
            if scored_options and scored_options[0][1] > 20:  # Minimum relevance threshold
                best_option = scored_options[0][0]
                self.logger.info(f"Selected disambiguation option: {best_option}")
                return await self._get_article_content(best_option, original_query)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to handle disambiguation: {str(e)}")
            return None
    
    async def _get_related_articles(self, article_title: str, original_query: str, max_related: int = 3) -> List[ResearchResult]:
        """Get related articles using Wikipedia's link structure."""
        results = []
        
        try:
            # Get the page to access its links
            page = wikipedia.page(article_title)
            links = getattr(page, 'links', [])
            
            # Filter links to find most relevant ones
            relevant_links = self._filter_relevant_links(links, original_query)
            
            # Get content for top related articles
            for link in relevant_links[:max_related]:
                try:
                    related_result = await self._get_article_content(link, original_query)
                    if related_result:
                        # Mark as related article
                        related_result.raw_data['is_related'] = True
                        related_result.raw_data['parent_article'] = article_title
                        results.append(related_result)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to get related article '{link}': {str(e)}")
                    continue
        
        except Exception as e:
            self.logger.warning(f"Failed to get related articles for '{article_title}': {str(e)}")
        
        return results
    
    def _filter_relevant_links(self, links: List[str], query: str) -> List[str]:
        """Filter links based on relevance to query."""
        query_terms = set(query.lower().split())
        relevant_links = []
        
        for link in links:
            link_lower = link.lower()
            link_terms = set(link_lower.split())
            
            # Calculate overlap between query terms and link terms
            overlap = len(query_terms & link_terms)
            
            if overlap > 0:
                relevant_links.append((link, overlap))
        
        # Sort by relevance and return top links
        relevant_links.sort(key=lambda x: x[1], reverse=True)
        return [link for link, _ in relevant_links[:10]]  # Top 10 relevant links
    
    def _extract_relevant_sections(self, content: str, query: str) -> Dict[str, str]:
        """Extract sections most relevant to the query."""
        sections = {}
        query_terms = set(query.lower().split())
        
        # Split content into sections (assuming == Section == format)
        section_pattern = r'\n\n==+\s*(.+?)\s*==+\n\n'
        parts = re.split(section_pattern, content)
        
        if len(parts) > 1:
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    section_title = parts[i].strip()
                    section_content = parts[i + 1].strip()
                    
                    # Check if section is relevant to query
                    section_lower = (section_title + " " + section_content).lower()
                    relevance = sum(1 for term in query_terms if term in section_lower)
                    
                    if relevance > 0 and len(section_content) > 100:
                        sections[section_title] = section_content
        
        # If no sections found, return first few paragraphs
        if not sections:
            paragraphs = content.split('\n\n')[:3]
            sections['Introduction'] = '\n\n'.join(paragraphs)
        
        return sections
    
    def _extract_key_points(self, summary: str, sections: Dict[str, str]) -> List[str]:
        """Extract key points from summary and relevant sections."""
        key_points = []
        
        # Extract from summary
        summary_sentences = re.split(r'[.!?]+', summary)
        key_points.extend([s.strip() for s in summary_sentences if len(s.strip()) > 20][:3])
        
        # Extract from sections
        for section_title, section_content in sections.items():
            # Get first sentence of each section as a key point
            first_sentence = re.split(r'[.!?]+', section_content)[0].strip()
            if len(first_sentence) > 20:
                key_points.append(f"{section_title}: {first_sentence}")
        
        return key_points[:8]  # Limit to 8 key points
    
    def _calculate_content_relevance(self, query: str, content: str, title: str) -> float:
        """Calculate how relevant the content is to the query."""
        query_terms = set(query.lower().split())
        content_lower = content.lower()
        title_lower = title.lower()
        
        # Count term matches in content
        content_matches = sum(1 for term in query_terms if term in content_lower)
        content_score = min(content_matches / len(query_terms), 1.0) * 0.6
        
        # Count term matches in title (weighted higher)
        title_matches = sum(1 for term in query_terms if term in title_lower)
        title_score = min(title_matches / len(query_terms), 1.0) * 0.4
        
        return min(content_score + title_score, 1.0)
    
    def _is_disambiguation_page(self, title: str) -> bool:
        """Check if a page is a disambiguation page."""
        disambiguation_indicators = [
            '(disambiguation)', 'disambiguation', 'may refer to',
            'can refer to', 'could refer to'
        ]
        
        title_lower = title.lower()
        return any(indicator in title_lower for indicator in disambiguation_indicators)
    
    async def search_by_category(self, category: str, max_results: int = 10) -> List[ResearchResult]:
        """
        Search Wikipedia articles by category.
        
        Args:
            category: Wikipedia category name
            max_results: Maximum number of results
            
        Returns:
            List of research results from the category
        """
        try:
            session = await self._get_session()
            
            # Use Wikipedia API to get category members
            url = f"https://en.wikipedia.org/api/rest_v1/page/list/{quote_plus(category)}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    results = []
                    articles = data.get('items', [])
                    
                    for article in articles[:max_results]:
                        article_title = article.get('title', '')
                        if article_title:
                            result = await self._get_article_content(article_title, category)
                            if result:
                                results.append(result)
                    
                    return results
            
            return []
            
        except Exception as e:
            self.logger.error(f"Category search failed: {str(e)}")
            return []
    
    async def get_page_sections(self, title: str) -> Dict[str, str]:
        """
        Get all sections of a Wikipedia page.
        
        Args:
            title: Page title
            
        Returns:
            Dictionary of section_title -> section_content
        """
        try:
            page = wikipedia.page(title)
            return self._extract_relevant_sections(page.content, title)
        
        except Exception as e:
            self.logger.error(f"Failed to get page sections: {str(e)}")
            return {}
    
    def set_language(self, language: str):
        """
        Set the language for Wikipedia searches.
        
        Args:
            language: Language code (e.g., 'en', 'es', 'fr')
        """
        if language in self.supported_languages:
            self.current_language = language
            wikipedia.set_lang(language)
            self.logger.info(f"Set Wikipedia language to: {language}")
        else:
            self.logger.warning(f"Unsupported language: {language}")
    
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