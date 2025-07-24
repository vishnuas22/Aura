"""
arXiv API integration for academic research.

Provides access to arXiv's repository of scientific papers with:
- Subject category filtering
- Date range searching  
- Author-based searches
- Full-text PDF access
- Citation formatting
"""

import asyncio
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus, urlencode

import aiohttp
import arxiv
from xml.etree import ElementTree as ET

from .base_research_tool import BaseResearchTool, ResearchResult, SourceMetadata, SearchQuery


class ArxivSearchTool(BaseResearchTool):
    """
    arXiv search tool for academic paper research.
    
    Features:
    - Subject category filtering
    - Author-based searches
    - Date range filtering
    - Full abstract extraction
    - PDF link provision
    - Citation metadata
    - Impact scoring
    """
    
    def __init__(self):
        super().__init__(
            name="arxiv_search",
            description="Search arXiv for academic papers and preprints",
            source_type="academic",
            max_requests_per_minute=30  # arXiv rate limit
        )
        
        # arXiv API configuration
        self.api_url = "http://export.arxiv.org/api/query"
        self.base_url = "https://arxiv.org"
        
        # Subject categories mapping
        self.subject_categories = {
            'cs': 'Computer Science',
            'math': 'Mathematics', 
            'physics': 'Physics',
            'astro-ph': 'Astrophysics',
            'cond-mat': 'Condensed Matter',
            'gr-qc': 'General Relativity and Quantum Cosmology',
            'hep-ex': 'High Energy Physics - Experiment',
            'hep-lat': 'High Energy Physics - Lattice',
            'hep-ph': 'High Energy Physics - Phenomenology',
            'hep-th': 'High Energy Physics - Theory',
            'math-ph': 'Mathematical Physics',
            'nlin': 'Nonlinear Sciences',
            'nucl-ex': 'Nuclear Experiment',
            'nucl-th': 'Nuclear Theory',
            'q-bio': 'Quantitative Biology',
            'q-fin': 'Quantitative Finance',
            'quant-ph': 'Quantum Physics',
            'stat': 'Statistics',
            'econ': 'Economics',
            'eess': 'Electrical Engineering and Systems Science'
        }
        
        # Session for connection pooling
        self._session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=60)  # arXiv can be slow
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def _perform_search(self, query: SearchQuery) -> List[ResearchResult]:
        """
        Perform arXiv search using the API.
        
        Args:
            query: Search query object
            
        Returns:
            List of research results
        """
        try:
            # Build search parameters
            search_params = self._build_search_params(query)
            
            # Make API request
            results = await self._make_api_request(search_params)
            
            # Process results
            processed_results = []
            for result in results:
                processed_result = self._process_arxiv_entry(result, query.query)
                if processed_result:
                    processed_results.append(processed_result)
            
            return processed_results[:query.max_results]
            
        except Exception as e:
            self.logger.error(f"arXiv search failed: {str(e)}")
            raise
    
    def _build_search_params(self, query: SearchQuery) -> Dict[str, Any]:
        """Build search parameters for arXiv API."""
        params = {
            'search_query': query.query,
            'start': 0,
            'max_results': min(query.max_results, 50),  # arXiv limit
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        # Handle filters
        filters = query.filters or {}
        
        # Subject category filter
        if 'category' in filters:
            category = filters['category']
            if category in self.subject_categories:
                params['search_query'] = f"cat:{category} AND ({query.query})"
        
        # Author filter
        if 'author' in filters:
            author = filters['author']
            params['search_query'] = f"au:{author} AND ({query.query})"
        
        # Date range filter
        if query.date_range:
            date_filter = self._build_date_filter(query.date_range)
            if date_filter:
                params['search_query'] = f"({params['search_query']}) AND {date_filter}"
        
        # Sort options
        if query.sort_by == 'date':
            params['sortBy'] = 'submittedDate'
        elif query.sort_by == 'updated':
            params['sortBy'] = 'lastUpdatedDate'
        
        return params
    
    def _build_date_filter(self, date_range: str) -> Optional[str]:
        """Build date filter for arXiv search."""
        now = datetime.now()
        
        if date_range == 'day':
            start_date = now - timedelta(days=1)
        elif date_range == 'week':
            start_date = now - timedelta(weeks=1)
        elif date_range == 'month':  
            start_date = now - timedelta(days=30)
        elif date_range == 'year':
            start_date = now - timedelta(days=365)
        else:
            return None
        
        # Format: YYYYMMDDHHMM
        date_str = start_date.strftime('%Y%m%d%H%M')
        return f"submittedDate:[{date_str}* TO *]"
    
    async def _make_api_request(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Make request to arXiv API."""
        session = await self._get_session()
        
        url = f"{self.api_url}?{urlencode(params)}"
        
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"arXiv API request failed: {response.status}")
            
            xml_content = await response.text()
            return self._parse_xml_response(xml_content)
    
    def _parse_xml_response(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse XML response from arXiv API."""
        results = []
        
        try:
            root = ET.fromstring(xml_content)
            
            # Namespace handling
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            # Find all entry elements
            entries = root.findall('atom:entry', namespaces)
            
            for entry in entries:
                try:
                    result = self._parse_entry(entry, namespaces)
                    if result:
                        results.append(result)
                except Exception as e:
                    self.logger.warning(f"Failed to parse entry: {str(e)}")
                    continue
            
        except ET.ParseError as e:
            self.logger.error(f"Failed to parse XML response: {str(e)}")
        
        return results
    
    def _parse_entry(self, entry: ET.Element, namespaces: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Parse individual entry from arXiv response."""
        try:
            # Extract basic information
            title_elem = entry.find('atom:title', namespaces)
            title = title_elem.text.strip() if title_elem is not None else ""
            
            summary_elem = entry.find('atom:summary', namespaces)
            summary = summary_elem.text.strip() if summary_elem is not None else ""
            
            # Extract ID and URL
            id_elem = entry.find('atom:id', namespaces)
            arxiv_id = id_elem.text.strip() if id_elem is not None else ""
            
            # Extract authors
            authors = []
            author_elems = entry.findall('atom:author', namespaces)
            for author_elem in author_elems:
                name_elem = author_elem.find('atom:name', namespaces)
                if name_elem is not None:
                    authors.append(name_elem.text.strip())
            
            # Extract categories
            categories = []
            category_elems = entry.findall('atom:category', namespaces)
            for cat_elem in category_elems:
                term = cat_elem.get('term', '')
                if term:
                    categories.append(term)
            
            # Extract dates
            published_elem = entry.find('atom:published', namespaces)
            published_date = None
            if published_elem is not None:
                try:
                    published_date = datetime.fromisoformat(
                        published_elem.text.replace('Z', '+00:00')
                    )
                except ValueError:
                    pass
            
            updated_elem = entry.find('atom:updated', namespaces)
            updated_date = None
            if updated_elem is not None:
                try:
                    updated_date = datetime.fromisoformat(
                        updated_elem.text.replace('Z', '+00:00')
                    )
                except ValueError:
                    pass
            
            # Extract DOI if available
            doi = None
            doi_elem = entry.find('arxiv:doi', namespaces)
            if doi_elem is not None:
                doi = doi_elem.text.strip()
            
            # Extract journal reference if available
            journal_ref = None
            journal_elem = entry.find('arxiv:journal_ref', namespaces)
            if journal_elem is not None:
                journal_ref = journal_elem.text.strip()
            
            # Extract PDF link
            pdf_url = None
            link_elems = entry.findall('atom:link', namespaces)
            for link_elem in link_elems:
                if link_elem.get('type') == 'application/pdf':
                    pdf_url = link_elem.get('href')
                    break
            
            return {
                'title': title,
                'summary': summary,
                'authors': authors,
                'categories': categories,
                'arxiv_id': arxiv_id,
                'published_date': published_date,
                'updated_date': updated_date,
                'doi': doi,
                'journal_ref': journal_ref,
                'pdf_url': pdf_url
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to parse entry: {str(e)}")
            return None
    
    def _process_arxiv_entry(self, entry: Dict[str, Any], original_query: str) -> Optional[ResearchResult]:
        """Convert arXiv entry to ResearchResult."""
        try:
            # Extract arXiv ID from full ID URL
            arxiv_id = entry['arxiv_id']
            if '/' in arxiv_id:
                arxiv_id = arxiv_id.split('/')[-1]
            
            # Create URL
            url = f"{self.base_url}/abs/{arxiv_id}"
            
            # Create metadata
            metadata = SourceMetadata(
                url=url,
                title=entry['title'],
                source_type="academic",
                domain="arxiv.org",
                author=', '.join(entry['authors'][:3]) if entry['authors'] else None,
                publish_date=entry['published_date'],
                credibility_score=0.85,  # arXiv is peer-reviewed
                language="en"
            )
            
            # Extract key points from abstract
            key_points = self._extract_key_points_from_abstract(entry['summary'])
            
            # Calculate confidence based on relevance and recency
            confidence = self._calculate_paper_confidence(entry, original_query)
            
            # Create research result
            return ResearchResult(
                content=entry['summary'],
                metadata=metadata,
                summary=self._create_paper_summary(entry),
                key_points=key_points,
                confidence=confidence,
                raw_data={
                    'arxiv_id': arxiv_id,
                    'authors': entry['authors'],
                    'categories': entry['categories'],
                    'doi': entry['doi'],
                    'journal_ref': entry['journal_ref'],
                    'pdf_url': entry['pdf_url'],
                    'published_date': entry['published_date'].isoformat() if entry['published_date'] else None,
                    'updated_date': entry['updated_date'].isoformat() if entry['updated_date'] else None
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to process arXiv entry: {str(e)}")
            return None
    
    def _extract_key_points_from_abstract(self, abstract: str) -> List[str]:
        """Extract key points from paper abstract."""
        # Split abstract into sentences
        sentences = re.split(r'[.!?]+', abstract)
        
        # Key phrases that often indicate important points
        key_indicators = [
            'we propose', 'we present', 'we show', 'we demonstrate',
            'we introduce', 'we develop', 'our approach', 'our method',
            'results show', 'we find', 'conclusion', 'achieve', 'improve'
        ]
        
        key_points = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            sentence_lower = sentence.lower()
            
            # Check if sentence contains key indicators
            if any(indicator in sentence_lower for indicator in key_indicators):
                key_points.append(sentence)
            # Or if it's a substantial sentence (for mathematical papers)
            elif len(sentence) > 50 and any(word in sentence_lower for word in ['theorem', 'algorithm', 'model', 'framework']):
                key_points.append(sentence)
        
        # If no key indicators found, take first few substantial sentences
        if not key_points:
            substantial_sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
            key_points = substantial_sentences[:3]
        
        return key_points[:5]  # Limit to 5 key points
    
    def _create_paper_summary(self, entry: Dict[str, Any]) -> str:
        """Create a summary of the paper."""
        authors_str = ', '.join(entry['authors'][:3])
        if len(entry['authors']) > 3:
            authors_str += ' et al.'
        
        summary_parts = [f"Authors: {authors_str}"]
        
        if entry['categories']:
            categories_str = ', '.join(entry['categories'][:3])
            summary_parts.append(f"Categories: {categories_str}")
        
        if entry['published_date']:
            date_str = entry['published_date'].strftime('%Y-%m-%d')
            summary_parts.append(f"Published: {date_str}")
        
        if entry['journal_ref']:
            summary_parts.append(f"Journal: {entry['journal_ref']}")
        
        # Add first sentence of abstract
        abstract_sentences = re.split(r'[.!?]+', entry['summary'])
        if abstract_sentences:
            first_sentence = abstract_sentences[0].strip()
            if len(first_sentence) > 20:
                summary_parts.append(f"Abstract: {first_sentence}...")
        
        return ' | '.join(summary_parts)
    
    def _calculate_paper_confidence(self, entry: Dict[str, Any], query: str) -> float:
        """Calculate confidence score for the paper."""
        score = 0.0
        query_terms = set(query.lower().split())
        
        # Title relevance (40%)
        title_lower = entry['title'].lower()
        title_matches = sum(1 for term in query_terms if term in title_lower)
        title_score = min(title_matches / len(query_terms), 1.0) * 0.4
        score += title_score
        
        # Abstract relevance (30%)
        abstract_lower = entry['summary'].lower()
        abstract_matches = sum(1 for term in query_terms if term in abstract_lower)
        abstract_score = min(abstract_matches / len(query_terms), 1.0) * 0.3
        score += abstract_score
        
        # Recency bonus (20%)
        if entry['published_date']:
            days_old = (datetime.now() - entry['published_date'].replace(tzinfo=None)).days
            if days_old < 30:
                recency_score = 0.2
            elif days_old < 365:
                recency_score = 0.15
            elif days_old < 1825:  # 5 years
                recency_score = 0.1
            else:
                recency_score = 0.05
            score += recency_score
        
        # Journal publication bonus (10%)
        if entry['journal_ref'] or entry['doi']:
            score += 0.1
        
        return min(score, 1.0)
    
    async def search_by_author(self, author: str, max_results: int = 10) -> List[ResearchResult]:
        """
        Search papers by author.
        
        Args:
            author: Author name
            max_results: Maximum number of results
            
        Returns:
            List of research results
        """
        search_query = SearchQuery(
            query="",  # Empty query, will be overridden by author filter
            max_results=max_results,
            filters={'author': author}
        )
        
        return await self.search(search_query)
    
    async def search_by_category(self, category: str, query: str = "", max_results: int = 10) -> List[ResearchResult]:
        """
        Search papers by subject category.
        
        Args:
            category: Subject category (e.g., 'cs', 'math', 'physics')
            query: Additional query terms
            max_results: Maximum number of results
            
        Returns:
            List of research results
        """
        search_query = SearchQuery(
            query=query or category,
            max_results=max_results,
            filters={'category': category}
        )
        
        return await self.search(search_query)
    
    def get_supported_categories(self) -> Dict[str, str]:
        """Get supported subject categories."""
        return self.subject_categories.copy()
    
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