"""
Reddit search integration for community insights.

Provides access to Reddit discussions with:
- Subreddit-specific searches
- Comment thread analysis
- Sentiment and popularity scoring
- User expertise detection
- Real-time trending topics
"""

import asyncio
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus

import aiohttp
import praw
from praw.exceptions import RedditAPIException

from .base_research_tool import BaseResearchTool, ResearchResult, SourceMetadata, SearchQuery


class RedditSearchTool(BaseResearchTool):
    """
    Reddit search tool for community insights and discussions.
    
    Features:
    - Subreddit-specific searches
    - Comment analysis and threading
    - Vote-based quality scoring
    - User credibility assessment
    - Trending topic detection
    - Sentiment analysis
    """
    
    def __init__(self):
        super().__init__(
            name="reddit_search",
            description="Search Reddit for community insights and discussions",
            source_type="social",
            max_requests_per_minute=60  # Reddit rate limit
        )
        
        # Reddit API credentials (optional - can work without)
        self.client_id = os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.user_agent = os.getenv('REDDIT_USER_AGENT', 'research-tool:v1.0 (by /u/research-bot)')
        
        # Reddit instance
        self.reddit = None
        self._init_reddit_client()
        
        # Subreddit credibility and relevance mapping
        self.subreddit_credibility = {
            'askscience': 0.95,
            'science': 0.9,
            'technology': 0.85,
            'programming': 0.85,
            'machinelearning': 0.85,
            'askhistorians': 0.95,
            'history': 0.8,
            'economics': 0.8,
            'politics': 0.6,  # Lower due to bias potential
            'worldnews': 0.7,
            'news': 0.7,
            'todayilearned': 0.75,
            'explainlikeimfive': 0.8,
            'dataisbeautiful': 0.8,
            'personalfinance': 0.8,
            'legaladvice': 0.7,  # Varies greatly
            'relationships': 0.6,
            'tifu': 0.5,
            'memes': 0.3,
            'funny': 0.3
        }
        
        # Common high-quality subreddits by topic
        self.topic_subreddits = {
            'technology': ['technology', 'programming', 'MachineLearning', 'artificial', 'compsci'],
            'science': ['science', 'askscience', 'Physics', 'chemistry', 'biology'],
            'history': ['history', 'AskHistorians', 'todayilearned'],
            'politics': ['politics', 'worldnews', 'news'],
            'economics': ['economics', 'personalfinance', 'investing'],
            'education': ['education', 'AskAcademia', 'GradSchool'],
            'health': ['health', 'medicine', 'askdocs'],
            'psychology': ['psychology', 'askpsychology']
        }
        
        # Session for HTTP requests (fallback)
        self._session = None
    
    def _init_reddit_client(self):
        """Initialize Reddit API client if credentials available."""
        try:
            if self.client_id and self.client_secret:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent,
                    check_for_async=False
                )
                self.logger.info("Reddit API client initialized")
            else:
                self.logger.info("Reddit API credentials not found, using HTTP fallback")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Reddit client: {str(e)}")
            self.reddit = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session for Reddit API fallback."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                'User-Agent': self.user_agent
            }
            self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self._session
    
    async def _perform_search(self, query: SearchQuery) -> List[ResearchResult]:
        """
        Perform Reddit search using available methods.
        
        Args:
            query: Search query object
            
        Returns:
            List of research results
        """
        try:
            results = []
            
            # Try PRAW first if available
            if self.reddit:
                praw_results = await self._search_with_praw(query)
                results.extend(praw_results)
            
            # Use HTTP API as fallback or supplement
            if len(results) < query.max_results:
                http_results = await self._search_with_http(query)
                results.extend(http_results)
            
            # Remove duplicates and sort
            results = self._remove_duplicate_posts(results)
            results.sort(key=lambda r: (r.metadata.relevance_score, r.confidence), reverse=True)
            
            return results[:query.max_results]
            
        except Exception as e:
            self.logger.error(f"Reddit search failed: {str(e)}")
            raise
    
    async def _search_with_praw(self, query: SearchQuery) -> List[ResearchResult]:
        """Search using PRAW (Python Reddit API Wrapper)."""
        results = []
        
        try:
            # Determine relevant subreddits
            subreddits = self._get_relevant_subreddits(query.query, query.filters)
            
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Search within subreddit
                    search_results = subreddit.search(
                        query.query,
                        sort='relevance' if query.sort_by == 'relevance' else 'new',
                        time_filter=self._get_time_filter(query.date_range),
                        limit=min(query.max_results // len(subreddits) + 2, 25)
                    )
                    
                    for submission in search_results:
                        result = self._process_praw_submission(submission, query.query)
                        if result:
                            results.append(result)
                            
                except Exception as e:
                    self.logger.warning(f"Failed to search subreddit {subreddit_name}: {str(e)}")
                    continue
            
        except Exception as e:
            self.logger.warning(f"PRAW search failed: {str(e)}")
        
        return results
    
    async def _search_with_http(self, query: SearchQuery) -> List[ResearchResult]:
        """Search using Reddit HTTP API."""
        results = []
        
        try:
            session = await self._get_session()
            
            # Build search URL
            params = {
                'q': query.query,
                'sort': 'relevance' if query.sort_by == 'relevance' else 'new',
                'limit': min(query.max_results, 100),
                'type': 'link'
            }
            
            # Add time filter
            if query.date_range:
                params['t'] = self._get_time_filter(query.date_range)
            
            # Search specific subreddits if specified
            subreddits = query.filters.get('subreddits', []) if query.filters else []
            if not subreddits:
                subreddits = self._get_relevant_subreddits(query.query, query.filters)
            
            for subreddit in subreddits[:5]:  # Limit to 5 subreddits
                try:
                    # Add subreddit restriction
                    subreddit_params = params.copy()
                    subreddit_params['restrict_sr'] = 'on'
                    
                    url = f"https://www.reddit.com/r/{subreddit}/search.json"
                    
                    async with session.get(url, params=subreddit_params) as response:
                        if response.status == 200:
                            data = await response.json()
                            subreddit_results = self._process_http_response(data, query.query, subreddit)
                            results.extend(subreddit_results)
                
                except Exception as e:
                    self.logger.warning(f"HTTP search failed for r/{subreddit}: {str(e)}")
                    continue
        
        except Exception as e:
            self.logger.warning(f"Reddit HTTP search failed: {str(e)}")
        
        return results
    
    def _get_relevant_subreddits(self, query: str, filters: Optional[Dict[str, Any]]) -> List[str]:
        """Get relevant subreddits based on query content."""
        if filters and 'subreddits' in filters:
            return filters['subreddits']
        
        query_lower = query.lower()
        relevant_subreddits = []
        
        # Check topic-specific subreddits
        for topic, subreddits in self.topic_subreddits.items():
            if topic in query_lower or any(keyword in query_lower for keyword in [
                'tech', 'computer', 'programming', 'ai', 'ml', 'science',
                'history', 'politics', 'economics', 'finance', 'education',
                'health', 'medicine', 'psychology'
            ]):
                relevant_subreddits.extend(subreddits[:2])  # Top 2 from each category
        
        # Default high-quality subreddits if no specific matches
        if not relevant_subreddits:
            relevant_subreddits = ['askreddit', 'todayilearned', 'explainlikeimfive', 'science']
        
        return relevant_subreddits[:8]  # Limit to 8 subreddits
    
    def _get_time_filter(self, date_range: Optional[str]) -> str:
        """Convert date range to Reddit time filter."""
        if not date_range:
            return 'all'
        
        mapping = {
            'day': 'day',
            'week': 'week', 
            'month': 'month',
            'year': 'year'
        }
        
        return mapping.get(date_range, 'all')
    
    def _process_praw_submission(self, submission, original_query: str) -> Optional[ResearchResult]:
        """Process PRAW submission object."""
        try:
            # Extract basic information
            title = submission.title
            selftext = getattr(submission, 'selftext', '') or ''
            url = f"https://reddit.com{submission.permalink}"
            
            # Get subreddit credibility
            subreddit_name = submission.subreddit.display_name.lower()
            credibility = self.subreddit_credibility.get(subreddit_name, 0.5)
            
            # Create metadata
            metadata = SourceMetadata(
                url=url,
                title=title,
                source_type="social",
                domain="reddit.com",
                author=f"u/{submission.author.name}" if submission.author else "[deleted]",
                publish_date=datetime.fromtimestamp(submission.created_utc),
                credibility_score=credibility,
                language="en"
            )
            
            # Combine title and content for analysis
            full_content = f"{title}\n\n{selftext}" if selftext else title
            
            # Extract key points from comments (top comments)
            key_points = self._extract_key_points_from_comments(submission)
            
            # Calculate confidence based on votes and relevance
            confidence = self._calculate_reddit_confidence(
                title, selftext, original_query, submission.score, 
                submission.num_comments, credibility
            )
            
            return ResearchResult(
                content=full_content,
                metadata=metadata,
                summary=self._create_reddit_summary(submission),
                key_points=key_points,
                confidence=confidence,
                raw_data={
                    'subreddit': subreddit_name,
                    'score': submission.score,
                    'upvote_ratio': getattr(submission, 'upvote_ratio', 0),
                    'num_comments': submission.num_comments,
                    'created_utc': submission.created_utc,
                    'url': submission.url if hasattr(submission, 'url') else url,
                    'is_self': submission.is_self,
                    'source': 'praw'
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to process PRAW submission: {str(e)}")
            return None
    
    def _process_http_response(self, data: Dict[str, Any], original_query: str, subreddit: str) -> List[ResearchResult]:
        """Process HTTP API response."""
        results = []
        
        try:
            posts = data.get('data', {}).get('children', [])
            
            for post_data in posts:
                post = post_data.get('data', {})
                
                try:
                    result = self._process_http_post(post, original_query, subreddit)
                    if result:
                        results.append(result)
                except Exception as e:
                    self.logger.warning(f"Failed to process HTTP post: {str(e)}")
                    continue
        
        except Exception as e:
            self.logger.warning(f"Failed to process HTTP response: {str(e)}")
        
        return results
    
    def _process_http_post(self, post: Dict[str, Any], original_query: str, subreddit: str) -> Optional[ResearchResult]:
        """Process individual post from HTTP API."""
        try:
            title = post.get('title', '')
            selftext = post.get('selftext', '')
            permalink = post.get('permalink', '')
            
            if not title:
                return None
            
            url = f"https://reddit.com{permalink}"
            
            # Get subreddit credibility
            credibility = self.subreddit_credibility.get(subreddit.lower(), 0.5)
            
            # Create metadata
            metadata = SourceMetadata(
                url=url,
                title=title,
                source_type="social",
                domain="reddit.com",
                author=f"u/{post.get('author', '[deleted]')}",
                publish_date=datetime.fromtimestamp(post.get('created_utc', 0)),
                credibility_score=credibility,
                language="en"
            )
            
            # Combine content
            full_content = f"{title}\n\n{selftext}" if selftext else title
            
            # Extract key points from title and content
            key_points = self._extract_key_points_from_text(full_content)
            
            # Calculate confidence
            confidence = self._calculate_reddit_confidence(
                title, selftext, original_query,
                post.get('score', 0), post.get('num_comments', 0), credibility
            )
            
            return ResearchResult(
                content=full_content,
                metadata=metadata,
                summary=f"r/{subreddit} • {post.get('score', 0)} points • {post.get('num_comments', 0)} comments",
                key_points=key_points,
                confidence=confidence,
                raw_data={
                    'subreddit': subreddit,
                    'score': post.get('score', 0),
                    'upvote_ratio': post.get('upvote_ratio', 0),
                    'num_comments': post.get('num_comments', 0),
                    'created_utc': post.get('created_utc', 0),
                    'url': post.get('url', url),
                    'is_self': post.get('is_self', False),
                    'source': 'http'
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to process HTTP post: {str(e)}")
            return None
    
    def _extract_key_points_from_comments(self, submission) -> List[str]:
        """Extract key points from top comments (PRAW only)."""
        key_points = []
        
        try:
            # Get top comments
            submission.comments.replace_more(limit=0)  # Remove "more comments" objects
            top_comments = submission.comments[:5]  # Top 5 comments
            
            for comment in top_comments:
                if hasattr(comment, 'body') and len(comment.body) > 50:
                    # Extract first meaningful sentence
                    sentences = re.split(r'[.!?]+', comment.body)
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if len(sentence) > 30 and not sentence.startswith('&gt;'):
                            key_points.append(f"Comment: {sentence}")
                            break
                
                if len(key_points) >= 3:
                    break
        
        except Exception as e:
            self.logger.warning(f"Failed to extract comment key points: {str(e)}")
        
        return key_points
    
    def _extract_key_points_from_text(self, text: str) -> List[str]:
        """Extract key points from post text."""
        if not text or len(text) < 50:
            return []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        key_points = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and not sentence.startswith('&gt;'):
                key_points.append(sentence)
            
            if len(key_points) >= 3:
                break
        
        return key_points
    
    def _calculate_reddit_confidence(self, title: str, content: str, query: str, 
                                   score: int, num_comments: int, credibility: float) -> float:
        """Calculate confidence score for Reddit result."""
        confidence = 0.0
        query_terms = set(query.lower().split())
        
        # Title relevance (35%)
        title_lower = title.lower()
        title_matches = sum(1 for term in query_terms if term in title_lower)
        title_score = min(title_matches / len(query_terms), 1.0) * 0.35
        confidence += title_score
        
        # Content relevance (25%)
        if content:
            content_lower = content.lower()
            content_matches = sum(1 for term in query_terms if term in content_lower)
            content_score = min(content_matches / len(query_terms), 1.0) * 0.25
            confidence += content_score
        
        # Community engagement (25%)
        engagement_score = 0
        if score > 100:
            engagement_score = 0.25
        elif score > 50:
            engagement_score = 0.2
        elif score > 10:
            engagement_score = 0.15
        elif score > 0:
            engagement_score = 0.1
        
        # Comments add to engagement
        if num_comments > 50:
            engagement_score = min(engagement_score + 0.05, 0.25)
        elif num_comments > 10:
            engagement_score = min(engagement_score + 0.03, 0.25)
        
        confidence += engagement_score
        
        # Source credibility (15%)
        confidence += credibility * 0.15
        
        return min(confidence, 1.0)
    
    def _create_reddit_summary(self, submission) -> str:
        """Create summary for Reddit submission."""
        subreddit_name = submission.subreddit.display_name
        score = submission.score
        num_comments = submission.num_comments
        author = submission.author.name if submission.author else "[deleted]"
        
        return f"r/{subreddit_name} • {score} points • {num_comments} comments • by u/{author}"
    
    def _remove_duplicate_posts(self, results: List[ResearchResult]) -> List[ResearchResult]:
        """Remove duplicate posts based on title similarity."""
        if not results:
            return results
        
        unique_results = []
        seen_titles = set()
        
        for result in results:
            title_lower = result.metadata.title.lower()
            
            # Simple deduplication
            title_key = re.sub(r'[^\w\s]', '', title_lower)  # Remove punctuation
            title_key = ' '.join(title_key.split())  # Normalize whitespace
            
            if title_key not in seen_titles:
                unique_results.append(result)
                seen_titles.add(title_key)
        
        return unique_results
    
    async def search_subreddit(self, subreddit: str, query: str = "", max_results: int = 10) -> List[ResearchResult]:
        """
        Search within a specific subreddit.
        
        Args:
            subreddit: Subreddit name (without r/)
            query: Search query (optional)
            max_results: Maximum number of results
            
        Returns:
            List of research results from the subreddit
        """
        search_query = SearchQuery(
            query=query or subreddit,
            max_results=max_results,
            filters={'subreddits': [subreddit]}
        )
        
        return await self.search(search_query)
    
    async def get_trending_topics(self, subreddit: str = "all", max_results: int = 10) -> List[ResearchResult]:
        """
        Get trending topics from Reddit.
        
        Args:
            subreddit: Subreddit to check (default: "all")
            max_results: Maximum number of results
            
        Returns:
            List of trending topics
        """
        try:
            if self.reddit:
                # Use PRAW to get hot posts
                subreddit_obj = self.reddit.subreddit(subreddit)
                hot_posts = subreddit_obj.hot(limit=max_results)
                
                results = []
                for submission in hot_posts:
                    result = self._process_praw_submission(submission, "trending")
                    if result:
                        results.append(result)
                
                return results
            else:
                # Use HTTP API
                session = await self._get_session()
                url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={max_results}"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_http_response(data, "trending", subreddit)
                
                return []
                
        except Exception as e:
            self.logger.error(f"Failed to get trending topics: {str(e)}")
            return []
    
    def get_supported_subreddits(self) -> Dict[str, List[str]]:
        """Get supported subreddits by topic."""
        return self.topic_subreddits.copy()
    
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