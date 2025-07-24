"""
Advanced relevance scoring for research results.

Provides sophisticated relevance scoring using:
- TF-IDF scoring
- Semantic similarity (basic)
- Context-aware scoring
- Multi-factor relevance assessment
- Query expansion and matching
"""

import re
import math
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import Counter, defaultdict

from .base_research_tool import ResearchResult, SearchQuery


class RelevanceScorer:
    """
    Advanced relevance scoring system for research results.
    
    Features:
    - TF-IDF based scoring
    - Semantic similarity assessment
    - Context-aware relevance
    - Query expansion
    - Multi-factor scoring
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Stop words (common words to ignore)
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'would', 'could', 'should', 'can',
            'this', 'these', 'those', 'they', 'them', 'their', 'there', 'where',
            'when', 'what', 'who', 'why', 'how', 'but', 'or', 'so', 'do', 'did',
            'does', 'have', 'had', 'been', 'being', 'were', 'we', 'you', 'your',
            'our', 'my', 'me', 'him', 'her', 'his', 'she', 'us', 'up', 'out',
            'down', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
        }
        
        # Importance weights for different content sections
        self.section_weights = {
            'title': 3.0,
            'summary': 2.0,
            'key_points': 2.5,
            'content': 1.0,
            'metadata': 1.5
        }
        
        # Domain credibility multipliers
        self.domain_multipliers = {
            'academic': 1.2,
            'news': 1.1,
            'reference': 1.3,
            'government': 1.2,
            'web': 1.0,
            'social': 0.9
        }
        
        # Query expansion synonyms (simplified)
        self.synonyms = {
            'ai': ['artificial intelligence', 'machine learning', 'ml', 'deep learning'],
            'artificial intelligence': ['ai', 'machine learning', 'ml'],
            'machine learning': ['ml', 'ai', 'artificial intelligence'],
            'ml': ['machine learning', 'ai', 'artificial intelligence'],
            'computer science': ['cs', 'computing', 'informatics'],
            'cs': ['computer science', 'computing'],
            'technology': ['tech', 'technological', 'innovation'],
            'tech': ['technology', 'technological'],
            'research': ['study', 'investigation', 'analysis'],
            'analysis': ['research', 'study', 'investigation'],
            'data': ['information', 'dataset', 'statistics'],
            'algorithm': ['method', 'approach', 'technique'],
            'model': ['framework', 'system', 'approach']
        }
    
    def score_results(self, 
                     results: List[ResearchResult], 
                     query: str,
                     context: Optional[Dict[str, Any]] = None) -> List[ResearchResult]:
        """
        Score and rank research results by relevance.
        
        Args:
            results: List of research results to score
            query: Original search query
            context: Additional context for scoring
            
        Returns:
            Sorted list of results with updated relevance scores
        """
        if not results:
            return results
        
        try:
            # Prepare query terms
            query_terms = self._prepare_query_terms(query)
            expanded_terms = self._expand_query(query_terms)
            
            # Build document corpus for TF-IDF
            corpus = self._build_corpus(results)
            idf_scores = self._calculate_idf(corpus)
            
            # Score each result
            scored_results = []
            for result in results:
                relevance_score = self._calculate_relevance_score(
                    result, query_terms, expanded_terms, idf_scores, context
                )
                
                # Update result's relevance score
                result.metadata.relevance_score = relevance_score
                scored_results.append((result, relevance_score))
            
            # Sort by relevance score (descending)
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # Return sorted results
            return [result for result, _ in scored_results]
        
        except Exception as e:
            self.logger.error(f"Relevance scoring failed: {str(e)}")
            return results  # Return original order if scoring fails
    
    def _prepare_query_terms(self, query: str) -> List[str]:
        """Prepare and normalize query terms."""
        # Convert to lowercase and extract words
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        
        # Remove stop words and short words
        meaningful_words = [
            word for word in words 
            if word not in self.stop_words and len(word) > 2
        ]
        
        return meaningful_words
    
    def _expand_query(self, query_terms: List[str]) -> Set[str]:
        """Expand query with synonyms and related terms."""
        expanded_terms = set(query_terms)
        
        for term in query_terms:
            if term in self.synonyms:
                expanded_terms.update(self.synonyms[term])
        
        return expanded_terms
    
    def _build_corpus(self, results: List[ResearchResult]) -> List[Dict[str, List[str]]]:
        """Build text corpus from research results."""
        corpus = []
        
        for result in results:
            document = {
                'title': self._tokenize_text(result.metadata.title),
                'content': self._tokenize_text(result.content),
                'summary': self._tokenize_text(result.summary or ''),
                'key_points': self._tokenize_text(' '.join(result.key_points or []))
            }
            corpus.append(document)
        
        return corpus
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into meaningful terms."""
        if not text:
            return []
        
        # Extract words and convert to lowercase
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter stop words and short words
        meaningful_words = [
            word for word in words 
            if word not in self.stop_words and len(word) > 2
        ]
        
        return meaningful_words
    
    def _calculate_idf(self, corpus: List[Dict[str, List[str]]]) -> Dict[str, float]:
        """Calculate Inverse Document Frequency scores."""
        # Count documents containing each term
        term_doc_count = defaultdict(int)
        total_docs = len(corpus)
        
        for document in corpus:
            # Get unique terms in this document
            doc_terms = set()
            for section_terms in document.values():
                doc_terms.update(section_terms)
            
            # Count document frequency for each term
            for term in doc_terms:
                term_doc_count[term] += 1
        
        # Calculate IDF scores
        idf_scores = {}
        for term, doc_freq in term_doc_count.items():
            if doc_freq > 0:
                idf_scores[term] = math.log(total_docs / doc_freq)
            else:
                idf_scores[term] = 0.0
        
        return idf_scores
    
    def _calculate_relevance_score(self,
                                  result: ResearchResult,
                                  query_terms: List[str],
                                  expanded_terms: Set[str],
                                  idf_scores: Dict[str, float],
                                  context: Optional[Dict[str, Any]]) -> float:
        """Calculate comprehensive relevance score for a result."""
        total_score = 0.0
        
        # TF-IDF scores for different sections
        sections = {
            'title': result.metadata.title,
            'content': result.content,
            'summary': result.summary or '',
            'key_points': ' '.join(result.key_points or [])
        }
        
        for section_name, section_text in sections.items():
            if not section_text:
                continue
            
            section_terms = self._tokenize_text(section_text)
            tf_idf_score = self._calculate_tf_idf_score(
                section_terms, query_terms, expanded_terms, idf_scores
            )
            
            # Apply section weight
            weighted_score = tf_idf_score * self.section_weights.get(section_name, 1.0)
            total_score += weighted_score
        
        # Normalize by number of sections
        if len([s for s in sections.values() if s]):
            total_score /= len([s for s in sections.values() if s])
        
        # Apply domain credibility multiplier
        source_type = result.metadata.source_type
        domain_multiplier = self.domain_multipliers.get(source_type, 1.0)
        total_score *= domain_multiplier
        
        # Apply additional factors
        total_score *= self._calculate_freshness_factor(result)
        total_score *= self._calculate_authority_factor(result)
        total_score *= self._calculate_context_factor(result, context)
        
        # Ensure score is between 0 and 1
        return max(0.0, min(total_score, 1.0))
    
    def _calculate_tf_idf_score(self,
                               section_terms: List[str],
                               query_terms: List[str],
                               expanded_terms: Set[str],
                               idf_scores: Dict[str, float]) -> float:
        """Calculate TF-IDF score for a text section."""
        if not section_terms:
            return 0.0
        
        # Calculate term frequencies
        tf_counts = Counter(section_terms)
        total_terms = len(section_terms)
        
        tf_idf_score = 0.0
        matched_terms = 0
        
        # Score exact query term matches (higher weight)
        for term in query_terms:
            if term in tf_counts:
                tf = tf_counts[term] / total_terms
                idf = idf_scores.get(term, 0.0)
                tf_idf_score += tf * idf * 2.0  # Double weight for exact matches
                matched_terms += 1
        
        # Score expanded term matches (lower weight)
        for term in expanded_terms:
            if term not in query_terms and term in tf_counts:
                tf = tf_counts[term] / total_terms
                idf = idf_scores.get(term, 0.0)
                tf_idf_score += tf * idf * 0.5  # Half weight for expanded terms
                matched_terms += 1
        
        # Bonus for multiple term matches
        if matched_terms > 1:
            coverage_bonus = min(matched_terms / len(query_terms), 1.0) * 0.2
            tf_idf_score += coverage_bonus
        
        return tf_idf_score
    
    def _calculate_freshness_factor(self, result: ResearchResult) -> float:
        """Calculate freshness factor based on publication date."""
        if not result.metadata.publish_date:
            return 1.0  # Neutral factor if no date
        
        try:
            from datetime import datetime, timezone
            
            # Ensure timezone-aware comparison
            publish_date = result.metadata.publish_date
            if publish_date.tzinfo is None:
                publish_date = publish_date.replace(tzinfo=timezone.utc)
            
            now = datetime.now(timezone.utc)
            age_days = (now - publish_date).days
            
            # Freshness scoring
            if age_days < 0:
                return 0.8  # Future dates are suspicious
            elif age_days <= 7:
                return 1.2  # Very fresh content
            elif age_days <= 30:
                return 1.1  # Recent content
            elif age_days <= 365:
                return 1.0  # Within a year
            elif age_days <= 1825:  # 5 years
                return 0.95
            else:
                return 0.9  # Older content
        
        except Exception:
            return 1.0  # Neutral factor on error
    
    def _calculate_authority_factor(self, result: ResearchResult) -> float:
        """Calculate authority factor based on source credibility."""
        authority_factor = 1.0
        
        # Base credibility score
        credibility = result.metadata.credibility_score
        if credibility > 0.8:
            authority_factor = 1.1
        elif credibility > 0.6:
            authority_factor = 1.05
        elif credibility < 0.4:
            authority_factor = 0.9
        
        # Author factor
        if result.metadata.author:
            author = result.metadata.author.lower()
            # Simple authority indicators
            if any(indicator in author for indicator in ['dr.', 'prof.', 'professor']):
                authority_factor *= 1.05
        
        # Content confidence factor
        if result.confidence > 0.8:
            authority_factor *= 1.02
        elif result.confidence < 0.5:
            authority_factor *= 0.98
        
        return authority_factor
    
    def _calculate_context_factor(self, 
                                 result: ResearchResult, 
                                 context: Optional[Dict[str, Any]]) -> float:
        """Calculate context-specific relevance factor."""
        if not context:
            return 1.0
        
        context_factor = 1.0
        
        # Domain preference
        preferred_domains = context.get('preferred_domains', [])
        if preferred_domains and result.metadata.domain:
            if result.metadata.domain in preferred_domains:
                context_factor *= 1.1
        
        # Source type preference
        preferred_source_types = context.get('preferred_source_types', [])
        if preferred_source_types and result.metadata.source_type:
            if result.metadata.source_type in preferred_source_types:
                context_factor *= 1.05
        
        # Language preference
        preferred_language = context.get('preferred_language')
        if preferred_language and result.metadata.language:
            if result.metadata.language == preferred_language:
                context_factor *= 1.02
        
        # Recency preference
        prefer_recent = context.get('prefer_recent', False)
        if prefer_recent and result.metadata.publish_date:
            age_days = (datetime.now() - result.metadata.publish_date.replace(tzinfo=None)).days
            if age_days <= 30:
                context_factor *= 1.05
        
        return context_factor
    
    def calculate_query_coverage(self, result: ResearchResult, query: str) -> Dict[str, Any]:
        """
        Calculate how well a result covers the query terms.
        
        Args:
            result: Research result to analyze
            query: Original search query
            
        Returns:
            Coverage analysis dictionary
        """
        query_terms = self._prepare_query_terms(query)
        expanded_terms = self._expand_query(query_terms)
        
        # Combine all result text
        all_text = ' '.join([
            result.metadata.title,
            result.content,
            result.summary or '',
            ' '.join(result.key_points or [])
        ])
        
        result_terms = self._tokenize_text(all_text)
        result_terms_set = set(result_terms)
        
        # Calculate coverage
        exact_matches = [term for term in query_terms if term in result_terms_set]
        expanded_matches = [term for term in expanded_terms if term in result_terms_set and term not in query_terms]
        
        coverage_score = len(exact_matches) / len(query_terms) if query_terms else 0.0
        
        return {
            'query_terms': query_terms,
            'exact_matches': exact_matches,
            'expanded_matches': expanded_matches,
            'coverage_score': coverage_score,
            'total_query_terms': len(query_terms),
            'matched_query_terms': len(exact_matches),
            'expansion_matches': len(expanded_matches)
        }
    
    def explain_relevance_score(self, 
                               result: ResearchResult, 
                               query: str,
                               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Provide detailed explanation of relevance score calculation.
        
        Args:
            result: Research result to explain
            query: Original search query
            context: Additional context
            
        Returns:
            Detailed explanation of score components
        """
        query_terms = self._prepare_query_terms(query)
        expanded_terms = self._expand_query(query_terms)
        
        # Build mini-corpus for IDF calculation
        corpus = self._build_corpus([result])
        idf_scores = self._calculate_idf(corpus)
        
        explanation = {
            'total_score': result.metadata.relevance_score,
            'query_terms': query_terms,
            'expanded_terms': list(expanded_terms),
            'section_scores': {},
            'factors': {}
        }
        
        # Calculate section scores
        sections = {
            'title': result.metadata.title,
            'content': result.content,
            'summary': result.summary or '',
            'key_points': ' '.join(result.key_points or [])
        }
        
        for section_name, section_text in sections.items():
            if section_text:
                section_terms = self._tokenize_text(section_text)
                tf_idf_score = self._calculate_tf_idf_score(
                    section_terms, query_terms, expanded_terms, idf_scores
                )
                
                explanation['section_scores'][section_name] = {
                    'raw_score': tf_idf_score,
                    'weight': self.section_weights.get(section_name, 1.0),
                    'weighted_score': tf_idf_score * self.section_weights.get(section_name, 1.0)
                }
        
        # Calculate factors
        explanation['factors'] = {
            'domain_multiplier': self.domain_multipliers.get(result.metadata.source_type, 1.0),
            'freshness_factor': self._calculate_freshness_factor(result),
            'authority_factor': self._calculate_authority_factor(result),
            'context_factor': self._calculate_context_factor(result, context)
        }
        
        # Query coverage
        explanation['query_coverage'] = self.calculate_query_coverage(result, query)
        
        return explanation
    
    def get_top_results_by_relevance(self,
                                   results: List[ResearchResult],
                                   query: str,
                                   top_k: int = 10,
                                   min_relevance: float = 0.1) -> List[ResearchResult]:
        """
        Get top K results by relevance score.
        
        Args:
            results: List of research results
            query: Search query
            top_k: Number of top results to return
            min_relevance: Minimum relevance threshold
            
        Returns:
            Top K most relevant results
        """
        # Score and sort results
        scored_results = self.score_results(results, query)
        
        # Filter by minimum relevance
        relevant_results = [
            result for result in scored_results
            if result.metadata.relevance_score >= min_relevance
        ]
        
        # Return top K
        return relevant_results[:top_k]