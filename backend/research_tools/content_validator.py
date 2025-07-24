"""
Content validation utilities for research results.

Provides validation and quality assessment including:
- Content quality scoring
- Language detection
- Spam/low-quality detection
- Factual consistency checking
- Source reliability assessment
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from urllib.parse import urlparse

from .base_research_tool import ResearchResult, SourceMetadata


class ContentValidator:
    """
    Validates and scores research content quality.
    
    Features:
    - Content quality assessment
    - Language detection
    - Spam detection
    - Source credibility validation
    - Factual consistency checking
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Quality indicators
        self.quality_indicators = {
            'positive': [
                'according to', 'research shows', 'study found', 'data indicates',
                'expert', 'professor', 'researcher', 'published', 'peer-reviewed',
                'university', 'institute', 'journal', 'analysis', 'methodology'
            ],
            'negative': [
                'click here', 'buy now', 'limited time', 'amazing deal',
                'you won\'t believe', 'shocking', 'weird trick', 'doctors hate',
                'get rich quick', 'miracle cure', 'conspiracy', 'fake news'
            ]
        }
        
        # Credible domains and patterns
        self.credible_domains = {
            'academic': [
                '.edu', '.ac.uk', '.ac.', 'scholar.google', 'arxiv.org',
                'pubmed.ncbi.nlm.nih.gov', 'jstor.org', 'springer.com',
                'nature.com', 'science.org', 'ieee.org', 'acm.org'
            ],
            'news': [
                'reuters.com', 'ap.org', 'bbc.com', 'npr.org',
                'nytimes.com', 'washingtonpost.com', 'guardian.com',
                'wsj.com', 'economist.com', 'bloomberg.com'
            ],
            'government': [
                '.gov', '.mil', 'who.int', 'un.org', 'europa.eu',
                'cdc.gov', 'nih.gov', 'nasa.gov', 'fda.gov'
            ],
            'reference': [
                'wikipedia.org', 'britannica.com', 'merriam-webster.com',
                'dictionary.com', 'encyclopedia.com'
            ]
        }
        
        # Language detection patterns (basic)
        self.language_patterns = {
            'en': r'[a-zA-Z\s.,!?;:\'"-]+',
            'es': r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ\s.,!?;:\'"-]+',
            'fr': r'[a-zA-ZàâäéèêëïîôöùûüÿçÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ\s.,!?;:\'"-]+',
            'de': r'[a-zA-ZäöüßÄÖÜ\s.,!?;:\'"-]+'
        }
        
        # Spam detection patterns
        self.spam_patterns = [
            r'(?i)click\s+here\s+for',
            r'(?i)buy\s+now',
            r'(?i)limited\s+time\s+offer',
            r'(?i)act\s+now',
            r'(?i)free\s+trial',
            r'(?i)get\s+rich\s+quick',
            r'(?i)miracle\s+cure',
            r'(?i)lose\s+weight\s+fast',
            r'(?i)make\s+money\s+from\s+home'
        ]
    
    def validate_result(self, result: ResearchResult) -> Dict[str, Any]:
        """
        Comprehensive validation of research result.
        
        Args:
            result: Research result to validate
            
        Returns:
            Validation report with scores and flags
        """
        validation_report = {
            'is_valid': True,
            'quality_score': 0.0,
            'issues': [],
            'warnings': [],
            'metadata_score': 0.0,
            'content_score': 0.0,
            'source_score': 0.0,
            'language': 'en',
            'spam_probability': 0.0,
            'factual_consistency': 0.0
        }
        
        try:
            # Validate content
            content_validation = self._validate_content(result.content)
            validation_report.update(content_validation)
            
            # Validate metadata
            metadata_validation = self._validate_metadata(result.metadata)
            validation_report['metadata_score'] = metadata_validation['score']
            validation_report['issues'].extend(metadata_validation['issues'])
            
            # Validate source credibility
            source_validation = self._validate_source(result.metadata)
            validation_report['source_score'] = source_validation['score']
            validation_report['warnings'].extend(source_validation['warnings'])
            
            # Calculate overall quality score
            validation_report['quality_score'] = self._calculate_overall_quality(
                validation_report['content_score'],
                validation_report['metadata_score'],
                validation_report['source_score']
            )
            
            # Determine validity
            validation_report['is_valid'] = (
                validation_report['quality_score'] >= 0.3 and
                validation_report['spam_probability'] < 0.7 and
                len(validation_report['issues']) == 0
            )
            
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            validation_report['is_valid'] = False
            validation_report['issues'].append(f"Validation error: {str(e)}")
        
        return validation_report
    
    def _validate_content(self, content: str) -> Dict[str, Any]:
        """Validate content quality and characteristics."""
        if not content:
            return {
                'content_score': 0.0,
                'language': 'unknown',
                'spam_probability': 0.0,
                'issues': ['Empty content']
            }
        
        content_report = {
            'content_score': 0.0,
            'language': 'en',
            'spam_probability': 0.0,
            'issues': []
        }
        
        # Basic content checks
        content_length = len(content)
        word_count = len(content.split())
        
        # Minimum content requirements
        if content_length < 50:
            content_report['issues'].append('Content too short')
            content_report['content_score'] = 0.1
            return content_report
        
        if word_count < 10:
            content_report['issues'].append('Too few words')
            content_report['content_score'] = 0.1
            return content_report
        
        # Language detection
        content_report['language'] = self._detect_language(content)
        
        # Spam detection
        content_report['spam_probability'] = self._detect_spam(content)
        
        # Quality indicators
        quality_score = self._assess_content_quality(content)
        content_report['content_score'] = quality_score
        
        # Readability assessment
        readability_score = self._assess_readability(content)
        content_report['readability'] = readability_score
        
        # Adjust content score based on readability
        content_report['content_score'] = (
            content_report['content_score'] * 0.7 + 
            readability_score * 0.3
        )
        
        return content_report
    
    def _validate_metadata(self, metadata: SourceMetadata) -> Dict[str, Any]:
        """Validate metadata completeness and quality."""
        metadata_report = {
            'score': 0.0,
            'issues': []
        }
        
        score = 0.0
        
        # Check required fields
        if not metadata.url:
            metadata_report['issues'].append('Missing URL')
        else:
            score += 0.3
        
        if not metadata.title:
            metadata_report['issues'].append('Missing title')
        else:
            score += 0.2
            
            # Check title quality
            if len(metadata.title) < 5:
                metadata_report['issues'].append('Title too short')
            elif len(metadata.title) > 200:
                metadata_report['issues'].append('Title too long')
            else:
                score += 0.1
        
        # Check optional but valuable fields
        if metadata.author:
            score += 0.1
        
        if metadata.publish_date:
            score += 0.1
            
            # Check if date is reasonable
            if isinstance(metadata.publish_date, datetime):
                age_days = (datetime.now() - metadata.publish_date.replace(tzinfo=None)).days
                if age_days < 0:
                    metadata_report['issues'].append('Future publication date')
                elif age_days > 36500:  # 100 years
                    metadata_report['issues'].append('Very old publication date')
        
        if metadata.domain:
            score += 0.1
        
        # Credibility score validation
        if metadata.credibility_score < 0 or metadata.credibility_score > 1:
            metadata_report['issues'].append('Invalid credibility score')
        else:
            score += metadata.credibility_score * 0.1
        
        metadata_report['score'] = min(score, 1.0)
        return metadata_report
    
    def _validate_source(self, metadata: SourceMetadata) -> Dict[str, Any]:
        """Validate source credibility and reliability."""
        source_report = {
            'score': 0.0,
            'warnings': []
        }
        
        if not metadata.url:
            source_report['score'] = 0.0
            source_report['warnings'].append('No URL to validate')
            return source_report
        
        try:
            parsed_url = urlparse(metadata.url)
            domain = parsed_url.netloc.lower()
            
            # Check against credible domain lists
            credibility_score = 0.0
            
            for category, domains in self.credible_domains.items():
                for credible_domain in domains:
                    if credible_domain in domain:
                        if category == 'academic':
                            credibility_score = max(credibility_score, 0.95)
                        elif category == 'government':
                            credibility_score = max(credibility_score, 0.9)
                        elif category == 'news':
                            credibility_score = max(credibility_score, 0.85)
                        elif category == 'reference':
                            credibility_score = max(credibility_score, 0.8)
                        break
            
            # Check for suspicious patterns
            suspicious_patterns = [
                '.tk', '.ml', '.ga', '.cf',  # Free domains
                'bit.ly', 'tinyurl.com',  # URL shorteners
                'blogspot.com', 'wordpress.com'  # Free hosting (lower credibility)
            ]
            
            for pattern in suspicious_patterns:
                if pattern in domain:
                    credibility_score = max(0, credibility_score - 0.2)
                    source_report['warnings'].append(f'Potentially low-credibility domain: {pattern}')
            
            # Default credibility for unknown domains
            if credibility_score == 0.0:
                credibility_score = 0.5
            
            source_report['score'] = credibility_score
            
        except Exception as e:
            source_report['warnings'].append(f'URL validation error: {str(e)}')
            source_report['score'] = 0.3  # Default low score for invalid URLs
        
        return source_report
    
    def _detect_language(self, content: str) -> str:
        """Detect content language (basic implementation)."""
        # Count matches for each language pattern
        language_scores = {}
        
        for lang, pattern in self.language_patterns.items():
            matches = len(re.findall(pattern, content))
            language_scores[lang] = matches / len(content) if content else 0
        
        # Return language with highest score
        if language_scores:
            return max(language_scores, key=language_scores.get)
        
        return 'en'  # Default to English
    
    def _detect_spam(self, content: str) -> float:
        """Detect spam probability in content."""
        spam_score = 0.0
        content_lower = content.lower()
        
        # Check spam patterns
        for pattern in self.spam_patterns:
            matches = len(re.findall(pattern, content_lower))
            spam_score += matches * 0.2
        
        # Check for excessive capitalization
        caps_ratio = sum(1 for c in content if c.isupper()) / len(content) if content else 0
        if caps_ratio > 0.3:
            spam_score += 0.3
        
        # Check for excessive punctuation
        punct_ratio = sum(1 for c in content if c in '!?') / len(content) if content else 0
        if punct_ratio > 0.05:
            spam_score += 0.2
        
        # Check negative quality indicators
        for indicator in self.quality_indicators['negative']:
            if indicator in content_lower:
                spam_score += 0.1
        
        return min(spam_score, 1.0)
    
    def _assess_content_quality(self, content: str) -> float:
        """Assess content quality based on various indicators."""
        quality_score = 0.5  # Start with neutral score
        content_lower = content.lower()
        
        # Positive quality indicators
        positive_count = 0
        for indicator in self.quality_indicators['positive']:
            if indicator in content_lower:
                positive_count += 1
        
        quality_score += positive_count * 0.05
        
        # Negative quality indicators
        negative_count = 0
        for indicator in self.quality_indicators['negative']:
            if indicator in content_lower:
                negative_count += 1
        
        quality_score -= negative_count * 0.1
        
        # Length-based quality
        word_count = len(content.split())
        if word_count > 500:
            quality_score += 0.1
        elif word_count > 200:
            quality_score += 0.05
        elif word_count < 50:
            quality_score -= 0.2
        
        # Sentence structure
        sentences = re.split(r'[.!?]+', content)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        if 10 <= avg_sentence_length <= 25:  # Reasonable sentence length
            quality_score += 0.05
        
        return max(0.0, min(quality_score, 1.0))
    
    def _assess_readability(self, content: str) -> float:
        """Assess content readability (simplified Flesch-Kincaid style)."""
        try:
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            words = content.split()
            syllables = sum(self._count_syllables(word) for word in words)
            
            if not sentences or not words:
                return 0.0
            
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables_per_word = syllables / len(words)
            
            # Simplified readability score
            readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            
            # Normalize to 0-1 scale
            return max(0.0, min(readability / 100.0, 1.0))
        
        except Exception:
            return 0.5  # Default neutral score
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximate)."""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        # Every word has at least one syllable
        return max(syllable_count, 1)
    
    def _calculate_overall_quality(self, content_score: float, metadata_score: float, source_score: float) -> float:
        """Calculate overall quality score from component scores."""
        # Weighted average
        weights = {
            'content': 0.5,
            'metadata': 0.2,
            'source': 0.3
        }
        
        overall_score = (
            content_score * weights['content'] +
            metadata_score * weights['metadata'] +
            source_score * weights['source']
        )
        
        return overall_score
    
    def filter_results(self, results: List[ResearchResult], min_quality: float = 0.5) -> List[ResearchResult]:
        """
        Filter research results by quality threshold.
        
        Args:
            results: List of research results
            min_quality: Minimum quality score threshold
            
        Returns:
            Filtered list of results
        """
        filtered_results = []
        
        for result in results:
            validation = self.validate_result(result)
            
            if validation['is_valid'] and validation['quality_score'] >= min_quality:
                # Update result confidence based on validation
                result.confidence = (result.confidence + validation['quality_score']) / 2
                filtered_results.append(result)
            else:
                self.logger.debug(f"Filtered out result with quality {validation['quality_score']:.2f}")
        
        return filtered_results
    
    def get_quality_report(self, results: List[ResearchResult]) -> Dict[str, Any]:
        """
        Generate quality report for a list of results.
        
        Args:
            results: List of research results
            
        Returns:
            Quality report with statistics
        """
        if not results:
            return {
                'total_results': 0,
                'valid_results': 0,
                'average_quality': 0.0,
                'quality_distribution': {},
                'common_issues': []
            }
        
        validations = [self.validate_result(result) for result in results]
        
        valid_count = sum(1 for v in validations if v['is_valid'])
        quality_scores = [v['quality_score'] for v in validations]
        
        # Quality distribution
        quality_ranges = {
            'excellent': sum(1 for s in quality_scores if s >= 0.8),
            'good': sum(1 for s in quality_scores if 0.6 <= s < 0.8),
            'fair': sum(1 for s in quality_scores if 0.4 <= s < 0.6),
            'poor': sum(1 for s in quality_scores if s < 0.4)
        }
        
        # Common issues
        all_issues = []
        for v in validations:
            all_issues.extend(v['issues'])
        
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_results': len(results),
            'valid_results': valid_count,
            'average_quality': sum(quality_scores) / len(quality_scores),
            'quality_distribution': quality_ranges,
            'common_issues': common_issues,
            'spam_detected': sum(1 for v in validations if v['spam_probability'] > 0.5)
        }