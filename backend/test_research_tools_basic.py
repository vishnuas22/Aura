#!/usr/bin/env python3
"""
Quick test script to verify research tools functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add research_tools to path
sys.path.append(str(Path(__file__).parent / 'research_tools'))

from research_tools.base_research_tool import ResearchResult, SourceMetadata, SearchQuery
from research_tools.content_validator import ContentValidator
from research_tools.relevance_scorer import RelevanceScorer
from research_tools.research_cache import ResearchCache

def test_base_classes():
    """Test basic class initialization."""
    print("🧪 Testing base classes...")
    
    # Test SourceMetadata
    metadata = SourceMetadata(
        url="https://example.com/test",
        title="Test Article",
        source_type="web",
        domain="example.com",
        credibility_score=0.8
    )
    
    assert metadata.url == "https://example.com/test"
    assert metadata.credibility_score == 0.8
    print("✅ SourceMetadata works correctly")
    
    # Test ResearchResult
    result = ResearchResult(
        content="This is test content for validation.",
        metadata=metadata,
        summary="Test summary",
        key_points=["Point 1", "Point 2"],
        confidence=0.75
    )
    
    assert result.content == "This is test content for validation."
    assert len(result.key_points) == 2
    print("✅ ResearchResult works correctly")
    
    # Test SearchQuery
    query = SearchQuery(
        query="test search",
        max_results=10,
        language="en"
    )
    
    assert query.query == "test search"
    assert query.max_results == 10
    print("✅ SearchQuery works correctly")

def test_content_validator():
    """Test content validation."""
    print("\n🧪 Testing content validator...")
    
    validator = ContentValidator()
    
    # Create test result
    good_result = ResearchResult(
        content="This is a well-researched article about machine learning. According to recent studies, AI has shown significant progress. The research methodology was comprehensive and peer-reviewed.",
        metadata=SourceMetadata(
            url="https://nature.com/article",
            title="Advances in Machine Learning",
            source_type="academic",
            domain="nature.com",
            credibility_score=0.95
        ),
        summary="High-quality research article",
        confidence=0.9
    )
    
    validation = validator.validate_result(good_result)
    
    assert validation['is_valid'] == True
    assert validation['quality_score'] > 0.5
    print(f"✅ Content validation works (quality: {validation['quality_score']:.2f})")
    
    # Test spam detection
    spam_content = "CLICK HERE NOW! Amazing deal! Buy now! Limited time!"
    spam_prob = validator._detect_spam(spam_content)
    
    assert spam_prob > 0.5
    print(f"✅ Spam detection works (spam probability: {spam_prob:.2f})")

def test_relevance_scorer():
    """Test relevance scoring."""
    print("\n🧪 Testing relevance scorer...")
    
    scorer = RelevanceScorer()
    
    # Create test results
    results = [
        ResearchResult(
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            metadata=SourceMetadata(
                url="https://example.com/ml",
                title="Machine Learning Fundamentals",
                source_type="web",
                domain="example.com",
                credibility_score=0.8
            ),
            summary="ML fundamentals",
            confidence=0.8
        ),
        ResearchResult(
            content="Cooking recipes and food preparation techniques are important for chefs.",
            metadata=SourceMetadata(
                url="https://example.com/cooking",
                title="Cooking Basics",
                source_type="web",
                domain="example.com",
                credibility_score=0.7
            ),
            summary="Cooking guide",
            confidence=0.7
        )
    ]
    
    # Score results
    scored_results = scorer.score_results(results, "machine learning")
    
    assert len(scored_results) == 2
    # First result should be more relevant to "machine learning"
    assert scored_results[0].metadata.relevance_score > scored_results[1].metadata.relevance_score
    print(f"✅ Relevance scoring works (scores: {scored_results[0].metadata.relevance_score:.2f}, {scored_results[1].metadata.relevance_score:.2f})")

async def test_research_cache():
    """Test research cache."""
    print("\n🧪 Testing research cache...")
    
    cache = ResearchCache(max_size=10, default_ttl=60)
    
    # Test data
    test_results = [
        ResearchResult(
            content="Test content for caching",
            metadata=SourceMetadata(
                url="https://example.com/cache-test",
                title="Cache Test Article",
                source_type="web",
                domain="example.com"
            )
        )
    ]
    
    # Set cache
    await cache.set("test query", "test_tool", test_results)
    
    # Get from cache
    cached_results = await cache.get("test query", "test_tool")
    
    assert cached_results is not None
    assert len(cached_results) == 1
    assert cached_results[0].content == "Test content for caching"
    print("✅ Research cache works correctly")
    
    # Test cache stats
    stats = cache.get_cache_stats()
    assert 'cache_size' in stats
    print(f"✅ Cache stats work (size: {stats['cache_size']})")

def test_query_expansion():
    """Test query expansion."""
    print("\n🧪 Testing query expansion...")
    
    scorer = RelevanceScorer()
    
    # Test synonym expansion
    query_terms = ['ai', 'machine learning']
    expanded = scorer._expand_query(query_terms)
    
    assert 'artificial intelligence' in expanded
    assert 'ml' in expanded
    print(f"✅ Query expansion works (expanded: {len(expanded)} terms)")

async def run_all_tests():
    """Run all tests."""
    print("🚀 Starting research tools tests...\n")
    
    try:
        test_base_classes()
        test_content_validator()
        test_relevance_scorer()
        await test_research_cache()
        test_query_expansion()
        
        print(f"\n🎉 All tests passed successfully!")
        print("✅ Research tools are working correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(run_all_tests())