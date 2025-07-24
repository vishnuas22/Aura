#!/usr/bin/env python3
"""
Simple test script for core research tool components.
"""

import asyncio
import sys
from pathlib import Path

# Add research_tools to path
sys.path.append(str(Path(__file__).parent / 'research_tools'))

def test_basic_imports():
    """Test that core modules can be imported."""
    print("🧪 Testing basic imports...")
    
    try:
        from research_tools.base_research_tool import ResearchResult, SourceMetadata, SearchQuery
        print("✅ Base research tool imported successfully")
        
        from research_tools.content_validator import ContentValidator
        print("✅ Content validator imported successfully")
        
        from research_tools.relevance_scorer import RelevanceScorer
        print("✅ Relevance scorer imported successfully")
        
        from research_tools.research_cache import ResearchCache
        print("✅ Research cache imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_data_structures():
    """Test basic data structures."""
    print("\n🧪 Testing data structures...")
    
    try:
        from research_tools.base_research_tool import ResearchResult, SourceMetadata, SearchQuery
        
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
            content="This is test content for validation with sufficient length to pass basic checks.",
            metadata=metadata,
            summary="Test summary",
            key_points=["Point 1", "Point 2"],
            confidence=0.75
        )
        
        assert result.content.startswith("This is test content")
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
        
        return True
    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
        return False

def test_content_validation():
    """Test content validation without external dependencies."""
    print("\n🧪 Testing content validation...")
    
    try:
        from research_tools.content_validator import ContentValidator
        from research_tools.base_research_tool import ResearchResult, SourceMetadata
        
        validator = ContentValidator()
        
        # Create high-quality test result
        good_result = ResearchResult(
            content="This is a well-researched article about machine learning. According to recent studies published in peer-reviewed journals, artificial intelligence has shown significant progress in various applications. The research methodology was comprehensive and involved extensive analysis of data from multiple credible sources.",
            metadata=SourceMetadata(
                url="https://nature.com/article",
                title="Advances in Machine Learning Research",
                source_type="academic",
                domain="nature.com",
                credibility_score=0.95
            ),
            summary="High-quality research article about ML advances",
            confidence=0.9
        )
        
        validation = validator.validate_result(good_result)
        
        assert validation['is_valid'] == True
        assert validation['quality_score'] > 0.3
        print(f"✅ Content validation works (quality: {validation['quality_score']:.3f})")
        
        # Test spam detection
        spam_content = "CLICK HERE NOW! Amazing deal! Buy now! Limited time offer! You won't believe this!"
        spam_prob = validator._detect_spam(spam_content)
        
        assert spam_prob > 0.3
        print(f"✅ Spam detection works (spam probability: {spam_prob:.3f})")
        
        return True
    except Exception as e:
        print(f"❌ Content validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_relevance_scoring():
    """Test relevance scoring."""
    print("\n🧪 Testing relevance scoring...")
    
    try:
        from research_tools.relevance_scorer import RelevanceScorer
        from research_tools.base_research_tool import ResearchResult, SourceMetadata
        
        scorer = RelevanceScorer()
        
        # Create test results with different relevance levels
        results = [
            ResearchResult(
                content="Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models. It enables computers to learn and improve from experience without being explicitly programmed.",
                metadata=SourceMetadata(
                    url="https://example.com/ml",
                    title="Machine Learning Fundamentals and Applications",
                    source_type="web",
                    domain="example.com",
                    credibility_score=0.8
                ),
                summary="Comprehensive guide to ML fundamentals",
                confidence=0.8
            ),
            ResearchResult(
                content="Cooking recipes and food preparation techniques are important skills for professional chefs. Understanding ingredients and cooking methods helps create delicious meals.",
                metadata=SourceMetadata(
                    url="https://example.com/cooking",
                    title="Professional Cooking Techniques",
                    source_type="web",
                    domain="example.com",
                    credibility_score=0.7
                ),
                summary="Guide to professional cooking",
                confidence=0.7
            )
        ]
        
        # Score results for ML query
        scored_results = scorer.score_results(results, "machine learning applications")
        
        assert len(scored_results) == 2
        # First result should be more relevant to "machine learning"
        ml_score = scored_results[0].metadata.relevance_score
        cooking_score = scored_results[1].metadata.relevance_score
        
        assert ml_score > cooking_score
        print(f"✅ Relevance scoring works (ML: {ml_score:.3f}, Cooking: {cooking_score:.3f})")
        
        # Test query expansion
        query_terms = ['ai', 'machine learning']
        expanded = scorer._expand_query(query_terms)
        
        assert 'artificial intelligence' in expanded
        assert 'ml' in expanded
        print(f"✅ Query expansion works (expanded to {len(expanded)} terms)")
        
        return True
    except Exception as e:
        print(f"❌ Relevance scoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_cache():
    """Test research cache."""
    print("\n🧪 Testing research cache...")
    
    try:
        from research_tools.research_cache import ResearchCache
        from research_tools.base_research_tool import ResearchResult, SourceMetadata
        
        cache = ResearchCache(max_size=10, default_ttl=60)
        
        # Create test data
        test_results = [
            ResearchResult(
                content="Test content for caching functionality validation",
                metadata=SourceMetadata(
                    url="https://example.com/cache-test",
                    title="Cache Test Article",
                    source_type="web",
                    domain="example.com",
                    credibility_score=0.7
                )
            )
        ]
        
        # Set cache
        await cache.set("test query", "test_tool", test_results)
        
        # Get from cache
        cached_results = await cache.get("test query", "test_tool")
        
        assert cached_results is not None
        assert len(cached_results) == 1
        assert cached_results[0].content == "Test content for caching functionality validation"
        print("✅ Research cache set/get works correctly")
        
        # Test cache stats
        stats = cache.get_stats()
        assert 'cache_size' in stats
        assert stats['cache_size'] == 1
        print(f"✅ Cache stats work (size: {stats['cache_size']})")
        
        # Test cache miss
        missed_results = await cache.get("non-existent query", "test_tool")
        assert missed_results is None
        print("✅ Cache miss handling works correctly")
        
        return True
    except Exception as e:
        print(f"❌ Cache test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_core_tests():
    """Run core functionality tests."""
    print("🚀 Starting core research tools tests...\n")
    
    test_results = []
    
    # Run tests
    test_results.append(("Basic Imports", test_basic_imports()))
    test_results.append(("Data Structures", test_data_structures()))
    test_results.append(("Content Validation", test_content_validation()))
    test_results.append(("Relevance Scoring", test_relevance_scoring()))
    test_results.append(("Research Cache", await test_cache()))
    
    # Summary
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All core tests passed successfully!")
        print("✅ Research tools core functionality is working correctly")
        return True
    else:
        failed_tests = [name for name, result in test_results if not result]
        print(f"❌ Failed tests: {', '.join(failed_tests)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_core_tests())
    sys.exit(0 if success else 1)