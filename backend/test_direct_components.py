#!/usr/bin/env python3
"""
Direct test of individual research tool components.
"""

import asyncio
import sys
from pathlib import Path

# Add research_tools to path
sys.path.append(str(Path(__file__).parent / 'research_tools'))

def test_base_components():
    """Test base components directly."""
    print("🧪 Testing base components...")
    
    try:
        # Import base components directly
        from base_research_tool import ResearchResult, SourceMetadata, SearchQuery
        
        # Test SourceMetadata
        metadata = SourceMetadata(
            url="https://example.com/test",
            title="Test Article",
            source_type="web",
            domain="example.com",
            credibility_score=0.8
        )
        
        assert metadata.url == "https://example.com/test"
        print("✅ SourceMetadata works")
        
        # Test ResearchResult
        result = ResearchResult(
            content="Test content with sufficient length for validation.",
            metadata=metadata,
            summary="Test summary",
            key_points=["Point 1", "Point 2"],
            confidence=0.75
        )
        
        assert result.content.startswith("Test content")
        print("✅ ResearchResult works")
        
        # Test SearchQuery
        query = SearchQuery(query="test", max_results=10)
        assert query.query == "test"
        print("✅ SearchQuery works")
        
        return True
    except Exception as e:
        print(f"❌ Base components failed: {e}")
        return False

def test_content_validator():
    """Test content validator."""
    print("\n🧪 Testing content validator...")
    
    try:
        from content_validator import ContentValidator
        from base_research_tool import ResearchResult, SourceMetadata
        
        validator = ContentValidator()
        
        # High-quality content
        good_result = ResearchResult(
            content="This is a well-researched article about machine learning. According to recent studies, AI has shown significant progress. The research methodology was comprehensive.",
            metadata=SourceMetadata(
                url="https://nature.com/article",
                title="Advances in Machine Learning",
                source_type="academic",
                domain="nature.com",
                credibility_score=0.95
            )
        )
        
        validation = validator.validate_result(good_result)
        assert validation['is_valid'] == True
        print(f"✅ Content validation works (quality: {validation['quality_score']:.3f})")
        
        # Spam detection
        spam_prob = validator._detect_spam("CLICK HERE NOW! Buy now!")
        assert spam_prob > 0.3
        print(f"✅ Spam detection works (probability: {spam_prob:.3f})")
        
        return True
    except Exception as e:
        print(f"❌ Content validator failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_relevance_scorer():
    """Test relevance scorer."""
    print("\n🧪 Testing relevance scorer...")
    
    try:
        from relevance_scorer import RelevanceScorer
        from base_research_tool import ResearchResult, SourceMetadata
        
        scorer = RelevanceScorer()
        
        # Create test results
        results = [
            ResearchResult(
                content="Machine learning is about algorithms and AI applications.",
                metadata=SourceMetadata(
                    url="https://example.com/ml",
                    title="Machine Learning Guide",
                    source_type="web",
                    domain="example.com",
                    credibility_score=0.8
                )
            ),
            ResearchResult(
                content="Cooking recipes and food preparation techniques.",
                metadata=SourceMetadata(
                    url="https://example.com/cook",
                    title="Cooking Guide",
                    source_type="web", 
                    domain="example.com",
                    credibility_score=0.7
                )
            )
        ]
        
        # Score for ML query
        scored = scorer.score_results(results, "machine learning")
        
        ml_score = scored[0].metadata.relevance_score
        cook_score = scored[1].metadata.relevance_score
        
        assert ml_score > cook_score
        print(f"✅ Relevance scoring works (ML: {ml_score:.3f} > Cooking: {cook_score:.3f})")
        
        # Test query expansion
        expanded = scorer._expand_query(['ai'])
        assert 'artificial intelligence' in expanded
        print("✅ Query expansion works")
        
        return True
    except Exception as e:
        print(f"❌ Relevance scorer failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_research_cache():
    """Test research cache."""
    print("\n🧪 Testing research cache...")
    
    try:
        from research_cache import ResearchCache
        from base_research_tool import ResearchResult, SourceMetadata
        
        cache = ResearchCache(max_size=10)
        
        # Test data
        test_results = [
            ResearchResult(
                content="Test content for caching",
                metadata=SourceMetadata(
                    url="https://example.com/test",
                    title="Test Article",
                    source_type="web",
                    domain="example.com"
                )
            )
        ]
        
        # Set and get cache
        await cache.set("test query", "test_tool", test_results)
        cached = await cache.get("test query", "test_tool")
        
        assert cached is not None
        assert len(cached) == 1
        print("✅ Cache set/get works")
        
        # Test stats
        stats = cache.get_stats()
        assert stats['cache_size'] == 1
        print(f"✅ Cache stats work (size: {stats['cache_size']})")
        
        return True
    except Exception as e:
        print(f"❌ Cache test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_duckduckgo_basic():
    """Test DuckDuckGo tool basic functionality."""
    print("\n🧪 Testing DuckDuckGo basic functionality...")
    
    try:
        from duckduckgo_search import DuckDuckGoSearchTool
        
        tool = DuckDuckGoSearchTool()
        
        # Test tool initialization
        assert tool.name == "duckduckgo_search"
        assert tool.source_type == "web"
        print("✅ DuckDuckGo tool initializes correctly")
        
        # Test domain credibility
        wiki_cred = tool._calculate_domain_credibility("wikipedia.org")
        random_cred = tool._calculate_domain_credibility("random-site.com")
        
        assert wiki_cred > random_cred
        print(f"✅ Domain credibility works (wiki: {wiki_cred:.2f} > random: {random_cred:.2f})")
        
        return True
    except Exception as e:
        print(f"❌ DuckDuckGo basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_direct_tests():
    """Run direct component tests."""
    print("🚀 Starting direct component tests...\n")
    
    tests = [
        ("Base Components", test_base_components),
        ("Content Validator", test_content_validator),
        ("Relevance Scorer", test_relevance_scorer),
        ("Research Cache", test_research_cache),
        ("DuckDuckGo Basic", test_duckduckgo_basic)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All direct tests passed!")
        print("✅ Core research tools functionality is working")
    else:
        failed = [name for name, result in results if not result]
        print(f"❌ Failed tests: {', '.join(failed)}")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(run_direct_tests())
    sys.exit(0 if success else 1)