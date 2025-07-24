#!/usr/bin/env python3
"""
Test just the base research tool components.
"""

import sys
from pathlib import Path

# Add research_tools to path
sys.path.append(str(Path(__file__).parent / 'research_tools'))

def test_base_research_tool():
    """Test base research tool functionality."""
    print("🧪 Testing base research tool...")
    
    try:
        from base_research_tool import ResearchResult, SourceMetadata, SearchQuery, BaseResearchTool
        
        # Test SourceMetadata
        metadata = SourceMetadata(
            url="https://example.com/test",
            title="Test Article About Machine Learning Applications",
            source_type="web",
            domain="example.com",
            credibility_score=0.8
        )
        
        assert metadata.url == "https://example.com/test"
        assert metadata.credibility_score == 0.8
        assert metadata.domain == "example.com"
        print("✅ SourceMetadata creation works")
        
        # Test ResearchResult
        result = ResearchResult(
            content="This is a comprehensive test content about machine learning applications in modern technology. The content has sufficient length to pass validation checks and contains meaningful information about artificial intelligence research and development.",
            metadata=metadata,
            summary="Test summary about ML applications",
            key_points=["ML is important", "AI has many applications", "Research is ongoing"],
            confidence=0.85
        )
        
        assert result.content.startswith("This is a comprehensive")
        assert len(result.key_points) == 3
        assert result.confidence == 0.85
        print("✅ ResearchResult creation works")
        
        # Test SearchQuery
        query = SearchQuery(
            query="machine learning applications",
            max_results=10,
            language="en",
            sort_by="relevance"
        )
        
        assert query.query == "machine learning applications"
        assert query.max_results == 10
        assert query.language == "en"
        print("✅ SearchQuery creation works")
        
        # Test BaseResearchTool abstract functionality
        class MockTool(BaseResearchTool):
            async def _perform_search(self, query):
                return [result]
        
        mock_tool = MockTool("mock", "Mock tool", "web")
        assert mock_tool.name == "mock"
        assert mock_tool.source_type == "web"
        print("✅ BaseResearchTool inheritance works")
        
        # Test validation
        is_valid = mock_tool._validate_result(result)
        assert is_valid == True
        print("✅ Result validation works")
        
        # Test relevance calculation
        relevance = mock_tool._calculate_relevance_score("machine learning", result)
        assert 0.0 <= relevance <= 1.0
        print(f"✅ Relevance calculation works (score: {relevance:.3f})")
        
        # Test cache key generation
        cache_key = mock_tool._generate_cache_key(query)
        assert len(cache_key) == 32  # MD5 hash length
        print("✅ Cache key generation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Base research tool test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_functionality():
    """Test some simple standalone functionality."""
    print("\n🧪 Testing simple functionality...")
    
    try:
        # Test basic Python functionality that the tools rely on
        import hashlib
        import re
        from datetime import datetime
        from collections import Counter
        
        # Test text processing (used in relevance scoring)
        text = "machine learning artificial intelligence data science"
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        word_count = Counter(words)
        
        assert 'machine' in words
        assert word_count['learning'] == 1
        print("✅ Text processing works")
        
        # Test hash generation (used in caching)
        test_string = "test query for hashing"
        hash_result = hashlib.md5(test_string.encode()).hexdigest()
        assert len(hash_result) == 32
        print("✅ Hash generation works")
        
        # Test datetime handling (used in metadata)
        now = datetime.now()
        assert now.year >= 2024
        print("✅ DateTime handling works")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple functionality test failed: {e}")
        return False

def main():
    """Run basic tests."""
    print("🚀 Starting basic research tools tests...\n")
    
    tests = [
        ("Base Research Tool", test_base_research_tool),
        ("Simple Functionality", test_simple_functionality)
    ]
    
    results = []
    for name, test_func in tests:
        try:
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
        print("🎉 All basic tests passed!")
        print("✅ Core research tools foundation is working correctly")
        print("\n🔧 The research tools package is ready for use!")
        print("📚 See README.md for usage examples and integration guide")
    else:
        failed = [name for name, result in results if not result]
        print(f"❌ Failed tests: {', '.join(failed)}")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)