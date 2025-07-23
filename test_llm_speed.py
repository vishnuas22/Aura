#!/usr/bin/env python3
"""
Quick LLM integration test to verify functionality and performance.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

async def test_llm_speed():
    """Test LLM integration speed and functionality."""
    print("🚀 Testing LLM Integration Speed...")
    
    try:
        # Import LLM components
        from llm.groq_client import GroqLLMClient, GroqModel
        from core.config import ConfigManager
        
        # Get API key
        config_manager = ConfigManager()
        system_config = config_manager.get_system_config()
        api_key = system_config.groq_api_key
        
        if not api_key or api_key == "your_groq_api_key_here":
            print("❌ Groq API key not configured")
            return False
        
        print(f"✅ API Key configured: {api_key[:20]}...")
        
        # Initialize client
        client = GroqLLMClient(
            api_key=api_key,
            default_model=GroqModel.LLAMA_3_1_8B_INSTANT,  # Use fastest model for test
            enable_caching=True,
            max_retries=2,
            request_timeout=30  # Shorter timeout for test
        )
        
        print("✅ LLM Client initialized")
        
        # Test health check
        start_time = time.time()
        health_status = await client.health_check()
        health_time = time.time() - start_time
        
        if health_status:
            print(f"✅ Health check passed in {health_time:.2f} seconds")
        else:
            print(f"❌ Health check failed in {health_time:.2f} seconds")
            return False
        
        # Test simple completion
        start_time = time.time()
        messages = [{"role": "user", "content": "Say 'Hello, I am working!' in exactly 5 words."}]
        
        response = await client.chat_completion(
            messages=messages,
            model=GroqModel.LLAMA_3_1_8B_INSTANT,
            temperature=0.1,
            max_tokens=20  # Very short response
        )
        
        completion_time = time.time() - start_time
        
        if response and response.content:
            print(f"✅ LLM response received in {completion_time:.2f} seconds")
            print(f"   Response: {response.content}")
            print(f"   Tokens used: {response.usage.total_tokens}")
            print(f"   Model: {response.model}")
            print(f"   Cached: {response.cached}")
        else:
            print("❌ No response from LLM")
            return False
        
        # Test cached response
        start_time = time.time()
        cached_response = await client.chat_completion(
            messages=messages,
            model=GroqModel.LLAMA_3_1_8B_INSTANT,
            temperature=0.1,
            max_tokens=20
        )
        cached_time = time.time() - start_time
        
        if cached_response and cached_response.cached:
            print(f"✅ Cached response received in {cached_time:.2f} seconds")
        else:
            print(f"⚠️  Response not cached, took {cached_time:.2f} seconds")
        
        # Test model recommendation
        rec_model = client.get_model_recommendation("research", "low")
        print(f"✅ Model recommendation for research/low: {rec_model.value}")
        
        # Clean up
        await client.close()
        
        print("\n🎉 LLM Integration working correctly!")
        print(f"📊 Performance Summary:")
        print(f"   - Health check: {health_time:.2f}s")
        print(f"   - First completion: {completion_time:.2f}s")  
        print(f"   - Cached completion: {cached_time:.2f}s")
        print(f"   - Total tokens: {response.usage.total_tokens}")
        
        # Performance assessment
        if completion_time < 10:
            print("🚀 Performance: Excellent (< 10s)")
        elif completion_time < 30:
            print("✅ Performance: Good (< 30s)")
        else:
            print("⚠️  Performance: Slow (> 30s) - May cause timeouts")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    # Reduce logging noise
    logging.basicConfig(level=logging.WARNING)
    
    success = await test_llm_speed()
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)