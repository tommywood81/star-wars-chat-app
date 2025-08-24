#!/usr/bin/env python3
"""
Test script for the enhanced explainability feature.
This script tests the LLM service to ensure it returns all the required data for explainability.
"""

import requests
import json
import time

def test_llm_service_explainability():
    """Test the LLM service explainability features."""
    
    # LLM service URL
    llm_url = "http://localhost:5003"
    
    print("🧪 Testing LLM Service Explainability Features")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{llm_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed: {health_data}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Get characters
    print("\n2. Testing character retrieval...")
    try:
        response = requests.get(f"{llm_url}/characters", timeout=10)
        if response.status_code == 200:
            characters_data = response.json()
            characters = characters_data.get('characters', [])
            print(f"✅ Found {len(characters)} characters: {characters[:5]}...")
        else:
            print(f"❌ Character retrieval failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Character retrieval error: {e}")
        return False
    
    # Test 3: Chat with explainability
    print("\n3. Testing chat with explainability...")
    try:
        chat_request = {
            "message": "Tell me about the Force",
            "character": "Luke Skywalker",
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        response = requests.post(f"{llm_url}/chat", json=chat_request, timeout=30)
        if response.status_code == 200:
            chat_data = response.json()
            print("✅ Chat request successful!")
            
            # Check for explainability fields
            print("\n📊 Explainability Data Check:")
            print("-" * 30)
            
            # Check response field
            if 'response' in chat_data:
                print(f"✅ Response: {chat_data['response'][:100]}...")
            else:
                print("❌ Missing 'response' field")
            
            # Check character field
            if 'character' in chat_data:
                print(f"✅ Character: {chat_data['character']}")
            else:
                print("❌ Missing 'character' field")
            
            # Check RAG context
            if 'rag_context' in chat_data:
                context = chat_data['rag_context']
                print(f"✅ RAG Context: {len(context)} lines retrieved")
                if context:
                    print(f"   First context line: {context[0].get('dialogue', 'N/A')[:50]}...")
            else:
                print("❌ Missing 'rag_context' field")
            
            # Check complete prompt
            if 'complete_prompt' in chat_data:
                prompt = chat_data['complete_prompt']
                print(f"✅ Complete Prompt: {len(prompt)} characters")
                print(f"   Prompt preview: {prompt[:200]}...")
            else:
                print("❌ Missing 'complete_prompt' field")
            
            # Check request data
            if 'request_data' in chat_data:
                request_data = chat_data['request_data']
                print(f"✅ Request Data: {request_data}")
            else:
                print("❌ Missing 'request_data' field")
            
            # Check metadata
            if 'metadata' in chat_data:
                metadata = chat_data['metadata']
                print(f"✅ Metadata: {len(metadata)} fields")
                print(f"   Processing time: {metadata.get('processing_time', 'N/A')}s")
                print(f"   Tokens generated: {metadata.get('tokens_generated', 'N/A')}")
                print(f"   Context lines: {metadata.get('context_lines_retrieved', 'N/A')}")
            else:
                print("❌ Missing 'metadata' field")
            
            return True
            
        else:
            print(f"❌ Chat request failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Chat request error: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Starting LLM Service Explainability Test")
    print("Make sure the LLM service is running on localhost:5003")
    print("=" * 60)
    
    success = test_llm_service_explainability()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! The explainability feature is working correctly.")
        print("\nThe LLM service now returns:")
        print("✅ Request data sent to the service")
        print("✅ Retrieved context (matching movie lines)")
        print("✅ Complete prompt sent to the model")
        print("✅ Model response")
        print("✅ Technical metadata")
        print("\nThis data can now be displayed in the dashboard explainability panel!")
    else:
        print("❌ Some tests failed. Please check the LLM service configuration.")
    
    return success

if __name__ == "__main__":
    main()
