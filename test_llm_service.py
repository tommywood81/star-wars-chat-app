#!/usr/bin/env python3
"""
Test script to check LLM service functionality.
"""

import requests
import json

def test_llm_service():
    """Test the LLM service chat endpoint."""
    
    url = "http://localhost:5003/chat"
    
    payload = {
        "message": "What is the Force?",
        "character": "Luke Skywalker",
        "max_tokens": 200,
        "temperature": 0.7
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print("🧪 Testing LLM Service...")
        print(f"URL: {url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        print("-" * 50)
        
        response = requests.post(url, json=payload, headers=headers)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Success! Response:")
            print(json.dumps(data, indent=2))
            
            # Check for context lines
            if 'rag_context' in data:
                context_lines = data['rag_context']
                print(f"\n📚 Context Lines Retrieved: {len(context_lines)}")
                for i, line in enumerate(context_lines, 1):
                    print(f"  {i}. {line.get('dialogue', 'N/A')} (from {line.get('movie_title', 'Unknown')})")
            else:
                print("❌ No 'rag_context' field in response")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_llm_service()
