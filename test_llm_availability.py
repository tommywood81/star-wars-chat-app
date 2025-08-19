#!/usr/bin/env python3
"""
Test script to check LLM availability and model loading.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_imports():
    """Test if required imports are available."""
    print("🔍 Testing imports...")
    
    try:
        from llama_cpp import Llama
        print("✅ llama-cpp-python is available")
        return True
    except ImportError as e:
        print(f"❌ llama-cpp-python not available: {e}")
        return False

def test_model_file():
    """Test if model file exists."""
    print("🔍 Testing model file...")
    
    model_path = "models/phi-2.Q4_K_M.gguf"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Model file found: {model_path} ({size_mb:.1f} MB)")
        return model_path
    else:
        print(f"❌ Model file not found: {model_path}")
        return None

def test_llm_loading(model_path):
    """Test actual LLM loading."""
    print("🔍 Testing LLM loading...")
    
    try:
        from star_wars_rag.llm import LocalLLM
        print("✅ LocalLLM class imported successfully")
        
        print(f"🤖 Attempting to load: {model_path}")
        llm = LocalLLM(
            model_path=model_path,
            n_ctx=512,  # Small context for testing
            n_threads=2,
            verbose=True
        )
        print("✅ LLM loaded successfully!")
        
        # Test generation
        response = llm.generate("Test: Say hello in character as Han Solo", max_tokens=20)
        print(f"✅ Test generation: {response.get('response', 'No response')}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🌟 LLM Availability Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Cannot proceed without llama-cpp-python")
        sys.exit(1)
    
    # Test model file
    model_path = test_model_file()
    if not model_path:
        print("\n❌ Cannot proceed without model file")
        sys.exit(1)
    
    # Test LLM loading
    if test_llm_loading(model_path):
        print("\n🎉 All tests passed! LLM is ready for use.")
        sys.exit(0)
    else:
        print("\n❌ LLM loading failed")
        sys.exit(1)
