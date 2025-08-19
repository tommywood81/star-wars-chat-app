#!/usr/bin/env python3
"""
Quick test of LLM processing functionality.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

from smart_pipeline_preprocessor import SmartPipelinePreprocessor

def test_llm_processing():
    """Test LLM processing with a small sample."""
    print("🧪 Testing LLM Processing")
    print("=" * 40)
    
    # Initialize with correct model path
    model_path = "models/phi-2.Q4_K_M.gguf"
    processor = SmartPipelinePreprocessor(model_path=model_path)
    
    # Test data
    test_script = """
    LUKE
    I can't believe he's gone.

    LEIA
    There wasn't anything you could have done, Luke, had you been there. You couldn't have stopped him from coming here.

    LUKE
    He sacrificed himself for us.
    """
    
    test_dialogue = "I can't believe he's gone."
    test_character = "LUKE"
    
    print(f"🎬 Test dialogue: {test_character}: \"{test_dialogue}\"")
    print(f"🤖 LLM available: {processor.llm is not None}")
    
    if processor.llm:
        print("🔄 Testing LLM analysis...")
        try:
            result = processor.analyze_dialogue_with_progress(
                script_chunk=test_script,
                character=test_character,
                dialogue=test_dialogue,
                line_idx=1,
                total_lines=1
            )
            
            print("✅ LLM Analysis Result:")
            for key, value in result.items():
                print(f"   {key}: {value}")
            
            return True
            
        except Exception as e:
            print(f"❌ LLM analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("❌ LLM not available - using rule-based")
        return False

if __name__ == "__main__":
    success = test_llm_processing()
    exit(0 if success else 1)
