#!/usr/bin/env python3
"""
Test script to verify enhanced context is working in the RAG prompt system.
"""

import sys
from pathlib import Path
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from star_wars_rag.prompt import StarWarsPromptBuilder

def test_enhanced_context_prompt():
    """Test that enhanced context fields are included in prompts."""
    
    # Create sample enhanced context data (like what our demo file contains)
    enhanced_context = [
        {
            "character": "HAN",
            "dialogue": "Ooops! I guess you haven't told Luke about that yet.",
            "movie": "The Empire Strikes Back",
            "addressee": "Luke", 
            "emotion": "awkward_embarrassed",
            "location": "Millennium_Falcon_cockpit",
            "stakes": "relationship_revelation",
            "context": "Han realizes Luke doesn't know about his feelings for Leia and tries to break the news gently"
        },
        {
            "character": "HAN",
            "dialogue": "Luke….Ah about Leia….I",
            "movie": "Star Wars The Return Of The Jedi",
            "addressee": "Luke",
            "emotion": "hesitant_romantic", 
            "location": "Endor_rebel_base",
            "stakes": "love_confession",
            "context": "Han is trying to confess his feelings about Leia to Luke before the dangerous mission"
        }
    ]
    
    # Initialize prompt builder
    prompt_builder = StarWarsPromptBuilder()
    
    # Build prompt with enhanced context
    prompt = prompt_builder.build_character_prompt(
        character="Han Solo",
        user_message="Who is Luke Skywalker?",
        retrieved_context=enhanced_context,
        max_context_lines=2
    )
    
    print("🔍 ENHANCED CONTEXT PROMPT TEST")
    print("=" * 50)
    print(prompt)
    print("=" * 50)
    
    # Verify enhanced context elements are included
    tests = [
        ("speaking to Luke", "addressee field"),
        ("emotion: awkward_embarrassed", "emotion field"),
        ("location: Millennium_Falcon_cockpit", "location field"), 
        ("stakes: relationship_revelation", "stakes field"),
        ("scene: Han realizes Luke doesn't know", "context field")
    ]
    
    print("\n✅ CONTEXT VERIFICATION:")
    all_passed = True
    
    for test_string, description in tests:
        if test_string in prompt:
            print(f"✅ {description}: FOUND")
        else:
            print(f"❌ {description}: MISSING - '{test_string}'")
            all_passed = False
    
    print(f"\n🎯 OVERALL RESULT: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    print("🌟 Testing Enhanced Context in RAG Prompts")
    print("=" * 60)
    
    success = test_enhanced_context_prompt()
    
    if success:
        print("\n🎉 Enhanced context is working correctly!")
        print("📋 The RAG prompts now include:")
        print("   👥 WHO the character is speaking to")
        print("   😊 Character's EMOTION in the scene")
        print("   📍 LOCATION where the dialogue occurs")
        print("   ⚡ STAKES of what's happening")
        print("   🎬 SCENE context and situation")
    else:
        print("\n🚨 Enhanced context needs fixes!")
    
    sys.exit(0 if success else 1)
