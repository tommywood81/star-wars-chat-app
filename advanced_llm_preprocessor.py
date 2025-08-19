#!/usr/bin/env python3
"""
Advanced LLM-Powered Script Preprocessor

Uses Phi-2 LLM to intelligently extract rich contextual information for each dialogue line:
- Who is speaking to whom
- When in the scene this occurs
- What is happening around the dialogue
- Emotional context and motivations
- Scene setting and dramatic tension
"""

import sys
import os
import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
import time

# Add src to path
sys.path.append('src')

try:
    from star_wars_rag.llm import LocalLLM
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️  LLM module not available. Install llama-cpp-python for LLM preprocessing.")

logger = logging.getLogger(__name__)

class AdvancedLLMPreprocessor:
    """Advanced LLM-powered script preprocessor for rich contextual analysis."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.llm = None
        
        # Try to load Phi-2 model
        if model_path and os.path.exists(model_path) and LLM_AVAILABLE:
            try:
                self.llm = LocalLLM(
                    model_path=model_path,
                    n_ctx=2048,  # Larger context for scene analysis
                    n_threads=2,
                    verbose=False
                )
                print(f"🤖 Phi-2 LLM loaded: {model_path}")
            except Exception as e:
                print(f"⚠️  Failed to load LLM: {e}")
        else:
            print("📝 Using rule-based preprocessing (no LLM model)")
    
    def analyze_dialogue_context_with_llm(self, script_chunk: str, character: str, dialogue: str) -> Dict[str, Any]:
        """Use Phi-2 LLM to extract rich contextual information about a dialogue line."""
        
        if not self.llm:
            return self._fallback_context_analysis(character, dialogue)
        
        try:
            # Create detailed analysis prompt
            prompt = f"""Analyze this Star Wars script excerpt to extract detailed context for the dialogue line.

SCRIPT EXCERPT:
{script_chunk}

TARGET DIALOGUE:
Character: {character}
Line: "{dialogue}"

Extract the following information in JSON format:
{{
    "speaker": "{character}",
    "dialogue": "{dialogue}",
    "addressee": "who is being spoken to (character name or 'group' or 'self')",
    "scene_location": "specific location where this takes place",
    "scene_timing": "when in the scene this occurs (beginning/middle/end/climax)",
    "surrounding_action": "what is happening around this dialogue",
    "speaker_emotion": "emotional state of the speaker",
    "speaker_motivation": "why the character is saying this",
    "dramatic_context": "dramatic tension or significance",
    "relationship_dynamic": "relationship between speaker and addressee",
    "scene_stakes": "what is at stake in this moment"
}}

Respond with ONLY the JSON object:"""

            # Generate analysis
            response = self.llm.generate(
                prompt=prompt,
                max_tokens=300,
                temperature=0.2,  # Low temperature for consistent analysis
                stop=["\n\n", "SCRIPT:", "Character:"]
            )
            
            response_text = response.get('response', '').strip()
            
            # Try to parse JSON response
            try:
                # Clean up response to extract JSON
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_text = response_text[json_start:json_end]
                    context_analysis = json.loads(json_text)
                    
                    # Validate required fields
                    required_fields = ['speaker', 'dialogue', 'addressee', 'scene_location']
                    if all(field in context_analysis for field in required_fields):
                        return context_analysis
                
            except json.JSONDecodeError:
                pass
            
            # If JSON parsing fails, fall back to rule-based
            print(f"⚠️  LLM response parsing failed for {character}, using fallback")
            return self._fallback_context_analysis(character, dialogue)
            
        except Exception as e:
            print(f"⚠️  LLM analysis failed: {e}")
            return self._fallback_context_analysis(character, dialogue)
    
    def _fallback_context_analysis(self, character: str, dialogue: str) -> Dict[str, Any]:
        """Fallback rule-based context analysis when LLM is unavailable."""
        
        # Basic emotion detection
        dialogue_lower = dialogue.lower()
        
        if '!' in dialogue or dialogue.isupper():
            emotion = "urgent/commanding"
            motivation = "asserting authority or urgency"
        elif '?' in dialogue:
            emotion = "inquisitive"
            motivation = "seeking information"
        elif any(word in dialogue_lower for word in ['please', 'help', 'need']):
            emotion = "pleading/desperate"
            motivation = "requesting assistance"
        elif any(word in dialogue_lower for word in ['no', 'never', 'stop', "don't"]):
            emotion = "resistant/defiant"
            motivation = "opposing or refusing"
        else:
            emotion = "neutral"
            motivation = "conversational"
        
        # Character-specific context
        addressee = "unknown"
        if 'luke' in character.lower():
            addressee = "companions/allies"
        elif 'vader' in character.lower():
            addressee = "subordinates/enemies"
        elif 'leia' in character.lower():
            addressee = "rebel allies"
        
        return {
            "speaker": character,
            "dialogue": dialogue,
            "addressee": addressee,
            "scene_location": "Star Wars universe setting",
            "scene_timing": "middle",
            "surrounding_action": "contextual scene action",
            "speaker_emotion": emotion,
            "speaker_motivation": motivation,
            "dramatic_context": "ongoing story progression",
            "relationship_dynamic": "character interaction",
            "scene_stakes": "story progression"
        }
    
    def process_script_with_advanced_context(self, script_path: Path) -> List[Dict[str, Any]]:
        """Process entire script with advanced contextual analysis."""
        
        print(f"🎬 Processing {script_path.name} with advanced context analysis...")
        
        try:
            with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"❌ Error reading {script_path}: {e}")
            return []
        
        lines = content.split('\n')
        processed_dialogue = []
        
        # Find dialogue lines with character detection
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Enhanced character detection pattern
            char_match = re.match(r'^([A-Z][A-Z\s\-\']+?)\s+(.+)$', line)
            
            if char_match and len(char_match.group(1)) < 50:
                character = char_match.group(1).strip()
                dialogue = char_match.group(2).strip()
                
                # Filter for substantial dialogue
                if (len(dialogue) > 10 and 
                    not dialogue.startswith('(') and
                    not any(indicator in dialogue.upper() for indicator in ['INT.', 'EXT.', 'CUT TO:', 'FADE'])):
                    
                    # Get surrounding context (larger window for better LLM analysis)
                    context_start = max(0, i - 15)
                    context_end = min(len(lines), i + 10)
                    context_chunk = '\n'.join(lines[context_start:context_end])
                    
                    # Clean dialogue
                    clean_dialogue = re.sub(r'\([^)]*\)', '', dialogue).strip()
                    clean_dialogue = re.sub(r'\s+', ' ', clean_dialogue).strip()
                    
                    if len(clean_dialogue) > 5:
                        # Get advanced context analysis
                        context_analysis = self.analyze_dialogue_context_with_llm(
                            context_chunk, character, clean_dialogue
                        )
                        
                        # Add metadata
                        context_analysis.update({
                            'movie': script_path.stem.replace('_', ' ').title(),
                            'line_number': i + 1,
                            'word_count': len(clean_dialogue.split()),
                            'processing_method': 'llm' if self.llm else 'rule_based'
                        })
                        
                        processed_dialogue.append(context_analysis)
        
        print(f"✅ Extracted {len(processed_dialogue)} context-rich dialogue lines")
        return processed_dialogue
    
    def process_all_trilogy_scripts(self, raw_scripts_dir: str = "data/raw") -> List[Dict[str, Any]]:
        """Process all Original Trilogy scripts with advanced context."""
        
        trilogy_files = [
            "STAR WARS A NEW HOPE.txt",
            "THE EMPIRE STRIKES BACK.txt", 
            "STAR WARS THE RETURN OF THE JEDI.txt"
        ]
        
        all_processed = []
        scripts_dir = Path(raw_scripts_dir)
        
        for filename in trilogy_files:
            script_path = scripts_dir / filename
            if script_path.exists():
                processed = self.process_script_with_advanced_context(script_path)
                all_processed.extend(processed)
                
                # Progress update
                print(f"📊 Running total: {len(all_processed)} dialogue lines processed")
            else:
                print(f"⚠️  Script not found: {filename}")
        
        return all_processed
    
    def save_enhanced_dataset(self, processed_data: List[Dict[str, Any]], 
                            output_dir: str = "data/llm_enhanced") -> None:
        """Save the LLM-enhanced dataset with rich context."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save as JSON for full structure preservation
        json_path = output_path / "llm_enhanced_dialogue_complete.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        # Save as readable text format
        txt_path = output_path / "llm_enhanced_dialogue_readable.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("# LLM-Enhanced Star Wars Dialogue with Rich Context\n")
            f.write(f"# Total Lines: {len(processed_data):,}\n")
            f.write("# Enhanced with: Speaker, Addressee, Scene Context, Emotions, Motivations\n\n")
            
            for item in processed_data:
                f.write(f"DIALOGUE: {item['speaker']} → {item['addressee']}\n")
                f.write(f"LINE: \"{item['dialogue']}\"\n")
                f.write(f"LOCATION: {item['scene_location']}\n")
                f.write(f"TIMING: {item['scene_timing']}\n")
                f.write(f"ACTION: {item['surrounding_action']}\n")
                f.write(f"EMOTION: {item['speaker_emotion']}\n")
                f.write(f"MOTIVATION: {item['speaker_motivation']}\n")
                f.write(f"STAKES: {item['scene_stakes']}\n")
                f.write(f"MOVIE: {item['movie']}\n")
                f.write("-" * 80 + "\n\n")
        
        print(f"💾 Enhanced dataset saved:")
        print(f"   📄 JSON: {json_path}")
        print(f"   📄 TXT: {txt_path}")
        
        # Generate statistics
        self._generate_statistics(processed_data)
    
    def _generate_statistics(self, processed_data: List[Dict[str, Any]]) -> None:
        """Generate comprehensive statistics about the enhanced dataset."""
        
        print(f"\n📊 LLM-Enhanced Dataset Statistics:")
        print(f"=" * 50)
        
        # Basic counts
        total_lines = len(processed_data)
        movies = set(item['movie'] for item in processed_data)
        speakers = set(item['speaker'] for item in processed_data)
        
        print(f"📝 Total dialogue lines: {total_lines:,}")
        print(f"🎬 Movies: {len(movies)}")
        print(f"👥 Unique speakers: {len(speakers)}")
        
        # Processing method breakdown
        llm_processed = sum(1 for item in processed_data if item.get('processing_method') == 'llm')
        rule_processed = total_lines - llm_processed
        
        print(f"\n🤖 Processing Methods:")
        print(f"   LLM-Enhanced: {llm_processed:,} ({llm_processed/total_lines*100:.1f}%)")
        print(f"   Rule-Based: {rule_processed:,} ({rule_processed/total_lines*100:.1f}%)")
        
        # Top speakers
        from collections import Counter
        speaker_counts = Counter(item['speaker'] for item in processed_data)
        
        print(f"\n🌟 Top 10 Speakers:")
        for i, (speaker, count) in enumerate(speaker_counts.most_common(10), 1):
            print(f"   {i:2d}. {speaker}: {count} lines")
        
        # Emotion distribution
        emotion_counts = Counter(item['speaker_emotion'] for item in processed_data)
        print(f"\n😊 Emotion Distribution:")
        for emotion, count in emotion_counts.most_common(5):
            print(f"   {emotion}: {count} lines ({count/total_lines*100:.1f}%)")


def main():
    """Main function to run advanced LLM preprocessing."""
    logging.basicConfig(level=logging.INFO)
    
    print("🌟 Advanced LLM-Powered Star Wars Script Preprocessing")
    print("=" * 60)
    
    # Check for Phi-2 model
    model_paths = [
        "models/phi-2-q4.gguf",
        "models/phi-2.gguf", 
        "phi-2-q4.gguf"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("⚠️  No Phi-2 model found. Trying these locations:")
        for path in model_paths:
            print(f"   - {path}")
        print("\n📥 Download Phi-2 GGUF model for best results")
        print("🔄 Proceeding with rule-based enhancement...")
    
    # Initialize processor
    processor = AdvancedLLMPreprocessor(model_path=model_path)
    
    start_time = time.time()
    
    try:
        # Process all scripts
        print(f"\n🎬 Processing Original Trilogy scripts...")
        enhanced_data = processor.process_all_trilogy_scripts()
        
        if not enhanced_data:
            print("❌ No dialogue extracted. Check script files in data/raw/")
            return 1
        
        # Save enhanced dataset
        print(f"\n💾 Saving enhanced dataset...")
        processor.save_enhanced_dataset(enhanced_data)
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n🎯 Processing Complete!")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print(f"📊 Processing rate: {len(enhanced_data) / total_time:.1f} lines/second")
        print(f"📁 Output directory: data/llm_enhanced/")
        
        return 0
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
