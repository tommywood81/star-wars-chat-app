#!/usr/bin/env python3
"""
Standalone script to run the complete Star Wars RAG pipeline
"""

import sys
import os
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

# Add src to path
sys.path.append('src')

from star_wars_rag.data_processor import DialogueProcessor

logger = logging.getLogger(__name__)

def extract_all_dialogue(raw_scripts_dir: str = "data/raw") -> List[Dict]:
    """Extract dialogue from all Original Trilogy scripts."""
    processor = DialogueProcessor()
    
    trilogy_files = [
        "STAR WARS A NEW HOPE.txt",
        "THE EMPIRE STRIKES BACK.txt", 
        "STAR WARS THE RETURN OF THE JEDI.txt"
    ]
    
    all_dialogue = []
    raw_dir = Path(raw_scripts_dir)
    
    for filename in trilogy_files:
        script_path = raw_dir / filename
        if not script_path.exists():
            logger.warning(f"Script not found: {filename}")
            continue
        
        try:
            # Load script content
            script_content = processor.load_script(script_path)
            movie_name = filename.replace('.txt', '').replace('_', ' ').title()
            
            # Extract dialogue using robust processor
            dialogue_lines = processor.extract_dialogue_lines(script_content, movie_name)
            
            print(f"📄 {movie_name}: {len(dialogue_lines)} dialogue lines extracted")
            all_dialogue.extend(dialogue_lines)
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
    
    return all_dialogue

def enhance_dialogue_with_context(dialogue_data: List[Dict]) -> List[Dict]:
    """Enhance dialogue with rich context information."""
    processor = DialogueProcessor()
    enhanced_data = []
    
    print("📝 Enhancing dialogue with context...")
    
    for i, line in enumerate(dialogue_data):
        if i % 200 == 0:  # Progress logging
            print(f"Processing line {i+1}/{len(dialogue_data)}")
        
        character = line['character']
        dialogue = line['dialogue']
        scene = line.get('scene', 'Unknown')
        
        # Enhanced context from scene
        context_parts = []
        if scene and scene != "Unknown":
            # Clean scene description
            clean_scene = scene.replace('INT.', 'Interior:').replace('EXT.', 'Exterior:')
            context_parts.append(clean_scene)
        
        # Analyze dialogue for emotional context
        dialogue_lower = dialogue.lower()
        character_state = "Neutral"
        dialogue_type = "Statement"
        motivation = "Conversational"
        
        # Emotional state inference
        if '!' in dialogue or dialogue.isupper():
            character_state = "Urgent/Commanding"
            dialogue_type = "Exclamation"
            motivation = "Assertive"
        elif '?' in dialogue:
            character_state = "Inquisitive"
            dialogue_type = "Question" 
            motivation = "Seeking information"
        elif any(word in dialogue_lower for word in ['please', 'help', 'need']):
            character_state = "Requesting"
            motivation = "Seeking assistance"
        elif any(word in dialogue_lower for word in ['no', 'never', 'stop', 'don\'t']):
            character_state = "Resistant/Negative"
            motivation = "Opposing"
        elif any(word in dialogue_lower for word in ['yes', 'good', 'right', 'excellent']):
            character_state = "Positive/Agreeable"
            motivation = "Supportive"
        
        # Character-specific context
        normalized_char = processor.normalize_character_name(character)
        if normalized_char:
            if 'Vader' in normalized_char:
                context_parts.append("Dark Side presence, Imperial authority")
            elif 'Luke' in normalized_char:
                context_parts.append("Jedi training, heroic journey")
            elif 'Leia' in normalized_char:
                context_parts.append("Rebel leadership, royal dignity")
            elif 'Han' in normalized_char:
                context_parts.append("Smuggler confidence, roguish charm")
            elif 'C-3PO' in normalized_char:
                context_parts.append("Protocol droid formality, worry")
            elif 'R2-D2' in normalized_char:
                context_parts.append("Astromech communication, loyalty")
        
        enhanced_line = {
            'character': character,
            'character_normalized': normalized_char or character,
            'dialogue': dialogue,
            'context': " | ".join(context_parts) if context_parts else "General conversation",
            'character_state': character_state,
            'dialogue_type': dialogue_type,
            'motivation': motivation,
            'line_number': line['line_number'],
            'scene': line['scene'],
            'movie': line['movie'],
            'word_count': len(dialogue.split()),
            'sentiment': 'positive' if character_state in ['Positive/Agreeable'] else 'negative' if character_state in ['Resistant/Negative'] else 'neutral'
        }
        
        enhanced_data.append(enhanced_line)
    
    return enhanced_data

def save_complete_dataset(enhanced_data: List[Dict], output_dir: str = "data/processed_complete") -> None:
    """Save the complete enhanced dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(enhanced_data)
    
    # Save as CSV for analysis
    csv_path = output_path / "complete_enhanced_dialogue.csv"
    df.to_csv(csv_path, index=False)
    
    # Save as text format for easy reading
    txt_path = output_path / "complete_enhanced_dialogue.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("# Complete Enhanced Star Wars Original Trilogy Dialogue\n")
        f.write("# Total Lines: {:,}\n".format(len(enhanced_data)))
        f.write("# Format: CHARACTER | DIALOGUE | CONTEXT | CHARACTER_STATE | DIALOGUE_TYPE | MOTIVATION | MOVIE\n\n")
        
        for line in enhanced_data:
            f.write(f"{line['character_normalized']} | {line['dialogue']} | "
                   f"{line['context']} | {line['character_state']} | "
                   f"{line['dialogue_type']} | {line['motivation']} | {line['movie']}\n")
    
    print(f"💾 Saved complete dataset: {len(enhanced_data):,} lines")
    print(f"📄 CSV: {csv_path}")
    print(f"📄 TXT: {txt_path}")
    
    # Print statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"   🎬 Movies: {df['movie'].nunique()}")
    print(f"   👥 Characters: {df['character_normalized'].nunique()}")
    print(f"   💬 Total dialogue lines: {len(df):,}")
    print(f"   📝 Average words per line: {df['word_count'].mean():.1f}")
    
    # Top characters
    print(f"\n🌟 Top 10 Characters by Dialogue Count:")
    char_counts = df['character_normalized'].value_counts().head(10)
    for i, (char, count) in enumerate(char_counts.items(), 1):
        print(f"   {i:2d}. {char}: {count} lines")

def main():
    """Main function to run the complete pipeline."""
    logging.basicConfig(level=logging.INFO)
    
    print("🌟 Star Wars RAG Complete Dialogue Processing Pipeline")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Step 1: Extract ALL dialogue lines
        print("\n📖 Step 1: Extracting dialogue from Original Trilogy scripts...")
        all_dialogue = extract_all_dialogue()
        
        if not all_dialogue:
            print("❌ No dialogue extracted. Check script files in data/raw/")
            return 1
        
        print(f"✅ Total raw dialogue lines extracted: {len(all_dialogue):,}")
        
        # Step 2: Enhance with context
        print(f"\n🔍 Step 2: Enhancing dialogue with rich context...")
        enhanced_data = enhance_dialogue_with_context(all_dialogue)
        
        print(f"✅ Total enhanced dialogue lines: {len(enhanced_data):,}")
        
        # Step 3: Save complete dataset
        print(f"\n💾 Step 3: Saving complete enhanced dataset...")
        save_complete_dataset(enhanced_data)
        
        # Final statistics
        total_time = time.time() - start_time
        print(f"\n🎯 Pipeline Complete!")
        print(f"⏱️  Total processing time: {total_time:.2f} seconds")
        print(f"📊 Processing rate: {len(enhanced_data) / total_time:.1f} lines/second")
        
        # Count unique characters
        df = pd.DataFrame(enhanced_data)
        characters = df['character_normalized'].nunique()
        movies = df['movie'].nunique()
        
        print(f"\n🎉 Final Results:")
        print(f"   📝 Total dialogue lines: {len(enhanced_data):,}")
        print(f"   👥 Unique characters: {characters}")
        print(f"   🎬 Movies processed: {movies}")
        print(f"   📁 Output directory: data/processed_complete/")
        
        return 0
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
