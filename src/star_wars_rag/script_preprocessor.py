#!/usr/bin/env python3
"""
Script Preprocessor for Star Wars RAG

This module preprocesses Star Wars movie scripts to add scene context
to each dialogue line for enhanced RAG responses.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class ScriptPreprocessor:
    """Preprocesses movie scripts to add scene context to dialogue lines."""
    
    def __init__(self, raw_scripts_dir: str = "data/raw", output_dir: str = "data/preprocessed"):
        self.raw_scripts_dir = Path(raw_scripts_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Patterns for identifying different elements
        self.scene_indicators = [
            r"INT\.\s+",  # Interior scenes
            r"EXT\.\s+",  # Exterior scenes
            r"FADE IN:",
            r"FADE OUT:",
            r"CUT TO:",
            r"CLOSE-UP",
            r"WIDE SHOT",
            r"ANGLE ON",
            r"POV",
            r"MONTAGE"
        ]
        
        self.character_pattern = r"^([A-Z][A-Z\s\-']+)$"
        self.dialogue_pattern = r"^[^A-Z\n].*"
        
    def extract_scene_context(self, text_lines: List[str], line_index: int, context_window: int = 5) -> str:
        """Extract scene context around a dialogue line."""
        start_idx = max(0, line_index - context_window)
        end_idx = min(len(text_lines), line_index + context_window + 1)
        
        context_lines = []
        current_scene = None
        current_location = None
        
        # Look backwards for scene heading
        for i in range(line_index, -1, -1):
            line = text_lines[i].strip()
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in self.scene_indicators):
                current_scene = line
                break
            if i <= start_idx:
                break
        
        # Extract location from scene heading
        if current_scene:
            # Clean up scene heading
            scene_clean = re.sub(r'(INT\.|EXT\.)', '', current_scene, flags=re.IGNORECASE).strip()
            scene_clean = re.sub(r'(DAY|NIGHT|MORNING|EVENING|CONTINUOUS)', '', scene_clean, flags=re.IGNORECASE).strip()
            scene_clean = re.sub(r'\s*-\s*', ' - ', scene_clean).strip()
            current_location = scene_clean
        
        # Get surrounding context for action/direction
        context_actions = []
        for i in range(max(0, line_index - 3), min(len(text_lines), line_index + 2)):
            if i == line_index:
                continue
            line = text_lines[i].strip()
            if line and not re.match(self.character_pattern, line) and not line.startswith('('):
                # This might be action/direction
                if len(line) > 10 and not line.isupper():
                    context_actions.append(line[:100])  # Limit length
        
        # Build context string
        context_parts = []
        if current_location:
            context_parts.append(f"Scene: {current_location}")
        if context_actions:
            context_parts.append(f"Action: {' '.join(context_actions[:2])}")
        
        return " | ".join(context_parts) if context_parts else "Scene context unavailable"
    
    def process_script(self, script_path: Path) -> List[Dict[str, str]]:
        """Process a single script file to extract dialogue with context."""
        logger.info(f"Processing script: {script_path.name}")
        
        try:
            with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading {script_path}: {e}")
            return []
        
        lines = content.split('\n')
        processed_lines = []
        current_character = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a character name
            if re.match(self.character_pattern, line) and len(line) < 50:
                current_character = line.strip()
                continue
            
            # Check if this is dialogue (not action/direction)
            if current_character and line and not line.isupper() and not line.startswith('('):
                # This appears to be dialogue
                if len(line) > 5 and not any(indicator in line.upper() for indicator in ['INT.', 'EXT.', 'CUT TO:', 'FADE']):
                    
                    # Get scene context
                    context = self.extract_scene_context(lines, i)
                    
                    # Clean up dialogue
                    dialogue = re.sub(r'\([^)]*\)', '', line).strip()  # Remove parentheticals
                    dialogue = re.sub(r'\s+', ' ', dialogue).strip()   # Normalize whitespace
                    
                    if len(dialogue) > 10:  # Only keep substantial dialogue
                        processed_lines.append({
                            'character': current_character,
                            'dialogue': dialogue,
                            'context': context,
                            'movie': script_path.stem.replace('_', ' ').title()
                        })
                        
                        # Reset character after capturing dialogue
                        current_character = None
        
        logger.info(f"Extracted {len(processed_lines)} dialogue lines from {script_path.name}")
        return processed_lines
    
    def preprocess_all_scripts(self) -> None:
        """Process all scripts in the raw directory."""
        # Original trilogy files
        trilogy_files = [
            "STAR WARS A NEW HOPE.txt",
            "THE EMPIRE STRIKES BACK.txt", 
            "STAR WARS THE RETURN OF THE JEDI.txt"
        ]
        
        all_processed = []
        
        for filename in trilogy_files:
            script_path = self.raw_scripts_dir / filename
            if script_path.exists():
                processed = self.process_script(script_path)
                all_processed.extend(processed)
            else:
                logger.warning(f"Script file not found: {filename}")
        
        # Save individual processed files
        for filename in trilogy_files:
            script_path = self.raw_scripts_dir / filename
            if script_path.exists():
                processed = self.process_script(script_path)
                output_file = self.output_dir / f"enhanced_{filename}"
                self.save_processed_script(processed, output_file)
        
        # Save combined file
        combined_file = self.output_dir / "enhanced_original_trilogy_combined.txt"
        self.save_processed_script(all_processed, combined_file)
        
        logger.info(f"Preprocessing complete. Total lines processed: {len(all_processed)}")
    
    def save_processed_script(self, processed_lines: List[Dict[str, str]], output_path: Path) -> None:
        """Save processed dialogue lines to file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# Enhanced Star Wars Script with Scene Context\n")
                f.write("# Format: CHARACTER | DIALOGUE | CONTEXT | MOVIE\n\n")
                
                for line in processed_lines:
                    f.write(f"{line['character']} | {line['dialogue']} | {line['context']} | {line['movie']}\n")
            
            logger.info(f"Saved {len(processed_lines)} enhanced lines to {output_path}")
        except Exception as e:
            logger.error(f"Error saving to {output_path}: {e}")


def main():
    """Main function to run script preprocessing."""
    logging.basicConfig(level=logging.INFO)
    
    preprocessor = ScriptPreprocessor()
    preprocessor.preprocess_all_scripts()
    
    print("✅ Script preprocessing completed!")
    print("📁 Enhanced scripts saved to: data/preprocessed/")
    print("🎬 Files created:")
    print("   - enhanced_STAR WARS A NEW HOPE.txt")
    print("   - enhanced_THE EMPIRE STRIKES BACK.txt") 
    print("   - enhanced_STAR WARS THE RETURN OF THE JEDI.txt")
    print("   - enhanced_original_trilogy_combined.txt")


if __name__ == "__main__":
    main()
