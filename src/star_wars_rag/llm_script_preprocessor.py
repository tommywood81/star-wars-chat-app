#!/usr/bin/env python3
"""
LLM-Powered Script Preprocessor using Phi-2

This module uses the Phi-2 LLM to intelligently preprocess Star Wars scripts,
extracting scene context and enhancing dialogue with rich metadata.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import json

from .llm import LocalLLM

logger = logging.getLogger(__name__)

class LLMScriptPreprocessor:
    """Uses Phi-2 LLM to intelligently preprocess Star Wars scripts."""
    
    def __init__(self, model_path: Optional[str] = None, raw_scripts_dir: str = "data/raw", output_dir: str = "data/preprocessed"):
        self.raw_scripts_dir = Path(raw_scripts_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize LLM if model path provided
        self.llm = None
        if model_path and os.path.exists(model_path):
            try:
                self.llm = LocalLLM(model_path=model_path, n_ctx=1024, verbose=False)
                logger.info(f"LLM loaded for intelligent preprocessing: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load LLM, falling back to rule-based preprocessing: {e}")
        else:
            logger.info("No LLM model provided, using rule-based preprocessing")
    
    def extract_scene_context_with_llm(self, script_chunk: str, character: str, dialogue: str) -> str:
        """Use LLM to extract rich scene context for dialogue."""
        if not self.llm:
            return self._extract_scene_context_rules(script_chunk)
        
        try:
            prompt = f"""Analyze this Star Wars script excerpt and extract the scene context for the character's dialogue.

Script excerpt:
{script_chunk}

Character: {character}
Dialogue: "{dialogue}"

Extract and summarize:
1. Location/Setting (where the scene takes place)
2. Situation (what's happening in the scene)
3. Character's emotional state or motivation
4. Key visual or dramatic elements

Provide a concise 1-2 sentence context summary:"""

            response = self.llm.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=0.3,  # Low temperature for consistent context extraction
                stop=["\n\n", "Character:", "Script:"]
            )
            
            context = response.get('response', '').strip()
            if len(context) > 10:  # Valid response
                return context
            else:
                return self._extract_scene_context_rules(script_chunk)
                
        except Exception as e:
            logger.warning(f"LLM context extraction failed, using rules: {e}")
            return self._extract_scene_context_rules(script_chunk)
    
    def _extract_scene_context_rules(self, script_chunk: str) -> str:
        """Fallback rule-based scene context extraction."""
        lines = script_chunk.split('\n')
        scene_info = []
        location = None
        actions = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Scene headings
            if re.match(r'(INT\.|EXT\.)', line, re.IGNORECASE):
                location = re.sub(r'(INT\.|EXT\.)', '', line, flags=re.IGNORECASE).strip()
                location = re.sub(r'(DAY|NIGHT|MORNING|EVENING)', '', location, flags=re.IGNORECASE).strip()
                location = location.replace(' - ', ' ').strip()
            
            # Action lines (not dialogue, not scene headings)
            elif (len(line) > 10 and 
                  not line.isupper() and 
                  not re.match(r'^[A-Z][A-Z\s\-\']+$', line) and
                  not line.startswith('(')):
                actions.append(line[:80])  # Limit length
        
        if location:
            scene_info.append(f"Location: {location}")
        if actions:
            scene_info.append(f"Action: {' '.join(actions[:2])}")
        
        return " | ".join(scene_info) if scene_info else "Context unavailable"
    
    def enhance_dialogue_with_llm(self, character: str, dialogue: str, scene_context: str) -> Dict[str, str]:
        """Use LLM to enhance dialogue with character insights."""
        if not self.llm:
            return {
                'character': character,
                'dialogue': dialogue,
                'context': scene_context,
                'character_state': 'Unknown',
                'dialogue_type': 'Unknown'
            }
        
        try:
            prompt = f"""Analyze this Star Wars dialogue and provide character insights.

Character: {character}
Dialogue: "{dialogue}"
Scene Context: {scene_context}

Classify the dialogue and character state:
1. Character's emotional state (angry, calm, determined, fearful, etc.)
2. Dialogue type (command, question, statement, exclamation, etc.)
3. Character motivation in this moment

Respond in JSON format:
{{"character_state": "emotional_state", "dialogue_type": "type", "motivation": "brief_motivation"}}"""

            response = self.llm.generate(
                prompt=prompt,
                max_tokens=80,
                temperature=0.2,
                stop=["\n", "Character:", "Dialogue:"]
            )
            
            try:
                # Parse JSON response
                insights = json.loads(response.get('response', '{}'))
                return {
                    'character': character,
                    'dialogue': dialogue,
                    'context': scene_context,
                    'character_state': insights.get('character_state', 'Unknown'),
                    'dialogue_type': insights.get('dialogue_type', 'Unknown'),
                    'motivation': insights.get('motivation', 'Unknown')
                }
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    'character': character,
                    'dialogue': dialogue,
                    'context': scene_context,
                    'character_state': 'Unknown',
                    'dialogue_type': 'Unknown'
                }
                
        except Exception as e:
            logger.warning(f"LLM dialogue enhancement failed: {e}")
            return {
                'character': character,
                'dialogue': dialogue,
                'context': scene_context,
                'character_state': 'Unknown',
                'dialogue_type': 'Unknown'
            }
    
    def process_script_with_llm(self, script_path: Path) -> List[Dict[str, str]]:
        """Process script using LLM for intelligent dialogue extraction."""
        logger.info(f"Processing script with LLM enhancement: {script_path.name}")
        
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
            
            # Character name detection
            if re.match(r'^[A-Z][A-Z\s\-\']+$', line) and len(line) < 50:
                current_character = line.strip()
                continue
            
            # Dialogue detection
            if current_character and line and not line.isupper() and not line.startswith('('):
                if len(line) > 5 and not any(indicator in line.upper() for indicator in ['INT.', 'EXT.', 'CUT TO:', 'FADE']):
                    
                    # Get surrounding context for LLM
                    start_idx = max(0, i - 10)
                    end_idx = min(len(lines), i + 5)
                    context_chunk = '\n'.join(lines[start_idx:end_idx])
                    
                    # Clean dialogue
                    dialogue = re.sub(r'\([^)]*\)', '', line).strip()
                    dialogue = re.sub(r'\s+', ' ', dialogue).strip()
                    
                    if len(dialogue) > 10:
                        # Use LLM for scene context extraction
                        scene_context = self.extract_scene_context_with_llm(context_chunk, current_character, dialogue)
                        
                        # Use LLM for dialogue enhancement
                        enhanced_data = self.enhance_dialogue_with_llm(current_character, dialogue, scene_context)
                        enhanced_data['movie'] = script_path.stem.replace('_', ' ').title()
                        enhanced_data['line_number'] = i + 1
                        
                        processed_lines.append(enhanced_data)
                        current_character = None  # Reset after capturing dialogue
        
        logger.info(f"Extracted {len(processed_lines)} enhanced dialogue lines from {script_path.name}")
        return processed_lines
    
    def preprocess_all_scripts_with_llm(self) -> None:
        """Process all scripts using LLM enhancement."""
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
                processed = self.process_script_with_llm(script_path)
                all_processed.extend(processed)
            else:
                logger.warning(f"Script file not found: {filename}")
        
        # Save enhanced files
        for filename in trilogy_files:
            script_path = self.raw_scripts_dir / filename
            if script_path.exists():
                processed = self.process_script_with_llm(script_path)
                output_file = self.output_dir / f"llm_enhanced_{filename}"
                self.save_enhanced_script(processed, output_file)
        
        # Save combined file
        combined_file = self.output_dir / "llm_enhanced_trilogy_complete.txt"
        self.save_enhanced_script(all_processed, combined_file)
        
        logger.info(f"LLM preprocessing complete. Total enhanced lines: {len(all_processed)}")
    
    def save_enhanced_script(self, processed_lines: List[Dict[str, str]], output_path: Path) -> None:
        """Save LLM-enhanced dialogue lines to file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# LLM-Enhanced Star Wars Script with Rich Context\n")
                f.write("# Format: CHARACTER | DIALOGUE | CONTEXT | CHARACTER_STATE | DIALOGUE_TYPE | MOTIVATION | MOVIE\n\n")
                
                for line in processed_lines:
                    f.write(f"{line['character']} | {line['dialogue']} | {line['context']} | "
                           f"{line.get('character_state', 'Unknown')} | {line.get('dialogue_type', 'Unknown')} | "
                           f"{line.get('motivation', 'Unknown')} | {line['movie']}\n")
            
            logger.info(f"Saved {len(processed_lines)} LLM-enhanced lines to {output_path}")
        except Exception as e:
            logger.error(f"Error saving to {output_path}: {e}")


def main():
    """Main function for LLM-powered script preprocessing."""
    logging.basicConfig(level=logging.INFO)
    
    # Check if Phi-2 model is available
    model_path = "models/phi-2-q4.gguf"  # Adjust path as needed
    if not os.path.exists(model_path):
        print(f"⚠️  Phi-2 model not found at {model_path}")
        print("📥 Please download Phi-2 GGUF model or run without LLM enhancement")
        model_path = None
    
    preprocessor = LLMScriptPreprocessor(model_path=model_path)
    preprocessor.preprocess_all_scripts_with_llm()
    
    print("✅ LLM-powered script preprocessing completed!")
    print("📁 Enhanced scripts saved to: data/preprocessed/")
    print("🤖 Files created:")
    print("   - llm_enhanced_STAR WARS A NEW HOPE.txt")
    print("   - llm_enhanced_THE EMPIRE STRIKES BACK.txt") 
    print("   - llm_enhanced_STAR WARS THE RETURN OF THE JEDI.txt")
    print("   - llm_enhanced_trilogy_complete.txt")


if __name__ == "__main__":
    main()
