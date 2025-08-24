"""
Improved script processor for extracting real Star Wars dialogue from script files.

This module provides sophisticated parsing logic to extract authentic character dialogue
from Star Wars script files and prepare it for use in the RAG system.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class DialogueLine:
    """Represents a single line of dialogue from a script."""
    character: str
    dialogue: str
    line_number: int
    scene: str
    movie: str
    normalized_character: str


class ImprovedScriptProcessor:
    """
    Advanced script processor that extracts real dialogue from Star Wars scripts.
    
    Features:
    - Sophisticated character name detection and normalization
    - Scene tracking and context preservation
    - Multi-line dialogue handling
    - Comprehensive character mapping
    """
    
    def __init__(self):
        """Initialize the processor with character mappings and patterns."""
        self.character_mapping = self._build_character_mapping()
        self.scene_descriptors = {
            'INT.', 'EXT.', 'FADE', 'CUT', 'DISSOLVE', 'CONTINUED',
            'INTERIOR', 'EXTERIOR', 'SCENE', 'SEQUENCE'
        }
        self.direction_words = {
            'CLOSE', 'WIDE', 'ANGLE', 'SHOT', 'CAMERA', 'PAN', 'ZOOM',
            'TRACK', 'DOLLY', 'CRANE', 'STEADICAM', 'HANDHELD'
        }
        
    def _build_character_mapping(self) -> Dict[str, str]:
        """Build comprehensive character name mapping."""
        return {
            # Droids
            'THREEPIO': 'C-3PO',
            'C-3PO': 'C-3PO',
            'SEE-THREEPIO': 'C-3PO',
            'ARTOO': 'R2-D2',
            'R2-D2': 'R2-D2',
            'ARTOO-DETOO': 'R2-D2',
            'ARTOO DETOO': 'R2-D2',
            
            # Main Characters
            'LUKE': 'Luke Skywalker',
            'LUKE SKYWALKER': 'Luke Skywalker',
            'LEIA': 'Princess Leia',
            'PRINCESS LEIA': 'Princess Leia',
            'PRINCESS LEIA ORGANA': 'Princess Leia',
            'HAN': 'Han Solo',
            'HAN SOLO': 'Han Solo',
            'CHEWBACCA': 'Chewbacca',
            'CHEWIE': 'Chewbacca',
            
            # Jedi/Sith
            'BEN': 'Obi-Wan Kenobi',
            'OBI-WAN': 'Obi-Wan Kenobi',
            'OBI-WAN KENOBI': 'Obi-Wan Kenobi',
            'VADER': 'Darth Vader',
            'DARTH VADER': 'Darth Vader',
            'YODA': 'Yoda',
            'EMPEROR': 'Emperor Palpatine',
            'PALPATINE': 'Emperor Palpatine',
            'EMPEROR PALPATINE': 'Emperor Palpatine',
            
            # Empire Characters
            'TARKIN': 'Grand Moff Tarkin',
            'GRAND MOFF TARKIN': 'Grand Moff Tarkin',
            'PIETT': 'Admiral Piett',
            'ADMIRAL PIETT': 'Admiral Piett',
            'OZZEL': 'Admiral Ozzel',
            'ADMIRAL OZZEL': 'Admiral Ozzel',
            'NEEDA': 'Captain Needa',
            'CAPTAIN NEEDA': 'Captain Needa',
            'VEERS': 'General Veers',
            'GENERAL VEERS': 'General Veers',
            
            # Rebels/Supporting Characters
            'DODONNA': 'General Dodonna',
            'GENERAL DODONNA': 'General Dodonna',
            'MOTHMA': 'Mon Mothma',
            'MON MOTHMA': 'Mon Mothma',
            'ACKBAR': 'Admiral Ackbar',
            'ADMIRAL ACKBAR': 'Admiral Ackbar',
            'BIGGS': 'Biggs Darklighter',
            'BIGGS DARKLIGHTER': 'Biggs Darklighter',
            'WEDGE': 'Wedge Antilles',
            'WEDGE ANTILLES': 'Wedge Antilles',
            
            # Other Characters
            'FIXER': 'Fixer',
            'CAMIE': 'Camie',
            'DEAK': 'Deak',
            'WINDY': 'Windy',
            'TROOPER': 'Stormtrooper',
            'STORMTROOPER': 'Stormtrooper',
            'IMPERIAL OFFICER': 'Imperial Officer',
            'REBEL OFFICER': 'Rebel Officer',
            'CHIEF PILOT': 'Chief Pilot',
            'CAPTAIN': 'Captain',
            'WOMAN': 'Woman',
            'MAN': 'Man',
            'BOY': 'Boy',
            'GIRL': 'Girl',
        }
    
    def normalize_character_name(self, name: str) -> str:
        """
        Normalize character names to consistent format.
        
        Args:
            name: Raw character name from script
            
        Returns:
            Normalized character name
        """
        # Clean up the name
        name = name.strip().upper()
        
        # Remove common suffixes and prefixes
        name = re.sub(r'\s+', ' ', name)  # Normalize whitespace
        name = re.sub(r'\([^)]*\)', '', name)  # Remove parentheticals
        name = name.strip()
        
        # Check mapping first
        if name in self.character_mapping:
            return self.character_mapping[name]
        
        # Handle variations
        if name.endswith('S'):
            # Try singular form
            singular = name[:-1]
            if singular in self.character_mapping:
                return self.character_mapping[singular]
        
        # Default: return title case of original
        return name.title()
    
    def is_character_line(self, line: str) -> bool:
        """
        Determine if a line represents a character speaking.
        
        Args:
            line: Line from script
            
        Returns:
            True if line is character dialogue
        """
        line = line.strip()
        
        # Must be non-empty
        if not line:
            return False
        
        # Must be reasonably short (character names aren't paragraphs)
        if len(line) > 50:
            return False
        
        # Must be all caps (character names in scripts are capitalized)
        if not line.isupper():
            return False
        
        # Must not be scene direction
        if any(descriptor in line for descriptor in self.scene_descriptors):
            return False
        
        # Must not be camera direction
        if any(word in line for word in self.direction_words):
            return False
        
        # Must not contain certain patterns
        if re.search(r'[0-9]', line):  # No numbers
            return False
        
        if re.search(r'[^\w\s\-\']', line):  # Only letters, spaces, hyphens, apostrophes
            return False
        
        # Must be reasonable length for a character name
        if len(line) < 2 or len(line) > 30:
            return False
        
        return True
    
    def extract_dialogue_from_script(self, script_path: Path, movie_name: str) -> List[DialogueLine]:
        """
        Extract dialogue from a script file.
        
        Args:
            script_path: Path to script file
            movie_name: Name of the movie
            
        Returns:
            List of dialogue lines with metadata
        """
        dialogue_lines = []
        
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            current_scene = "Unknown"
            line_number = 0
            
            for line in lines:
                line_number += 1
                line = line.strip()
                
                # Track scene changes
                if line.startswith('INT.') or line.startswith('EXT.'):
                    current_scene = line[:100]  # Truncate long scene descriptions
                    continue
                
                # Skip empty lines
                if not line:
                    continue
                
                # Look for character dialogue pattern: CHARACTER_NAME followed by dialogue
                # Pattern: ALL CAPS character name followed by dialogue on same line
                char_match = re.match(r'^([A-Z][A-Z\s\-\']+?)\s+(.+)$', line)
                
                if char_match:
                    character = char_match.group(1).strip()
                    dialogue = char_match.group(2).strip()
                    
                    # Filter criteria for valid character lines
                    if self._is_valid_dialogue_line(character, dialogue):
                        normalized_char = self.normalize_character_name(character)
                        dialogue_lines.append(DialogueLine(
                            character=character,
                            dialogue=dialogue,
                            line_number=line_number,
                            scene=current_scene,
                            movie=movie_name,
                            normalized_character=normalized_char
                        ))
            
            logger.info(f"Extracted {len(dialogue_lines)} dialogue lines from {movie_name}")
            return dialogue_lines
            
        except Exception as e:
            logger.error(f"Error processing script {script_path}: {e}")
            return []
    
    def _is_valid_dialogue_line(self, character: str, dialogue: str) -> bool:
        """
        Validate if a line contains valid character dialogue.
        
        Args:
            character: Character name
            dialogue: Dialogue text
            
        Returns:
            True if valid dialogue line
        """
        # Character must be all caps and reasonable length
        if not character.isupper() or len(character) < 2 or len(character) > 30:
            return False
        
        # Character must not be scene direction
        if any(descriptor in character for descriptor in self.scene_descriptors):
            return False
        
        # Character must not be camera direction
        if any(word in character for word in self.direction_words):
            return False
        
        # Character must not contain numbers or special characters (except hyphens and apostrophes)
        if re.search(r'[0-9]', character) or re.search(r'[^\w\s\-\']', character):
            return False
        
        # Dialogue must be reasonable length
        if len(dialogue) < 5 or len(dialogue) > 500:
            return False
        
        # Dialogue must not be all caps (that would be scene description)
        if dialogue.isupper():
            return False
        
        return True
    
    def get_character_statistics(self, dialogue_lines: List[DialogueLine]) -> Dict[str, int]:
        """
        Get statistics about character dialogue frequency.
        
        Args:
            dialogue_lines: List of dialogue lines
            
        Returns:
            Dictionary mapping character names to line counts
        """
        stats = {}
        for line in dialogue_lines:
            char = line.normalized_character
            stats[char] = stats.get(char, 0) + 1
        return stats
    
    def filter_by_character(self, dialogue_lines: List[DialogueLine], 
                          characters: Set[str]) -> List[DialogueLine]:
        """
        Filter dialogue lines to only include specified characters.
        
        Args:
            dialogue_lines: List of dialogue lines
            characters: Set of character names to include
            
        Returns:
            Filtered list of dialogue lines
        """
        return [line for line in dialogue_lines if line.normalized_character in characters]
    
    def export_to_json(self, dialogue_lines: List[DialogueLine], output_path: Path):
        """
        Export dialogue lines to JSON format for use in RAG system.
        
        Args:
            dialogue_lines: List of dialogue lines
            output_path: Path to output JSON file
        """
        data = []
        for line in dialogue_lines:
            data.append({
                'character': line.normalized_character,
                'dialogue': line.dialogue,
                'scene': line.scene,
                'movie': line.movie,
                'line_number': line.line_number,
                'original_character': line.character
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(data)} dialogue lines to {output_path}")


def process_all_scripts(scripts_dir: Path, output_dir: Path):
    """
    Process all script files in a directory.
    
    Args:
        scripts_dir: Directory containing script files
        output_dir: Directory to save processed data
    """
    processor = ImprovedScriptProcessor()
    all_dialogue = []
    
    # Process each script file
    for script_file in scripts_dir.glob("*.txt"):
        movie_name = script_file.stem.replace('_', ' ').title()
        logger.info(f"Processing {movie_name}...")
        
        dialogue_lines = processor.extract_dialogue_from_script(script_file, movie_name)
        all_dialogue.extend(dialogue_lines)
    
    # Get statistics
    stats = processor.get_character_statistics(all_dialogue)
    logger.info("Character dialogue statistics:")
    for char, count in sorted(stats.items(), key=lambda x: x[1], reverse=True)[:20]:
        logger.info(f"  {char}: {count} lines")
    
    # Export all dialogue
    output_dir.mkdir(exist_ok=True)
    processor.export_to_json(all_dialogue, output_dir / "all_dialogue.json")
    
    # Export statistics
    with open(output_dir / "character_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    return all_dialogue


if __name__ == "__main__":
    # Example usage
    scripts_dir = Path("data/raw")
    output_dir = Path("data/processed")
    
    if scripts_dir.exists():
        dialogue_lines = process_all_scripts(scripts_dir, output_dir)
        print(f"Processed {len(dialogue_lines)} total dialogue lines")
    else:
        print(f"Scripts directory not found: {scripts_dir}")
