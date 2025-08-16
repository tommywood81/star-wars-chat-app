"""
Data processing module for Star Wars script dialogue extraction and cleaning.

This module extracts character dialogue from Star Wars script files and
prepares it for use in the RAG system.
"""

import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DialogueProcessor:
    """Process Star Wars script files to extract clean character dialogue."""
    
    def __init__(self):
        """Initialize the dialogue processor."""
        self.character_mapping = {
            'THREEPIO': 'C-3PO',
            'C-3PO': 'C-3PO',
            'SEE-THREEPIO': 'C-3PO',
            'ARTOO': 'R2-D2',
            'R2-D2': 'R2-D2',
            'ARTOO-DETOO': 'R2-D2',
            'LUKE': 'Luke Skywalker',
            'LEIA': 'Princess Leia',
            'PRINCESS LEIA': 'Princess Leia',
            'HAN': 'Han Solo',
            'HAN SOLO': 'Han Solo',
            'BEN': 'Obi-Wan Kenobi',
            'OBI-WAN': 'Obi-Wan Kenobi',
            'VADER': 'Darth Vader',
            'DARTH VADER': 'Darth Vader',
            'CHEWBACCA': 'Chewbacca',
            'CHEWIE': 'Chewbacca'
        }
    
    def load_script(self, script_path: Union[str, Path]) -> str:
        """Load script text from file.
        
        Args:
            script_path: Path to the script file
            
        Returns:
            Script text content
            
        Raises:
            FileNotFoundError: If script file doesn't exist
            UnicodeDecodeError: If file encoding is problematic
        """
        script_path = Path(script_path)
        if not script_path.exists():
            raise FileNotFoundError(f"Script file not found: {script_path}")
        
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Loaded script: {script_path.name} ({len(content):,} characters)")
            return content
        except UnicodeDecodeError:
            # Try with different encoding
            with open(script_path, 'r', encoding='latin-1') as f:
                content = f.read()
            logger.warning(f"Used latin-1 encoding for {script_path.name}")
            return content
    
    def extract_dialogue_lines(self, script_text: str, movie_name: str = "Unknown") -> List[Dict]:
        """Extract character dialogue from script text.
        
        Args:
            script_text: Raw script content
            movie_name: Name of the movie for metadata
            
        Returns:
            List of dialogue dictionaries with metadata
        """
        lines = script_text.splitlines()
        dialogue_data = []
        current_scene = "Unknown"
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Track scene changes
            if line.startswith('INT.') or line.startswith('EXT.'):
                current_scene = line[:50]  # Truncate long scene descriptions
                continue
                
            # Skip empty lines
            if not line:
                continue
                
            # Look for character dialogue pattern
            char_match = re.match(r'^([A-Z][A-Z\s\-\']+?)\s+(.+)$', line)
            
            if char_match:
                character = char_match.group(1).strip()
                dialogue = char_match.group(2).strip()
                
                # Filter criteria for valid character lines
                if self._is_valid_dialogue_line(character, dialogue):
                    dialogue_data.append({
                        'line_number': i + 1,
                        'scene': current_scene,
                        'character': character,
                        'dialogue': dialogue,
                        'movie': movie_name
                    })
        
        logger.info(f"Extracted {len(dialogue_data)} dialogue lines from {movie_name}")
        return dialogue_data
    
    def _is_valid_dialogue_line(self, character: str, dialogue: str) -> bool:
        """Check if a character/dialogue pair is valid.
        
        Args:
            character: Character name
            dialogue: Dialogue text
            
        Returns:
            True if valid dialogue line
        """
        return (
            len(character) <= 25 and  # Reasonable name length
            len(dialogue) >= 5 and   # Minimum dialogue length
            not character.endswith('.') and  # Not scene direction
            not character.startswith('FADE') and  # Not script direction
            not character.startswith('CUT') and   # Not script direction
            not character.startswith('DISSOLVE')  # Not script direction
        )
    
    def normalize_character_name(self, name: str) -> str:
        """Normalize character names to consistent format.
        
        Args:
            name: Original character name
            
        Returns:
            Normalized character name
        """
        name = name.strip().upper()
        return self.character_mapping.get(name, name.title())
    
    def clean_dialogue_for_rag(self, dialogue_df: pd.DataFrame, 
                              min_lines_per_character: int = 5) -> pd.DataFrame:
        """Clean dialogue data for RAG pipeline.
        
        Args:
            dialogue_df: Raw dialogue DataFrame
            min_lines_per_character: Minimum lines to keep a character
            
        Returns:
            Cleaned dialogue DataFrame
        """
        clean_df = dialogue_df.copy()
        
        # Normalize character names
        clean_df['character_normalized'] = clean_df['character'].apply(
            self.normalize_character_name
        )
        
        # Add text metrics
        clean_df['word_count'] = clean_df['dialogue'].str.split().str.len()
        clean_df['char_length'] = clean_df['dialogue'].str.len()
        
        # Filter criteria for high-quality dialogue
        quality_mask = (
            (clean_df['word_count'] >= 3) &  # At least 3 words
            (clean_df['word_count'] <= 40) &  # Not too long
            (clean_df['char_length'] >= 10) &  # At least 10 characters
            (~clean_df['dialogue'].str.contains(r'^\(.*\)$', regex=True))  # Not just parenthetical
        )
        
        clean_df = clean_df[quality_mask].copy()
        
        # Clean dialogue text - remove parenthetical directions
        clean_df['dialogue_clean'] = clean_df['dialogue'].apply(self._clean_dialogue_text)
        
        # Remove lines where cleaning left very little content
        clean_df = clean_df[clean_df['dialogue_clean'].str.len() > 5]
        
        # Focus on characters with sufficient dialogue
        char_counts = clean_df['character_normalized'].value_counts()
        main_chars = char_counts[char_counts >= min_lines_per_character].index.tolist()
        clean_df = clean_df[clean_df['character_normalized'].isin(main_chars)]
        
        logger.info(f"Cleaned dataset: {len(clean_df)} lines, {len(main_chars)} characters")
        return clean_df
    
    def _clean_dialogue_text(self, text: str) -> str:
        """Clean individual dialogue text.
        
        Args:
            text: Raw dialogue text
            
        Returns:
            Cleaned dialogue text
        """
        # Remove parenthetical directions
        cleaned = re.sub(r'\([^)]*\)', '', text)
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    def process_script_file(self, script_path: Union[str, Path], 
                           movie_name: Optional[str] = None) -> pd.DataFrame:
        """Process a complete script file from raw text to clean dialogue.
        
        Args:
            script_path: Path to script file
            movie_name: Movie name (inferred from filename if not provided)
            
        Returns:
            Clean dialogue DataFrame ready for embedding
        """
        script_path = Path(script_path)
        
        if movie_name is None:
            movie_name = script_path.stem.replace('_', ' ').title()
        
        # Load and extract dialogue
        script_text = self.load_script(script_path)
        dialogue_data = self.extract_dialogue_lines(script_text, movie_name)
        
        # Convert to DataFrame and clean
        dialogue_df = pd.DataFrame(dialogue_data)
        if dialogue_df.empty:
            logger.warning(f"No dialogue extracted from {script_path.name}")
            return dialogue_df
        
        clean_df = self.clean_dialogue_for_rag(dialogue_df)
        
        logger.info(f"Processed {script_path.name}: {len(clean_df)} clean dialogue lines")
        return clean_df
    
    def process_multiple_scripts(self, script_directory: Union[str, Path], 
                               pattern: str = "*.txt") -> pd.DataFrame:
        """Process multiple script files and combine them.
        
        Args:
            script_directory: Directory containing script files
            pattern: File pattern to match
            
        Returns:
            Combined clean dialogue DataFrame
        """
        script_dir = Path(script_directory)
        if not script_dir.exists():
            raise FileNotFoundError(f"Script directory not found: {script_dir}")
        
        script_files = list(script_dir.glob(pattern))
        if not script_files:
            raise ValueError(f"No script files found matching {pattern} in {script_dir}")
        
        all_dialogue = []
        
        for script_file in script_files:
            try:
                df = self.process_script_file(script_file)
                if not df.empty:
                    all_dialogue.append(df)
            except Exception as e:
                logger.error(f"Failed to process {script_file.name}: {e}")
                continue
        
        if not all_dialogue:
            logger.warning("No dialogue data extracted from any files")
            return pd.DataFrame()
        
        # Combine all DataFrames
        combined_df = pd.concat(all_dialogue, ignore_index=True)
        
        logger.info(f"Combined {len(script_files)} scripts: {len(combined_df)} total dialogue lines")
        logger.info(f"Movies: {combined_df['movie'].unique().tolist()}")
        logger.info(f"Characters: {len(combined_df['character_normalized'].unique())}")
        
        return combined_df
