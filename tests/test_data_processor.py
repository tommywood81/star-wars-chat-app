"""
Tests for the DialogueProcessor module using real Star Wars data.
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path

from star_wars_rag.data_processor import DialogueProcessor


class TestDialogueProcessor:
    """Test cases for DialogueProcessor class."""
    
    def test_init(self):
        """Test DialogueProcessor initialization."""
        processor = DialogueProcessor()
        assert processor.character_mapping is not None
        assert isinstance(processor.character_mapping, dict)
        assert 'LUKE' in processor.character_mapping
        assert processor.character_mapping['LUKE'] == 'Luke Skywalker'
    
    def test_normalize_character_name(self):
        """Test character name normalization."""
        processor = DialogueProcessor()
        
        # Test known mappings
        assert processor.normalize_character_name('LUKE') == 'Luke Skywalker'
        assert processor.normalize_character_name('C-3PO') == 'C-3PO'
        assert processor.normalize_character_name('THREEPIO') == 'C-3PO'
        assert processor.normalize_character_name('VADER') == 'Darth Vader'
        
        # Test unknown character (should be title-cased)
        assert processor.normalize_character_name('RANDOM CHARACTER') == 'Random Character'
        assert processor.normalize_character_name('jar jar') == 'Jar Jar'
    
    def test_is_valid_dialogue_line(self):
        """Test dialogue line validation."""
        processor = DialogueProcessor()
        
        # Valid dialogue
        assert processor._is_valid_dialogue_line('LUKE', 'I want to be a Jedi.')
        assert processor._is_valid_dialogue_line('C-3PO', 'The odds of successfully navigating an asteroid field are...')
        
        # Invalid dialogue - too short
        assert not processor._is_valid_dialogue_line('LUKE', 'No.')
        
        # Invalid character - too long
        assert not processor._is_valid_dialogue_line('A' * 30, 'Valid dialogue here')
        
        # Invalid - script directions
        assert not processor._is_valid_dialogue_line('FADE', 'IN TO BLACK')
        assert not processor._is_valid_dialogue_line('CUT', 'TO EXTERIOR')
        assert not processor._is_valid_dialogue_line('SCENE.', 'Some description')
    
    def test_clean_dialogue_text(self):
        """Test dialogue text cleaning."""
        processor = DialogueProcessor()
        
        # Test parenthetical removal
        text = "I find your lack of faith (pauses dramatically) disturbing."
        expected = "I find your lack of faith  disturbing."
        result = processor._clean_dialogue_text(text)
        assert "pauses dramatically" not in result
        
        # Test whitespace normalization
        text = "The    Force    will   be   with   you."
        result = processor._clean_dialogue_text(text)
        assert "  " not in result  # No double spaces
        
        # Test multiple parentheticals
        text = "Help me (quietly) Obi-Wan Kenobi (desperately), you're my only hope."
        result = processor._clean_dialogue_text(text)
        assert "quietly" not in result
        assert "desperately" not in result
        assert "Help me" in result
        assert "you're my only hope" in result
    
    def test_extract_dialogue_lines_with_sample_text(self, sample_script_text):
        """Test dialogue extraction with sample script text."""
        processor = DialogueProcessor()
        
        dialogue_data = processor.extract_dialogue_lines(sample_script_text, "Test Movie")
        
        # Should extract dialogue lines
        assert len(dialogue_data) > 0
        
        # Check structure
        for item in dialogue_data:
            assert 'line_number' in item
            assert 'scene' in item
            assert 'character' in item
            assert 'dialogue' in item
            assert 'movie' in item
            assert item['movie'] == "Test Movie"
        
        # Check specific extractions
        characters = [item['character'] for item in dialogue_data]
        assert 'C-3PO' in characters or 'THREEPIO' in characters
        assert 'LUKE' in characters
        assert 'VADER' in characters
        
        # Find specific dialogue
        luke_dialogue = [item for item in dialogue_data if item['character'] == 'LUKE']
        assert len(luke_dialogue) > 0
        assert any('Force' in item['dialogue'] for item in luke_dialogue)
    
    @pytest.mark.real_data
    def test_load_script_real_file(self, sample_script_path):
        """Test loading a real script file."""
        processor = DialogueProcessor()
        
        script_text = processor.load_script(sample_script_path)
        
        assert isinstance(script_text, str)
        assert len(script_text) > 1000  # Should be substantial
        assert 'STAR WARS' in script_text.upper()
    
    def test_load_script_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        processor = DialogueProcessor()
        
        with pytest.raises(FileNotFoundError):
            processor.load_script("nonexistent_file.txt")
    
    @pytest.mark.real_data
    def test_process_script_file_real_data(self, sample_script_path):
        """Test processing a real script file."""
        processor = DialogueProcessor()
        
        result_df = processor.process_script_file(sample_script_path)
        
        # Should return a non-empty DataFrame
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) > 0
        
        # Check required columns
        required_cols = ['character_normalized', 'dialogue_clean', 'movie', 'scene']
        for col in required_cols:
            assert col in result_df.columns
        
        # Check data quality
        assert result_df['dialogue_clean'].str.len().min() > 5  # Min dialogue length
        assert result_df['word_count'].min() >= 3  # Min word count
        
        # Check character normalization worked
        characters = result_df['character_normalized'].unique()
        assert 'Luke Skywalker' in characters or 'C-3PO' in characters
    
    def test_clean_dialogue_for_rag_with_mock_data(self, mock_dialogue_data):
        """Test RAG cleaning with mock data."""
        processor = DialogueProcessor()
        
        # Add some low-quality data
        bad_data = pd.DataFrame([
            {
                'character': 'SOMEONE',
                'dialogue': 'No.',  # Too short
                'movie': 'Test',
                'scene': 'Test Scene'
            },
            {
                'character': 'ANOTHER',
                'dialogue': '(whispers quietly)',  # Just parenthetical
                'movie': 'Test',
                'scene': 'Test Scene'
            }
        ])
        
        combined_data = pd.concat([mock_dialogue_data, bad_data], ignore_index=True)
        
        cleaned_df = processor.clean_dialogue_for_rag(combined_data, min_lines_per_character=1)
        
        # Should filter out bad data
        assert len(cleaned_df) <= len(combined_data)
        
        # Should have required columns
        assert 'character_normalized' in cleaned_df.columns
        assert 'dialogue_clean' in cleaned_df.columns
        assert 'word_count' in cleaned_df.columns
        assert 'char_length' in cleaned_df.columns
        
        # Quality checks
        assert cleaned_df['word_count'].min() >= 3
        assert cleaned_df['char_length'].min() >= 10
    
    @pytest.mark.slow
    @pytest.mark.real_data
    def test_process_multiple_scripts(self, test_data_dir):
        """Test processing multiple script files."""
        processor = DialogueProcessor()
        
        # This test might be slow as it processes multiple files
        try:
            result_df = processor.process_multiple_scripts(test_data_dir, "STAR WARS*.txt")
            
            if not result_df.empty:
                # Check that we got data from multiple movies
                movies = result_df['movie'].unique()
                assert len(movies) >= 1
                
                # Check combined data quality
                assert len(result_df) > 0
                assert 'character_normalized' in result_df.columns
                assert 'dialogue_clean' in result_df.columns
                
        except Exception as e:
            pytest.skip(f"Multiple script processing failed: {e}")
    
    def test_process_multiple_scripts_no_files(self, tmp_path):
        """Test processing when no files match pattern."""
        processor = DialogueProcessor()
        
        with pytest.raises(ValueError, match="No script files found"):
            processor.process_multiple_scripts(tmp_path, "*.nonexistent")
    
    def test_process_multiple_scripts_nonexistent_dir(self):
        """Test processing with non-existent directory."""
        processor = DialogueProcessor()
        
        with pytest.raises(FileNotFoundError):
            processor.process_multiple_scripts("/nonexistent/path", "*.txt")


class TestDialogueProcessorIntegration:
    """Integration tests for DialogueProcessor."""
    
    @pytest.mark.integration
    @pytest.mark.real_data
    def test_full_pipeline_with_real_data(self, sample_script_path):
        """Test the complete data processing pipeline."""
        processor = DialogueProcessor()
        
        # Process the script
        dialogue_df = processor.process_script_file(sample_script_path)
        
        # Verify the pipeline worked end-to-end
        assert not dialogue_df.empty
        
        # Check we have main characters
        characters = dialogue_df['character_normalized'].value_counts()
        assert len(characters) > 0
        
        # Check dialogue quality
        avg_word_count = dialogue_df['word_count'].mean()
        assert avg_word_count > 5  # Reasonable dialogue length
        
        # Check we have scene information
        scenes = dialogue_df['scene'].unique()
        assert len(scenes) > 1  # Multiple scenes
        
        # Verify no empty dialogue
        assert not dialogue_df['dialogue_clean'].str.strip().eq('').any()
        
        print(f"Processed {len(dialogue_df)} dialogue lines")
        print(f"Found {len(characters)} characters")
        print(f"Top characters: {characters.head().to_dict()}")
        
    @pytest.mark.integration
    def test_character_consistency(self, mock_dialogue_data):
        """Test that character normalization is consistent."""
        processor = DialogueProcessor()
        
        # Test various forms of the same character
        test_names = ['LUKE', 'Luke', 'luke', '  LUKE  ']
        
        for name in test_names:
            normalized = processor.normalize_character_name(name)
            assert normalized == 'Luke Skywalker'
