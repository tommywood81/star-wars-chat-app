#!/usr/bin/env python3
"""
Tests for script preprocessor functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from src.star_wars_rag.script_preprocessor import ScriptPreprocessor


class TestScriptPreprocessor:
    """Test the script preprocessor functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.raw_dir = Path(self.temp_dir) / "raw"
        self.output_dir = Path(self.temp_dir) / "processed"
        self.raw_dir.mkdir(exist_ok=True)
        
        # Create test script content
        self.test_script_content = """
INT. REBEL BLOCKADE RUNNER - MAIN PASSAGEWAY

The ship is under attack. Laser fire exchanges in the corridor.

CAPTAIN ANTILLES
Did you hear that? They've shut down the main reactor.

He speaks into his comlink urgently.

CAPTAIN ANTILLES
We'll be destroyed for sure. This is madness!

EXT. TATOOINE - DESERT WASTELAND - DAY

Two droids wander across the barren landscape.

C-3PO
We seem to be made to suffer. It's our lot in life.

R2-D2 beeps and whistles in response.

C-3PO
I suggest a new strategy, R2. Let the Wookiee win.
"""
        
        # Write test script
        test_file = self.raw_dir / "TEST_SCRIPT.txt"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(self.test_script_content)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initializes correctly."""
        preprocessor = ScriptPreprocessor(str(self.raw_dir), str(self.output_dir))
        assert preprocessor.raw_scripts_dir == self.raw_dir
        assert preprocessor.output_dir == self.output_dir
        assert self.output_dir.exists()
    
    def test_extract_scene_context(self):
        """Test scene context extraction."""
        preprocessor = ScriptPreprocessor(str(self.raw_dir), str(self.output_dir))
        lines = self.test_script_content.split('\n')
        
        # Test context extraction for a dialogue line
        context = preprocessor.extract_scene_context(lines, 5)  # Captain Antilles line
        assert "REBEL BLOCKADE RUNNER" in context or "MAIN PASSAGEWAY" in context
    
    def test_process_script(self):
        """Test script processing functionality."""
        preprocessor = ScriptPreprocessor(str(self.raw_dir), str(self.output_dir))
        test_file = self.raw_dir / "TEST_SCRIPT.txt"
        
        processed_lines = preprocessor.process_script(test_file)
        
        # Should extract dialogue lines
        assert len(processed_lines) > 0
        
        # Check structure of processed lines
        for line in processed_lines:
            assert 'character' in line
            assert 'dialogue' in line
            assert 'context' in line
            assert 'movie' in line
            assert len(line['dialogue']) > 5  # Substantial dialogue
    
    def test_character_recognition(self):
        """Test character name recognition."""
        preprocessor = ScriptPreprocessor(str(self.raw_dir), str(self.output_dir))
        test_file = self.raw_dir / "TEST_SCRIPT.txt"
        
        processed_lines = preprocessor.process_script(test_file)
        
        # Should recognize character names
        character_names = [line['character'] for line in processed_lines]
        assert any('ANTILLES' in name for name in character_names)
        assert any('C-3PO' in name for name in character_names)
    
    def test_save_processed_script(self):
        """Test saving processed script."""
        preprocessor = ScriptPreprocessor(str(self.raw_dir), str(self.output_dir))
        
        test_data = [
            {
                'character': 'LUKE SKYWALKER',
                'dialogue': 'I want to learn the ways of the Force.',
                'context': 'Scene: Dagobah Swamp | Action: Training with Yoda',
                'movie': 'The Empire Strikes Back'
            }
        ]
        
        output_file = self.output_dir / "test_output.txt"
        preprocessor.save_processed_script(test_data, output_file)
        
        assert output_file.exists()
        
        # Check content
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert 'LUKE SKYWALKER' in content
            assert 'I want to learn the ways of the Force' in content
            assert 'Dagobah Swamp' in content


def test_preprocessed_files_exist():
    """Test that preprocessed files were created successfully."""
    preprocessed_dir = Path("data/preprocessed")
    assert preprocessed_dir.exists()
    
    expected_files = [
        "enhanced_STAR WARS A NEW HOPE.txt",
        "enhanced_THE EMPIRE STRIKES BACK.txt", 
        "enhanced_STAR WARS THE RETURN OF THE JEDI.txt",
        "enhanced_original_trilogy_combined.txt"
    ]
    
    for filename in expected_files:
        file_path = preprocessed_dir / filename
        assert file_path.exists(), f"Missing file: {filename}"
        assert file_path.stat().st_size > 0, f"Empty file: {filename}"


def test_enhanced_script_format():
    """Test the format of enhanced script files."""
    combined_file = Path("data/preprocessed/enhanced_original_trilogy_combined.txt")
    
    with open(combined_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Should have header
    assert lines[0].startswith("# Enhanced Star Wars Script")
    assert "CHARACTER | DIALOGUE | CONTEXT | MOVIE" in lines[1]
    
    # Check data lines format
    data_lines = [line for line in lines[3:] if line.strip() and not line.startswith('#')]
    assert len(data_lines) > 0
    
    for line in data_lines[:5]:  # Check first 5 data lines
        parts = line.split(' | ')
        assert len(parts) >= 4, f"Invalid format: {line}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
