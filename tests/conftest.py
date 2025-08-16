"""
Pytest configuration and fixtures for Star Wars RAG tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from star_wars_rag.data_processor import DialogueProcessor
from star_wars_rag.embeddings import StarWarsEmbedder
from star_wars_rag.retrieval import DialogueRetriever


@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Get the test data directory."""
    return project_root / "data" / "raw"


@pytest.fixture(scope="session")
def sample_script_path(test_data_dir):
    """Get path to a sample script file."""
    script_path = test_data_dir / "STAR WARS A NEW HOPE.txt"
    if not script_path.exists():
        pytest.skip(f"Test script not found: {script_path}")
    return script_path


@pytest.fixture(scope="session")
def dialogue_processor():
    """Create a DialogueProcessor instance."""
    return DialogueProcessor()


@pytest.fixture(scope="session")
def embedder():
    """Create a StarWarsEmbedder instance."""
    return StarWarsEmbedder()


@pytest.fixture(scope="session")
def sample_dialogue_data(dialogue_processor, sample_script_path):
    """Load and process sample dialogue data."""
    return dialogue_processor.process_script_file(sample_script_path)


@pytest.fixture(scope="session")
def sample_embeddings(embedder, sample_dialogue_data):
    """Generate sample embeddings for test data."""
    if sample_dialogue_data.empty:
        return np.array([])
    
    texts = sample_dialogue_data['dialogue_clean'].tolist()[:50]  # Limit for speed
    return embedder.embed_batch(texts, show_progress=False)


@pytest.fixture(scope="session")
def retriever_with_data(sample_dialogue_data, sample_embeddings):
    """Create a DialogueRetriever with loaded data."""
    retriever = DialogueRetriever()
    if not sample_dialogue_data.empty and len(sample_embeddings) > 0:
        # Use subset for faster tests
        subset_data = sample_dialogue_data.head(50)
        retriever.load_dialogue_data(subset_data, sample_embeddings)
    return retriever


# Mock data fixtures for unit tests
@pytest.fixture
def mock_dialogue_data():
    """Create mock dialogue data for testing."""
    return pd.DataFrame([
        {
            'line_number': 1,
            'scene': 'INT. REBEL BLOCKADE RUNNER',
            'character': 'C-3PO',
            'dialogue': 'Did you hear that? They shut down the main reactor.',
            'dialogue_clean': 'Did you hear that? They shut down the main reactor.',
            'character_normalized': 'C-3PO',
            'movie': 'A New Hope',
            'word_count': 10,
            'char_length': 49
        },
        {
            'line_number': 2,
            'scene': 'INT. REBEL BLOCKADE RUNNER',
            'character': 'LUKE',
            'dialogue': 'I want to learn the ways of the Force.',
            'dialogue_clean': 'I want to learn the ways of the Force.',
            'character_normalized': 'Luke Skywalker',
            'movie': 'A New Hope',
            'word_count': 8,
            'char_length': 37
        },
        {
            'line_number': 3,
            'scene': 'INT. DEATH STAR',
            'character': 'VADER',
            'dialogue': 'I find your lack of faith disturbing.',
            'dialogue_clean': 'I find your lack of faith disturbing.',
            'character_normalized': 'Darth Vader',
            'movie': 'A New Hope',
            'word_count': 7,
            'char_length': 36
        }
    ])


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings for testing."""
    # Create simple mock embeddings (3 samples, 384 dimensions)
    np.random.seed(42)
    return np.random.random((3, 384)).astype(np.float32)


@pytest.fixture
def sample_script_text():
    """Sample script text for testing."""
    return """
STAR WARS: A NEW HOPE

FADE IN:

INT. REBEL BLOCKADE RUNNER - MAIN PASSAGEWAY

The awesome yellow planet of Tatooine emerges from a total
eclipse, her two moons glowing against the darkness. A tiny
silver spacecraft, a Rebel blockade runner firing lasers
from the back of the ship, races through space. It is
pursued by a giant Imperial Starship. Hundreds of deadly
laserbolts streak from the Imperial ship, causing the main
solar fin of the Rebel craft to disintegrate.

C-3PO                    Did you hear that? They shut down the main reactor.

ARTOO                    We'll be destroyed for sure. This is madness!

THREEPIO                 We're doomed! There'll be no escape for the Princess this time.

LEIA                     Help me, Obi-Wan Kenobi, you're my only hope.

VADER                    I find your lack of faith disturbing.

LUKE                     I want to learn the ways of the Force and be a Jedi like my father.

HAN                      Never tell me the odds!
"""
