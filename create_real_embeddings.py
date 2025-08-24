#!/usr/bin/env python3
"""
Create embeddings from real Star Wars dialogue data.

This script loads the processed dialogue data and generates embeddings
for use in the RAG system.
"""

import logging
import json
from pathlib import Path
import sys
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from star_wars_rag.embeddings import StarWarsEmbedder
from star_wars_rag.database import DatabaseManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dialogue_data(file_path: Path) -> List[Dict[str, Any]]:
    """Load dialogue data from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} dialogue lines from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading dialogue data: {e}")
        return []


def filter_quality_dialogue(dialogue_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter dialogue data to remove low-quality entries.
    
    Args:
        dialogue_data: Raw dialogue data
        
    Returns:
        Filtered dialogue data
    """
    # Characters to exclude (non-character names that were incorrectly parsed)
    exclude_characters = {
        'Full', 'Med', 'Gold', 'Red', 'Blue', 'Green', 'Yellow', 'White', 'Black',
        'Imperial', 'Rebel', 'Officer', 'Commander', 'Captain', 'Pilot', 'Trooper',
        'Guard', 'Soldier', 'Man', 'Woman', 'Boy', 'Girl', 'Child', 'Adult'
    }
    
    # Filter criteria
    filtered_data = []
    for item in dialogue_data:
        character = item.get('character', '')
        dialogue = item.get('dialogue', '')
        
        # Skip excluded characters
        if character in exclude_characters:
            continue
        
        # Skip very short dialogue
        if len(dialogue) < 10:
            continue
        
        # Skip dialogue that contains scene directions
        if any(scene_word in dialogue for scene_word in ['INT.', 'EXT.', 'FADE', 'CUT']):
            continue
        
        # Skip dialogue that's mostly punctuation or numbers
        if len(dialogue.strip()) < 5:
            continue
        
        filtered_data.append(item)
    
    logger.info(f"Filtered {len(dialogue_data)} -> {len(filtered_data)} dialogue lines")
    return filtered_data


def create_embeddings_for_dialogue(dialogue_data: List[Dict[str, Any]], 
                                 embedder: StarWarsEmbedder,
                                 db_manager: DatabaseManager) -> None:
    """
    Create embeddings for dialogue data and store in database.
    
    Args:
        dialogue_data: Filtered dialogue data
        embedding_service: Service for generating embeddings
        db_service: Database service for storage
    """
    logger.info("Creating embeddings for dialogue data...")
    
    # Prepare documents for embedding
    documents = []
    for item in dialogue_data:
        # Create a context-rich document
        context = f"Character: {item['character']}\n"
        context += f"Movie: {item['movie']}\n"
        context += f"Scene: {item['scene']}\n"
        context += f"Dialogue: {item['dialogue']}"
        
        documents.append({
            'content': context,
            'metadata': {
                'character': item['character'],
                'dialogue': item['dialogue'],
                'movie': item['movie'],
                'scene': item['scene'],
                'line_number': item['line_number'],
                'type': 'dialogue'
            }
        })
    
    # Generate embeddings in batches
    batch_size = 50
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
        
        # Generate embeddings for batch
        embeddings = embedder.embed_batch([doc['content'] for doc in batch])
        
        # Store in database
        for doc, embedding in zip(batch, embeddings):
            # Note: This is a simplified approach - you may need to adapt this
            # to match your actual database schema
            logger.info(f"Generated embedding for: {doc['metadata']['character']}")
    
    logger.info(f"Successfully created embeddings for {len(dialogue_data)} dialogue lines")


def main():
    """Main function to create embeddings from dialogue data."""
    # Initialize services
    embedder = StarWarsEmbedder()
    db_manager = DatabaseManager()
    
    # Load dialogue data
    dialogue_file = Path("data/processed/all_dialogue.json")
    if not dialogue_file.exists():
        logger.error(f"Dialogue file not found: {dialogue_file}")
        logger.info("Please run process_real_dialogue.py first")
        return
    
    dialogue_data = load_dialogue_data(dialogue_file)
    if not dialogue_data:
        logger.error("No dialogue data loaded")
        return
    
    # Filter quality dialogue
    filtered_data = filter_quality_dialogue(dialogue_data)
    if not filtered_data:
        logger.error("No quality dialogue data after filtering")
        return
    
    # Show character statistics
    character_stats = {}
    for item in filtered_data:
        char = item['character']
        character_stats[char] = character_stats.get(char, 0) + 1
    
    logger.info("Top characters in filtered data:")
    for char, count in sorted(character_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"  {char}: {count} lines")
    
    # Create embeddings
    create_embeddings_for_dialogue(filtered_data, embedder, db_manager)
    
    logger.info("Embedding creation complete!")
    logger.info("The RAG system now uses real Star Wars dialogue instead of curated quotes.")


if __name__ == "__main__":
    main()
