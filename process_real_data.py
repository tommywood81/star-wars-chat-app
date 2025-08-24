#!/usr/bin/env python3
"""
Process real Star Wars script data and populate database with authentic dialogue.
"""

import asyncio
import asyncpg
import logging
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StarWarsDataProcessor:
    """Process Star Wars script files and populate database."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.embedding_model = None
        self.tokenizer = None
        self.db_pool = None
        
        # Character name mappings (script names to database names)
        self.character_mappings = {
            'LUKE': 'Luke Skywalker',
            'VADER': 'Darth Vader',
            'YODA': 'Yoda',
            'HAN': 'Han Solo',
            'LEIA': 'Princess Leia',
            'OBI-WAN': 'Obi-Wan Kenobi',
            'THREEPIO': 'C-3PO',
            'ARTOO': 'R2-D2',
            'CHEWBACCA': 'Chewbacca',
            'LANDO': 'Lando Calrissian',
            'EMPEROR': 'Emperor Palpatine',
            'TARKIN': 'Grand Moff Tarkin',
            'BIGGS': 'Biggs Darklighter',
            'OWEN': 'Uncle Owen',
            'BERU': 'Aunt Beru',
            'JABBA': 'Jabba the Hutt',
            'BOBA': 'Boba Fett',
            'PIETT': 'Admiral Piett',
            'VEERS': 'General Veers',
            'JERJERROD': 'Moff Jerjerrod'
        }
        
        # Movie mappings
        self.movie_mappings = {
            'STAR WARS A NEW HOPE.txt': {
                'title': 'Star Wars: Episode IV - A New Hope',
                'year': 1977,
                'description': 'The original Star Wars film'
            },
            'THE EMPIRE STRIKES BACK.txt': {
                'title': 'Star Wars: Episode V - The Empire Strikes Back',
                'year': 1980,
                'description': 'The second Star Wars film'
            },
            'STAR WARS THE RETURN OF THE JEDI.txt': {
                'title': 'Star Wars: Episode VI - Return of the Jedi',
                'year': 1983,
                'description': 'The third Star Wars film'
            }
        }
    
    async def initialize_database(self):
        """Initialize database connection."""
        host = os.getenv("POSTGRES_HOST", "star_wars_postgres")
        port = int(os.getenv("POSTGRES_PORT", "5432"))
        database = os.getenv("POSTGRES_DB", "star_wars_rag")
        user = os.getenv("POSTGRES_USER", "starwars_admin")
        password = os.getenv("POSTGRES_PASSWORD", "your_secure_password_123")
        
        self.db_pool = await asyncpg.create_pool(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        logger.info("✅ Database connection established")
    
    def load_embedding_model(self):
        """Load the embedding model."""
        try:
            model_name = 'sentence-transformers/all-MiniLM-L6-v2'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.embedding_model = AutoModel.from_pretrained(model_name)
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                attention_mask = inputs['attention_mask']
                embeddings = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
                embeddings = embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                return embeddings[0].numpy().tolist()
        except Exception as e:
            logger.error(f"❌ Failed to generate embedding: {e}")
            return None
    
    def extract_dialogue_from_script(self, script_path: Path) -> List[Dict[str, Any]]:
        """Extract dialogue from a script file."""
        dialogue_lines = []
        
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into lines
            lines = content.split('\n')
            
            current_character = None
            current_dialogue = []
            line_number = 0
            
            for line in lines:
                line_number += 1
                line = line.strip()
                
                if not line:
                    continue
                
                # Check if line is a character name (ALL CAPS)
                if line.isupper() and len(line) > 2 and line not in ['INT.', 'EXT.', 'FADE', 'CUT', 'DISSOLVE']:
                    # Save previous dialogue if exists
                    if current_character and current_dialogue:
                        dialogue_text = ' '.join(current_dialogue).strip()
                        if dialogue_text:
                            dialogue_lines.append({
                                'character': current_character,
                                'dialogue': dialogue_text,
                                'line_number': line_number - len(current_dialogue)
                            })
                    
                    # Start new character
                    current_character = line
                    current_dialogue = []
                
                # If we have a character and line is not empty, it's dialogue
                elif current_character and line and not line.isupper():
                    current_dialogue.append(line)
            
            # Save last dialogue
            if current_character and current_dialogue:
                dialogue_text = ' '.join(current_dialogue).strip()
                if dialogue_text:
                    dialogue_lines.append({
                        'character': current_character,
                        'dialogue': dialogue_text,
                        'line_number': line_number - len(current_dialogue)
                    })
            
            logger.info(f"📝 Extracted {len(dialogue_lines)} dialogue lines from {script_path.name}")
            return dialogue_lines
            
        except Exception as e:
            logger.error(f"❌ Failed to extract dialogue from {script_path}: {e}")
            return []
    
    async def populate_database(self):
        """Populate database with real Star Wars data."""
        try:
            # Clear existing data
            async with self.db_pool.acquire() as conn:
                await conn.execute("DELETE FROM dialogue_lines")
                await conn.execute("DELETE FROM characters")
                await conn.execute("DELETE FROM movies")
                logger.info("🗑️ Cleared existing data")
            
            # Add movies
            async with self.db_pool.acquire() as conn:
                for filename, movie_info in self.movie_mappings.items():
                    await conn.execute(
                        "INSERT INTO movies (title, year, description) VALUES ($1, $2, $3)",
                        movie_info['title'], movie_info['year'], movie_info['description']
                    )
                logger.info("✅ Movies added")
            
            # Add characters
            async with self.db_pool.acquire() as conn:
                for script_name, db_name in self.character_mappings.items():
                    await conn.execute(
                        "INSERT INTO characters (name) VALUES ($1) ON CONFLICT (name) DO NOTHING",
                        db_name
                    )
                logger.info("✅ Characters added")
            
            # Process each script file
            script_dir = Path("data/raw")
            total_dialogue_added = 0
            
            for script_file in script_dir.glob("*.txt"):
                if script_file.name not in self.movie_mappings:
                    continue
                
                logger.info(f"🎬 Processing {script_file.name}...")
                
                # Extract dialogue
                dialogue_lines = self.extract_dialogue_from_script(script_file)
                
                # Get movie ID
                async with self.db_pool.acquire() as conn:
                    movie_id = await conn.fetchval(
                        "SELECT id FROM movies WHERE title = $1",
                        self.movie_mappings[script_file.name]['title']
                    )
                    
                    # Process each dialogue line
                    for dialogue_data in dialogue_lines:
                        character_name = dialogue_data['character']
                        
                        # Map character name
                        db_character_name = self.character_mappings.get(character_name, character_name)
                        
                        # Get character ID
                        character_id = await conn.fetchval(
                            "SELECT id FROM characters WHERE name = $1",
                            db_character_name
                        )
                        
                        if not character_id:
                            # Create character if not exists
                            character_id = await conn.fetchval(
                                "INSERT INTO characters (name) VALUES ($1) RETURNING id",
                                db_character_name
                            )
                        
                        # Generate embedding
                        embedding = self.generate_embedding(dialogue_data['dialogue'])
                        
                        if embedding:
                            # Insert dialogue line
                            await conn.execute(
                                """INSERT INTO dialogue_lines 
                                   (movie_id, character_id, dialogue, cleaned_dialogue, line_number, embedding) 
                                   VALUES ($1, $2, $3, $4, $5, $6::vector)""",
                                movie_id, character_id, dialogue_data['dialogue'], 
                                dialogue_data['dialogue'], dialogue_data['line_number'], str(embedding)
                            )
                            total_dialogue_added += 1
                
                logger.info(f"✅ Processed {script_file.name}")
            
            logger.info(f"🎉 Total dialogue lines added: {total_dialogue_added}")
            
            # Verify data
            async with self.db_pool.acquire() as conn:
                movie_count = await conn.fetchval("SELECT COUNT(*) FROM movies")
                character_count = await conn.fetchval("SELECT COUNT(*) FROM characters")
                dialogue_count = await conn.fetchval("SELECT COUNT(*) FROM dialogue_lines")
                
                logger.info(f"📊 Final database contents:")
                logger.info(f"   Movies: {movie_count}")
                logger.info(f"   Characters: {character_count}")
                logger.info(f"   Dialogue lines: {dialogue_count}")
            
        except Exception as e:
            logger.error(f"❌ Failed to populate database: {e}")
            raise

async def main():
    """Main function."""
    logger.info("🌟 Processing Real Star Wars Data")
    logger.info("=" * 50)
    
    processor = StarWarsDataProcessor()
    
    try:
        # Initialize database
        await processor.initialize_database()
        
        # Load embedding model
        processor.load_embedding_model()
        
        # Populate database
        await processor.populate_database()
        
        logger.info("🎉 Real Star Wars data processing completed!")
        logger.info("The RAG system now has authentic dialogue data!")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise
    finally:
        if processor.db_pool:
            await processor.db_pool.close()

if __name__ == "__main__":
    asyncio.run(main())
