#!/usr/bin/env python3
"""
Database initialization script for Star Wars RAG system.

This script:
1. Creates the database tables
2. Processes the raw Star Wars script files
3. Loads dialogue data into the database
4. Generates embeddings for vector search
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from star_wars_rag.database import DatabaseManager
from star_wars_rag.data_processor import DialogueProcessor
from star_wars_rag.embeddings import StarWarsEmbedder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def init_database():
    """Initialize the database with Star Wars dialogue data."""
    
    logger.info("🚀 Starting database initialization...")
    
    # Initialize database manager
    db_manager = DatabaseManager(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "star_wars_rag"),
        user=os.getenv("POSTGRES_USER", "starwars_admin"),
        password=os.getenv("POSTGRES_PASSWORD", "your_secure_password_123")
    )
    
    try:
        # Step 1: Initialize database connection
        logger.info("🔌 Initializing database connection...")
        await db_manager.initialize()
        logger.info("✅ Database connection established")
        
        # Step 2: Create tables
        logger.info("📋 Creating database tables...")
        await db_manager.create_tables()
        logger.info("✅ Tables created successfully")
        
        # Step 2: Process raw script files
        logger.info("📝 Processing Star Wars script files...")
        processor = DialogueProcessor()
        
        # Define movie mappings
        movies = {
            "STAR WARS A NEW HOPE.txt": {
                "title": "Star Wars: Episode IV - A New Hope",
                "year": 1977,
                "description": "The original Star Wars film"
            },
            "THE EMPIRE STRIKES BACK.txt": {
                "title": "Star Wars: Episode V - The Empire Strikes Back", 
                "year": 1980,
                "description": "The second Star Wars film"
            },
            "STAR WARS THE RETURN OF THE JEDI.txt": {
                "title": "Star Wars: Episode VI - Return of the Jedi",
                "year": 1983,
                "description": "The third Star Wars film"
            }
        }
        
        # Process each movie
        all_dialogue = []
        for filename, movie_info in movies.items():
            script_path = Path("data/raw") / filename
            if script_path.exists():
                logger.info(f"Processing {filename}...")
                
                # Extract dialogue
                dialogue_data = processor.extract_dialogue(script_path)
                
                # Add movie information
                for entry in dialogue_data:
                    entry['movie_title'] = movie_info['title']
                    entry['movie_year'] = movie_info['year']
                
                all_dialogue.extend(dialogue_data)
                logger.info(f"✅ Extracted {len(dialogue_data)} dialogue lines from {filename}")
            else:
                logger.warning(f"⚠️ Script file not found: {script_path}")
        
        if not all_dialogue:
            logger.error("❌ No dialogue data extracted!")
            return False
        
        logger.info(f"✅ Total dialogue lines extracted: {len(all_dialogue)}")
        
        # Step 3: Load movies and characters into database
        logger.info("🎬 Loading movies and characters...")
        
        # Get unique movies and characters
        unique_movies = list(set(entry['movie_title'] for entry in all_dialogue))
        unique_characters = list(set(entry['character'] for entry in all_dialogue))
        
        # Load movies
        for movie_title in unique_movies:
            movie_info = next(m for m in movies.values() if m['title'] == movie_title)
            await db_manager.add_movie(
                title=movie_title,
                year=movie_info['year'],
                description=movie_info['description']
            )
        
        # Load characters
        for character in unique_characters:
            await db_manager.add_character(name=character)
        
        logger.info(f"✅ Loaded {len(unique_movies)} movies and {len(unique_characters)} characters")
        
        # Step 4: Generate embeddings and load dialogue
        logger.info("🧠 Generating embeddings for dialogue...")
        embedder = StarWarsEmbedder()
        
        # Process dialogue in batches
        batch_size = 100
        total_loaded = 0
        
        for i in range(0, len(all_dialogue), batch_size):
            batch = all_dialogue[i:i + batch_size]
            
            # Generate embeddings for this batch
            texts = [entry['dialogue'] for entry in batch]
            embeddings = embedder.embed_batch(texts)
            
            # Load batch into database
            for j, entry in enumerate(batch):
                await db_manager.add_dialogue_line(
                    movie_title=entry['movie_title'],
                    character_name=entry['character'],
                    dialogue=entry['dialogue'],
                    cleaned_dialogue=entry.get('dialogue_clean', entry['dialogue']),
                    scene_info=entry.get('scene_info', ''),
                    line_number=entry.get('line_number', 0),
                    word_count=entry.get('word_count', len(entry['dialogue'].split())),
                    embedding=embeddings[j].tolist()
                )
            
            total_loaded += len(batch)
            logger.info(f"✅ Loaded batch {i//batch_size + 1}: {len(batch)} lines (Total: {total_loaded})")
        
        logger.info(f"🎉 Database initialization complete! Loaded {total_loaded} dialogue lines")
        
        # Step 5: Verify the data
        logger.info("🔍 Verifying database contents...")
        
        movie_count = await db_manager.get_movie_count()
        character_count = await db_manager.get_character_count()
        dialogue_count = await db_manager.get_dialogue_count()
        
        logger.info(f"📊 Database contents:")
        logger.info(f"   Movies: {movie_count}")
        logger.info(f"   Characters: {character_count}")
        logger.info(f"   Dialogue lines: {dialogue_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False

def main():
    """Main function."""
    logger.info("🌟 Star Wars RAG Database Initialization")
    logger.info("=" * 50)
    
    success = asyncio.run(init_database())
    
    if success:
        logger.info("🎉 Database initialization completed successfully!")
        logger.info("The RAG system is now ready to use.")
    else:
        logger.error("❌ Database initialization failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
