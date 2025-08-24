#!/usr/bin/env python3
"""
Simple database initialization script for Star Wars RAG system.
"""

import asyncio
import asyncpg
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection details
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "star_wars_rag",
    "user": "starwars_admin",
    "password": "your_secure_password_123"
}

# SQL for creating tables
CREATE_EXTENSION_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
"""

CREATE_TABLES_SQL = """
-- Movies table
CREATE TABLE IF NOT EXISTS movies (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL UNIQUE,
    year INTEGER,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Characters table  
CREATE TABLE IF NOT EXISTS characters (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Dialogue lines table with vector embeddings
CREATE TABLE IF NOT EXISTS dialogue_lines (
    id SERIAL PRIMARY KEY,
    movie_id INTEGER REFERENCES movies(id),
    character_id INTEGER REFERENCES characters(id),
    dialogue TEXT NOT NULL,
    cleaned_dialogue TEXT,
    scene_info TEXT,
    line_number INTEGER,
    word_count INTEGER,
    embedding vector(384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_dialogue_lines_embedding ON dialogue_lines USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_dialogue_lines_character ON dialogue_lines(character_id);
CREATE INDEX IF NOT EXISTS idx_dialogue_lines_movie ON dialogue_lines(movie_id);
"""

async def init_database():
    """Initialize the database with basic structure."""
    
    logger.info("🚀 Starting database initialization...")
    
    try:
        # Connect to database
        conn = await asyncpg.connect(**DB_CONFIG)
        logger.info("✅ Connected to database")
        
        # Create pgvector extension
        await conn.execute(CREATE_EXTENSION_SQL)
        logger.info("✅ pgvector extension created")
        
        # Create tables
        await conn.execute(CREATE_TABLES_SQL)
        logger.info("✅ Tables created")
        
        # Add some basic movies
        movies = [
            ("Star Wars: Episode IV - A New Hope", 1977, "The original Star Wars film"),
            ("Star Wars: Episode V - The Empire Strikes Back", 1980, "The second Star Wars film"),
            ("Star Wars: Episode VI - Return of the Jedi", 1983, "The third Star Wars film")
        ]
        
        for title, year, description in movies:
            await conn.execute(
                "INSERT INTO movies (title, year, description) VALUES ($1, $2, $3) ON CONFLICT (title) DO NOTHING",
                title, year, description
            )
        logger.info("✅ Movies added")
        
        # Add some basic characters
        characters = [
            "Luke Skywalker",
            "Darth Vader", 
            "Yoda",
            "Han Solo",
            "Princess Leia",
            "Obi-Wan Kenobi",
            "C-3PO",
            "R2-D2",
            "Chewbacca",
            "Lando Calrissian"
        ]
        
        for character in characters:
            await conn.execute(
                "INSERT INTO characters (name) VALUES ($1) ON CONFLICT (name) DO NOTHING",
                character
            )
        logger.info("✅ Characters added")
        
        # Add some sample dialogue lines (without embeddings for now)
        sample_dialogue = [
            ("Star Wars: Episode IV - A New Hope", "Luke Skywalker", "The Force is strong with this one."),
            ("Star Wars: Episode IV - A New Hope", "Darth Vader", "I find your lack of faith disturbing."),
            ("Star Wars: Episode V - The Empire Strikes Back", "Yoda", "Do or do not. There is no try."),
            ("Star Wars: Episode IV - A New Hope", "Han Solo", "May the Force be with you."),
            ("Star Wars: Episode IV - A New Hope", "Princess Leia", "Help me, Obi-Wan Kenobi. You're my only hope.")
        ]
        
        for movie_title, character_name, dialogue in sample_dialogue:
            # Get movie and character IDs
            movie_id = await conn.fetchval("SELECT id FROM movies WHERE title = $1", movie_title)
            character_id = await conn.fetchval("SELECT id FROM characters WHERE name = $1", character_name)
            
            if movie_id and character_id:
                # Create a simple embedding (zeros for now)
                embedding = [0.0] * 384
                
                await conn.execute(
                    """INSERT INTO dialogue_lines 
                       (movie_id, character_id, dialogue, cleaned_dialogue, word_count, embedding) 
                       VALUES ($1, $2, $3, $4, $5, $6)""",
                    movie_id, character_id, dialogue, dialogue, len(dialogue.split()), embedding
                )
        
        logger.info("✅ Sample dialogue added")
        
        # Verify the data
        movie_count = await conn.fetchval("SELECT COUNT(*) FROM movies")
        character_count = await conn.fetchval("SELECT COUNT(*) FROM characters")
        dialogue_count = await conn.fetchval("SELECT COUNT(*) FROM dialogue_lines")
        
        logger.info(f"📊 Database contents:")
        logger.info(f"   Movies: {movie_count}")
        logger.info(f"   Characters: {character_count}")
        logger.info(f"   Dialogue lines: {dialogue_count}")
        
        await conn.close()
        logger.info("🎉 Database initialization completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False

def main():
    """Main function."""
    logger.info("🌟 Simple Star Wars RAG Database Initialization")
    logger.info("=" * 50)
    
    success = asyncio.run(init_database())
    
    if success:
        logger.info("🎉 Database initialization completed successfully!")
        logger.info("The RAG system should now be able to retrieve context!")
    else:
        logger.error("❌ Database initialization failed!")
        exit(1)

if __name__ == "__main__":
    main()
