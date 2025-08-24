#!/usr/bin/env python3
"""
Script to populate the database with sample Star Wars dialogue data.
This script should be run inside the LLM service container.
"""

import asyncio
import asyncpg
import logging
import os
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def populate_database():
    """Populate the database with sample Star Wars dialogue data."""
    
    # Database connection details (from environment variables in container)
    host = os.getenv("POSTGRES_HOST", "star_wars_postgres")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    database = os.getenv("POSTGRES_DB", "star_wars_rag")
    user = os.getenv("POSTGRES_USER", "starwars_admin")
    password = os.getenv("POSTGRES_PASSWORD", "your_secure_password_123")
    
    try:
        # Connect to database
        conn = await asyncpg.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        logger.info("✅ Connected to database")
        
        # Create pgvector extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        logger.info("✅ pgvector extension created")
        
        # Create tables
        await conn.execute("""
            -- Movies table
            CREATE TABLE IF NOT EXISTS movies (
                id SERIAL PRIMARY KEY,
                title TEXT NOT NULL UNIQUE,
                year INTEGER,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        await conn.execute("""
            -- Characters table  
            CREATE TABLE IF NOT EXISTS characters (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        await conn.execute("""
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
        """)
        
        # Create indexes
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_dialogue_lines_embedding 
            ON dialogue_lines USING ivfflat (embedding vector_cosine_ops);
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_dialogue_lines_character 
            ON dialogue_lines(character_id);
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_dialogue_lines_movie 
            ON dialogue_lines(movie_id);
        """)
        
        logger.info("✅ Tables and indexes created")
        
        # Add movies
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
        
        # Add characters
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
        
        # Add sample dialogue lines with Force-related content
        sample_dialogue = [
            ("Star Wars: Episode IV - A New Hope", "Luke Skywalker", "The Force is strong with this one."),
            ("Star Wars: Episode IV - A New Hope", "Darth Vader", "I find your lack of faith disturbing."),
            ("Star Wars: Episode V - The Empire Strikes Back", "Yoda", "Do or do not. There is no try."),
            ("Star Wars: Episode IV - A New Hope", "Han Solo", "May the Force be with you."),
            ("Star Wars: Episode IV - A New Hope", "Princess Leia", "Help me, Obi-Wan Kenobi. You're my only hope."),
            ("Star Wars: Episode IV - A New Hope", "Obi-Wan Kenobi", "The Force is what gives a Jedi his power."),
            ("Star Wars: Episode V - The Empire Strikes Back", "Yoda", "The Force is my ally and a powerful ally it is."),
            ("Star Wars: Episode IV - A New Hope", "Luke Skywalker", "I believe in the Force."),
            ("Star Wars: Episode V - The Empire Strikes Back", "Darth Vader", "The Force is with you, young Skywalker, but you are not a Jedi yet."),
            ("Star Wars: Episode IV - A New Hope", "Obi-Wan Kenobi", "The Force will be with you, always."),
            ("Star Wars: Episode V - The Empire Strikes Back", "Yoda", "Size matters not. Look at me. Judge me by my size, do you?"),
            ("Star Wars: Episode IV - A New Hope", "Luke Skywalker", "The Force is what gives a Jedi his power."),
            ("Star Wars: Episode V - The Empire Strikes Back", "Yoda", "Fear is the path to the dark side."),
            ("Star Wars: Episode IV - A New Hope", "Darth Vader", "The Force is strong in my family."),
            ("Star Wars: Episode V - The Empire Strikes Back", "Yoda", "A Jedi uses the Force for knowledge and defense, never for attack.")
        ]
        
        for movie_title, character_name, dialogue in sample_dialogue:
            # Get movie and character IDs
            movie_id = await conn.fetchval("SELECT id FROM movies WHERE title = $1", movie_title)
            character_id = await conn.fetchval("SELECT id FROM characters WHERE name = $1", character_name)
            
            if movie_id and character_id:
                # Create a simple embedding (random values for now - in production this would be generated by the embedding model)
                embedding = np.random.rand(384).tolist()
                
                await conn.execute(
                    """INSERT INTO dialogue_lines 
                       (movie_id, character_id, dialogue, cleaned_dialogue, word_count, embedding) 
                       VALUES ($1, $2, $3, $4, $5, $6::vector) ON CONFLICT DO NOTHING""",
                    movie_id, character_id, dialogue, dialogue, len(dialogue.split()), str(embedding)
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
        logger.info("🎉 Database population completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database population failed: {e}")
        return False

def main():
    """Main function."""
    logger.info("🌟 Populating Star Wars RAG Database")
    logger.info("=" * 50)
    
    success = asyncio.run(populate_database())
    
    if success:
        logger.info("🎉 Database population completed successfully!")
        logger.info("The RAG system should now be able to retrieve context!")
    else:
        logger.error("❌ Database population failed!")
        exit(1)

if __name__ == "__main__":
    main()

