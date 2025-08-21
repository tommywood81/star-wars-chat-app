"""
PostgreSQL + pgvector database integration for Star Wars RAG system.

This module provides database functionality for storing and retrieving
dialogue embeddings with vector similarity search capabilities.
"""

import logging
import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

import asyncpg
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

# SQL queries for database setup and operations
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
    embedding vector(384),  -- Assuming 384-dim embeddings from all-MiniLM-L6-v2
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chat sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    character_id INTEGER REFERENCES characters(id),
    role TEXT NOT NULL CHECK (role IN ('user', 'character')),
    content TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_dialogue_lines_embedding ON dialogue_lines USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_dialogue_lines_character ON dialogue_lines(character_id);
CREATE INDEX IF NOT EXISTS idx_dialogue_lines_movie ON dialogue_lines(movie_id);
CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at);
"""

DROP_TABLES_SQL = """
DROP TABLE IF EXISTS sessions CASCADE;
DROP TABLE IF EXISTS dialogue_lines CASCADE;
DROP TABLE IF EXISTS characters CASCADE;
DROP TABLE IF EXISTS movies CASCADE;
"""


class DatabaseManager:
    """PostgreSQL + pgvector database manager for Star Wars RAG system."""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 5432,
                 database: str = "star_wars_rag",
                 user: str = "postgres",
                 password: str = "password",
                 embedding_dimension: int = 384):
        """Initialize database manager.
        
        Args:
            host: Database host
            port: Database port  
            database: Database name
            user: Database user
            password: Database password
            embedding_dimension: Dimension of embedding vectors
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.embedding_dimension = embedding_dimension
        
        self.connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        self.pool: Optional[asyncpg.Pool] = None
        
        # Cache for IDs
        self._character_id_cache: Dict[str, int] = {}
        self._movie_id_cache: Dict[str, int] = {}
    
    async def initialize(self):
        """Initialize database connection and setup."""
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            
            logger.info("Database connection pool created")
            
            # Test connection
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
            
            logger.info("Database connection verified")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise RuntimeError(f"Database initialization failed: {e}")
    
    async def create_tables(self):
        """Create required database tables and indexes."""
        try:
            async with self.pool.acquire() as conn:
                # Create pgvector extension
                await conn.execute(CREATE_EXTENSION_SQL)
                logger.info("pgvector extension created/verified")
                
                # Create tables
                await conn.execute(CREATE_TABLES_SQL)
                logger.info("Database tables created/verified")
                
        except Exception as e:
            logger.error(f"Table creation failed: {e}")
            raise RuntimeError(f"Table creation failed: {e}")
    
    async def test_connection(self) -> bool:
        """Test database connection."""
        try:
            if not self.pool:
                return False
            
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
                
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    async def has_pgvector(self) -> bool:
        """Check if pgvector extension is available."""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                )
                return bool(result)
                
        except Exception as e:
            logger.error(f"pgvector check failed: {e}")
            return False
    
    async def list_tables(self) -> List[str]:
        """List all tables in the database."""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
                )
                return [row['tablename'] for row in rows]
                
        except Exception as e:
            logger.error(f"List tables failed: {e}")
            return []
    
    async def _get_or_create_character_id(self, character_name: str) -> int:
        """Get or create character ID."""
        if character_name in self._character_id_cache:
            return self._character_id_cache[character_name]
        
        async with self.pool.acquire() as conn:
            # Try to get existing character
            char_id = await conn.fetchval(
                "SELECT id FROM characters WHERE name = $1",
                character_name
            )
            
            if char_id is None:
                # Create new character
                char_id = await conn.fetchval(
                    "INSERT INTO characters (name) VALUES ($1) RETURNING id",
                    character_name
                )
            
            self._character_id_cache[character_name] = char_id
            return char_id
    
    async def _get_or_create_movie_id(self, movie_title: str) -> int:
        """Get or create movie ID."""
        if movie_title in self._movie_id_cache:
            return self._movie_id_cache[movie_title]
        
        async with self.pool.acquire() as conn:
            # Try to get existing movie
            movie_id = await conn.fetchval(
                "SELECT id FROM movies WHERE title = $1",
                movie_title
            )
            
            if movie_id is None:
                # Create new movie
                movie_id = await conn.fetchval(
                    "INSERT INTO movies (title) VALUES ($1) RETURNING id",
                    movie_title
                )
            
            self._movie_id_cache[movie_title] = movie_id
            return movie_id
    
    async def store_dialogue_data(self, 
                                 dialogue_df: pd.DataFrame, 
                                 embeddings: np.ndarray) -> int:
        """Store dialogue data with embeddings in the database.
        
        Args:
            dialogue_df: DataFrame with dialogue data
            embeddings: Corresponding embeddings array
            
        Returns:
            Number of records stored
        """
        if len(dialogue_df) != len(embeddings):
            raise ValueError("DataFrame and embeddings length mismatch")
        
        stored_count = 0
        
        try:
            async with self.pool.acquire() as conn:
                # Start transaction
                async with conn.transaction():
                    for idx, row in dialogue_df.iterrows():
                        # Get or create character and movie IDs
                        character_id = await self._get_or_create_character_id(row['character'])
                        movie_id = await self._get_or_create_movie_id(row['movie'])
                        
                        # Convert embedding to list for storage
                        embedding_list = embeddings[idx].tolist()
                        
                        # Insert dialogue line
                        await conn.execute("""
                            INSERT INTO dialogue_lines 
                            (movie_id, character_id, dialogue, cleaned_dialogue, 
                             word_count, embedding)
                            VALUES ($1, $2, $3, $4, $5, $6)
                        """, 
                        movie_id,
                        character_id,
                        row['dialogue'],
                        row.get('cleaned_dialogue', row['dialogue']),
                        len(row['dialogue'].split()),
                        embedding_list
                        )
                        
                        stored_count += 1
            
            logger.info(f"Stored {stored_count} dialogue lines in database")
            return stored_count
            
        except Exception as e:
            logger.error(f"Failed to store dialogue data: {e}")
            raise RuntimeError(f"Failed to store dialogue data: {e}")
    
    async def similarity_search(self,
                               query_embedding: np.ndarray,
                               limit: int = 10,
                               character_filter: Optional[str] = None,
                               movie_filter: Optional[str] = None,
                               min_similarity: float = 0.0) -> List[Dict[str, Any]]:
        """Perform vector similarity search.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            character_filter: Filter by character name
            movie_filter: Filter by movie title
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of search results with similarity scores
        """
        try:
            # Convert query embedding to list
            query_vector = query_embedding.tolist()
            
            # Build query
            base_query = """
                SELECT 
                    dl.id,
                    c.name as character,
                    m.title as movie,
                    dl.dialogue,
                    dl.cleaned_dialogue,
                    1 - (dl.embedding <=> $1) as similarity
                FROM dialogue_lines dl
                JOIN characters c ON dl.character_id = c.id
                JOIN movies m ON dl.movie_id = m.id
                WHERE 1=1
            """
            
            params = [query_vector]
            param_count = 1
            
            # Add filters
            if character_filter:
                param_count += 1
                base_query += f" AND c.name = ${param_count}"
                params.append(character_filter)
            
            if movie_filter:
                param_count += 1
                base_query += f" AND m.title = ${param_count}"
                params.append(movie_filter)
            
            if min_similarity > 0:
                param_count += 1
                base_query += f" AND (1 - (dl.embedding <=> $1)) >= ${param_count}"
                params.append(min_similarity)
            
            # Add ordering and limit
            base_query += f" ORDER BY dl.embedding <=> $1 LIMIT {limit}"
            
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(base_query, *params)
                
                results = []
                for row in rows:
                    results.append({
                        "id": row['id'],
                        "character": row['character'],
                        "movie": row['movie'],
                        "dialogue": row['dialogue'],
                        "cleaned_dialogue": row['cleaned_dialogue'],
                        "similarity": float(row['similarity'])
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise RuntimeError(f"Similarity search failed: {e}")
    
    async def get_characters(self) -> List[str]:
        """Get list of all characters."""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("SELECT name FROM characters ORDER BY name")
                return [row['name'] for row in rows]
                
        except Exception as e:
            logger.error(f"Get characters failed: {e}")
            return []
    
    async def get_movies(self) -> List[str]:
        """Get list of all movies."""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("SELECT title FROM movies ORDER BY title")
                return [row['title'] for row in rows]
                
        except Exception as e:
            logger.error(f"Get movies failed: {e}")
            return []
    
    async def count_dialogue_lines(self) -> int:
        """Count total dialogue lines."""
        try:
            async with self.pool.acquire() as conn:
                count = await conn.fetchval("SELECT COUNT(*) FROM dialogue_lines")
                return int(count) if count else 0
                
        except Exception as e:
            logger.error(f"Count dialogue lines failed: {e}")
            return 0
    
    async def get_character_statistics(self) -> List[Dict[str, Any]]:
        """Get dialogue statistics by character."""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT 
                        c.name as character,
                        COUNT(dl.id) as dialogue_count,
                        SUM(dl.word_count) as total_words,
                        AVG(dl.word_count) as avg_words_per_line
                    FROM characters c
                    LEFT JOIN dialogue_lines dl ON c.id = dl.character_id
                    GROUP BY c.id, c.name
                    ORDER BY dialogue_count DESC
                """)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Get character statistics failed: {e}")
            return []
    
    async def store_session_message(self,
                                   session_id: str,
                                   character: str,
                                   role: str,
                                   content: str,
                                   metadata: Optional[Dict[str, Any]] = None):
        """Store a chat session message."""
        try:
            character_id = await self._get_or_create_character_id(character)
            
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO sessions 
                    (session_id, character_id, role, content, metadata)
                    VALUES ($1, $2, $3, $4, $5)
                """,
                session_id,
                character_id,
                role,
                content,
                metadata or {}
                )
                
        except Exception as e:
            logger.error(f"Store session message failed: {e}")
            raise RuntimeError(f"Store session message failed: {e}")
    
    async def get_session_history(self,
                                 session_id: str,
                                 limit: int = 20) -> List[Dict[str, Any]]:
        """Get chat session history."""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT 
                        s.role,
                        s.content,
                        c.name as character,
                        s.metadata,
                        s.created_at
                    FROM sessions s
                    JOIN characters c ON s.character_id = c.id
                    WHERE s.session_id = $1
                    ORDER BY s.created_at DESC
                    LIMIT $2
                """,
                session_id,
                limit
                )
                
                # Reverse to get chronological order
                return [dict(row) for row in reversed(rows)]
                
        except Exception as e:
            logger.error(f"Get session history failed: {e}")
            return []
    
    async def get_embedding_dimension(self) -> int:
        """Get the embedding dimension from stored vectors."""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("""
                    SELECT array_length(embedding, 1) 
                    FROM dialogue_lines 
                    WHERE embedding IS NOT NULL 
                    LIMIT 1
                """)
                return int(result) if result else self.embedding_dimension
                
        except Exception as e:
            logger.error(f"Get embedding dimension failed: {e}")
            return self.embedding_dimension
    
    async def cleanup_test_data(self):
        """Clean up test data (for testing purposes)."""
        try:
            async with self.pool.acquire() as conn:
                # Only clean up if this is a test database
                if "test" in self.database.lower():
                    await conn.execute("DELETE FROM sessions")
                    await conn.execute("DELETE FROM dialogue_lines")
                    await conn.execute("DELETE FROM characters")
                    await conn.execute("DELETE FROM movies")
                    logger.info("Test data cleaned up")
                else:
                    logger.warning("Cleanup skipped - not a test database")
                    
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    async def close(self):
        """Close database connections."""
        if self.pool:
            await self.pool.close()
            logger.info("Database connections closed")


# Utility functions for non-async usage
def create_database_sync(database_config: Dict[str, Any]) -> bool:
    """Create database synchronously (for setup scripts)."""
    try:
        # Connect to postgres database to create our database
        conn_params = database_config.copy()
        target_db = conn_params.pop('database')
        conn_params['database'] = 'postgres'
        
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = True
        
        with conn.cursor() as cursor:
            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (target_db,)
            )
            
            if not cursor.fetchone():
                # Create database
                cursor.execute(f'CREATE DATABASE "{target_db}"')
                logger.info(f"Database '{target_db}' created")
            else:
                logger.info(f"Database '{target_db}' already exists")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Database creation failed: {e}")
        return False


async def setup_database(database_config: Dict[str, Any]) -> DatabaseManager:
    """Setup database with tables and extensions."""
    db = DatabaseManager(**database_config)
    
    await db.initialize()
    await db.create_tables()
    
    return db
