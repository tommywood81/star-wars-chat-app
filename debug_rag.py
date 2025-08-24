#!/usr/bin/env python3
"""
Debug script to test RAG functionality step by step.
"""

import asyncio
import asyncpg
import logging
import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_rag():
    """Debug the RAG functionality step by step."""
    
    # Database connection details
    host = os.getenv("POSTGRES_HOST", "star_wars_postgres")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    database = os.getenv("POSTGRES_DB", "star_wars_rag")
    user = os.getenv("POSTGRES_USER", "starwars_admin")
    password = os.getenv("POSTGRES_PASSWORD", "your_secure_password_123")
    
    try:
        # Step 1: Test database connection
        logger.info("🔍 Step 1: Testing database connection...")
        conn = await asyncpg.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        logger.info("✅ Database connection successful")
        
        # Step 2: Check if data exists
        logger.info("🔍 Step 2: Checking database contents...")
        movie_count = await conn.fetchval("SELECT COUNT(*) FROM movies")
        character_count = await conn.fetchval("SELECT COUNT(*) FROM characters")
        dialogue_count = await conn.fetchval("SELECT COUNT(*) FROM dialogue_lines")
        
        logger.info(f"📊 Movies: {movie_count}, Characters: {character_count}, Dialogue: {dialogue_count}")
        
        # Step 3: Check if Luke Skywalker exists
        logger.info("🔍 Step 3: Checking if Luke Skywalker exists...")
        luke_id = await conn.fetchval("SELECT id FROM characters WHERE name = $1", "Luke Skywalker")
        logger.info(f"Luke Skywalker ID: {luke_id}")
        
        if luke_id:
            # Step 4: Check Luke's dialogue lines
            logger.info("🔍 Step 4: Checking Luke's dialogue lines...")
            luke_dialogue = await conn.fetch(
                "SELECT dialogue, embedding FROM dialogue_lines WHERE character_id = $1 LIMIT 3",
                luke_id
            )
            logger.info(f"Found {len(luke_dialogue)} dialogue lines for Luke")
            for i, row in enumerate(luke_dialogue):
                logger.info(f"  {i+1}. {row['dialogue']}")
                logger.info(f"     Embedding: {row['embedding'][:5]}..." if row['embedding'] else "     No embedding")
        
        # Step 5: Test embedding model
        logger.info("🔍 Step 5: Testing embedding model...")
        try:
            model_name = 'sentence-transformers/all-MiniLM-L6-v2'
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            embedding_model = AutoModel.from_pretrained(model_name)
            logger.info("✅ Embedding model loaded successfully")
            
            # Test embedding generation
            message = "What is the Force?"
            inputs = tokenizer(message, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = embedding_model(**inputs)
                attention_mask = inputs['attention_mask']
                embeddings = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
                embeddings = embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                message_embedding = embeddings[0].numpy()
            
            logger.info(f"✅ Generated embedding: {message_embedding[:5]}...")
            
            # Step 6: Test similarity search
            if luke_id and luke_dialogue:
                logger.info("🔍 Step 6: Testing similarity search...")
                query = """
                SELECT dl.dialogue, dl.embedding, m.title as movie_title
                FROM dialogue_lines dl
                JOIN movies m ON dl.movie_id = m.id
                WHERE dl.character_id = $1
                ORDER BY dl.embedding <=> $2
                LIMIT 3
                """
                
                rows = await conn.fetch(query, luke_id, message_embedding.tolist())
                logger.info(f"✅ Similarity search returned {len(rows)} results")
                
                for i, row in enumerate(rows):
                    logger.info(f"  {i+1}. {row['dialogue']} (from {row['movie_title']})")
                    if row['embedding']:
                        logger.info(f"     Has embedding: {len(row['embedding'])} dimensions")
                    else:
                        logger.info(f"     No embedding!")
                        
        except Exception as e:
            logger.error(f"❌ Embedding model error: {e}")
        
        await conn.close()
        
    except Exception as e:
        logger.error(f"❌ Debug failed: {e}")

def main():
    """Main function."""
    logger.info("🔧 Debugging RAG System")
    logger.info("=" * 50)
    
    asyncio.run(debug_rag())

if __name__ == "__main__":
    main()
