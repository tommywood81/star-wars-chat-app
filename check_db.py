#!/usr/bin/env python3
"""
Simple script to check database status and contents.
"""

import asyncio
import asyncpg
import logging

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

async def check_database():
    """Check database status and contents."""
    
    try:
        # Connect to database
        conn = await asyncpg.connect(**DB_CONFIG)
        logger.info("✅ Connected to database successfully")
        
        # Check if tables exist
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        
        logger.info(f"📋 Tables found: {[t['table_name'] for t in tables]}")
        
        # Check table contents
        for table in ['movies', 'characters', 'dialogue_lines']:
            try:
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                logger.info(f"📊 {table}: {count} records")
            except Exception as e:
                logger.warning(f"⚠️ Could not query {table}: {e}")
        
        # Check if pgvector extension is installed
        extensions = await conn.fetch("""
            SELECT extname FROM pg_extension WHERE extname = 'vector'
        """)
        
        if extensions:
            logger.info("✅ pgvector extension is installed")
        else:
            logger.warning("⚠️ pgvector extension not found")
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Database check failed: {e}")
        return False

def main():
    """Main function."""
    logger.info("🔍 Checking Star Wars RAG Database Status")
    logger.info("=" * 50)
    
    success = asyncio.run(check_database())
    
    if success:
        logger.info("🎉 Database check completed!")
    else:
        logger.error("❌ Database check failed!")

if __name__ == "__main__":
    main()

