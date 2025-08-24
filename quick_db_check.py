#!/usr/bin/env python3
import asyncio
import asyncpg
import os

async def check_db():
    try:
        conn = await asyncpg.connect(
            host='star_wars_postgres',
            database='star_wars_rag',
            user='starwars_admin',
            password='your_secure_password_123'
        )
        
        movie_count = await conn.fetchval('SELECT COUNT(*) FROM movies')
        char_count = await conn.fetchval('SELECT COUNT(*) FROM characters')
        dialogue_count = await conn.fetchval('SELECT COUNT(*) FROM dialogue_lines')
        
        print(f"📊 Database Status:")
        print(f"   Movies: {movie_count}")
        print(f"   Characters: {char_count}")
        print(f"   Dialogue lines: {dialogue_count}")
        
        chars = await conn.fetch('SELECT name FROM characters ORDER BY name')
        print(f"\n🎭 Characters in database:")
        for char in chars:
            print(f"   - {char['name']}")
        
        await conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_db())

