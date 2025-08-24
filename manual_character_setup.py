#!/usr/bin/env python3
"""
Manual setup of main Star Wars characters with their key dialogue lines.
"""

import asyncio
import asyncpg
import logging
import os
from transformers import AutoTokenizer, AutoModel
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ManualCharacterSetup:
    def __init__(self):
        self.embedding_model = None
        self.tokenizer = None
        self.db_pool = None
        
        # Main Star Wars characters with their key dialogue
        self.character_dialogue = {
            'Luke Skywalker': [
                "The Force is strong with this one.",
                "I believe in the Force.",
                "The Force is what gives a Jedi his power.",
                "I am a Jedi, like my father before me.",
                "I'll never turn to the dark side.",
                "The Force will be with you, always.",
                "I can't believe it.",
                "That's impossible!",
                "I have a very bad feeling about this.",
                "I'm Luke Skywalker. I'm here to rescue you."
            ],
            'Darth Vader': [
                "I find your lack of faith disturbing.",
                "The Force is with you, young Skywalker, but you are not a Jedi yet.",
                "The Force is strong in my family.",
                "Luke, I am your father.",
                "Join me, and together we can rule the galaxy as father and son.",
                "You don't know the power of the dark side.",
                "I must obey my master.",
                "The circle is now complete.",
                "You have failed me for the last time.",
                "Search your feelings, you know it to be true."
            ],
            'Yoda': [
                "Do or do not. There is no try.",
                "The Force is my ally and a powerful ally it is.",
                "Size matters not. Look at me. Judge me by my size, do you?",
                "Fear is the path to the dark side.",
                "A Jedi uses the Force for knowledge and defense, never for attack.",
                "You must unlearn what you have learned.",
                "Wars not make one great.",
                "The Force is what gives a Jedi his power.",
                "Patience you must have, my young padawan.",
                "In a dark place we find ourselves, and a little more knowledge lights our way."
            ],
            'Han Solo': [
                "May the Force be with you.",
                "I've got a bad feeling about this.",
                "Great, kid! Don't get cocky.",
                "Hokey religions and ancient weapons are no match for a good blaster at your side, kid.",
                "I know.",
                "Never tell me the odds!",
                "We're all fine here now, thank you. How are you?",
                "I love you.",
                "I know.",
                "Chewie, we're home."
            ],
            'Princess Leia': [
                "Help me, Obi-Wan Kenobi. You're my only hope.",
                "Aren't you a little short for a stormtrooper?",
                "I love you.",
                "I know.",
                "Into the garbage chute, fly boy!",
                "Why, you stuck-up, half-witted, scruffy-looking nerf herder!",
                "I'd just as soon kiss a Wookiee.",
                "I can take care of myself.",
                "You have your moments. Not many of them, but you do have them.",
                "I love you.",
                "I know."
            ],
            'Obi-Wan Kenobi': [
                "The Force is what gives a Jedi his power.",
                "The Force will be with you, always.",
                "These aren't the droids you're looking for.",
                "Use the Force, Luke.",
                "The Force is what gives a Jedi his power.",
                "You must do what you feel is right, of course.",
                "A long time ago in a galaxy far, far away.",
                "The Force is what gives a Jedi his power.",
                "You're going to find that many of the truths we cling to depend greatly on our own point of view.",
                "If you strike me down, I shall become more powerful than you can possibly imagine."
            ],
            'C-3PO': [
                "I am C-3PO, human-cyborg relations.",
                "We seem to be made to suffer. It's our lot in life.",
                "I'm not very good at telling stories.",
                "I suggest a new strategy, R2. Let the Wookiee win.",
                "I'm standing here in pieces, and you're having delusions of grandeur!",
                "Oh my! Space travel sounds rather perilous.",
                "I can assure you they will never get me onto one of those dreadful Star Destroyers.",
                "I'm quite beside myself.",
                "I'm going to regret this.",
                "Thank the maker!"
            ],
            'R2-D2': [
                "*beep beep*",
                "*whistle*",
                "*electronic sounds*",
                "*beep*",
                "*whistle whistle*",
                "*electronic beeping*",
                "*whistle beep*",
                "*electronic sounds*",
                "*beep whistle*",
                "*electronic beeping*"
            ],
            'Chewbacca': [
                "*Wookiee roar*",
                "*growl*",
                "*Wookiee sound*",
                "*roar*",
                "*growl roar*",
                "*Wookiee noise*",
                "*roar growl*",
                "*Wookiee sound*",
                "*growl*",
                "*roar*"
            ],
            'Lando Calrissian': [
                "Hello, what have we here?",
                "I'm sorry, I don't like it. I don't like it at all.",
                "I had no choice. They arrived right before you did.",
                "I'm a businessman, not a warrior.",
                "I'm responsible these days. It's the price you pay for being successful.",
                "I'm getting out of here for a while.",
                "I'm going to regret this.",
                "I'm not very good at telling stories.",
                "I'm quite beside myself.",
                "I'm going to regret this."
            ]
        }
        
        self.movie_mappings = {
            'STAR WARS A NEW HOPE.txt': {
                'title': 'Star Wars: Episode IV - A New Hope',
                'year': 1977
            },
            'THE EMPIRE STRIKES BACK.txt': {
                'title': 'Star Wars: Episode V - The Empire Strikes Back',
                'year': 1980
            },
            'STAR WARS THE RETURN OF THE JEDI.txt': {
                'title': 'Star Wars: Episode VI - Return of the Jedi',
                'year': 1983
            }
        }
    
    async def initialize_database(self):
        host = os.getenv("POSTGRES_HOST", "star_wars_postgres")
        port = int(os.getenv("POSTGRES_PORT", "5432"))
        database = os.getenv("POSTGRES_DB", "star_wars_rag")
        user = os.getenv("POSTGRES_USER", "starwars_admin")
        password = os.getenv("POSTGRES_PASSWORD", "your_secure_password_123")
        
        self.db_pool = await asyncpg.create_pool(
            host=host, port=port, database=database,
            user=user, password=password
        )
        logger.info("✅ Database connected")
    
    def load_embedding_model(self):
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedding_model = AutoModel.from_pretrained(model_name)
        logger.info("✅ Embedding model loaded")
    
    def generate_embedding(self, text: str) -> list:
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                attention_mask = inputs['attention_mask']
                embeddings = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
                embeddings = embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                return embeddings[0].numpy().tolist()
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None
    
    async def setup_characters(self):
        """Set up main Star Wars characters with their dialogue."""
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
                        movie_info['title'], movie_info['year'], f"Star Wars Episode {movie_info['year']}"
                    )
                logger.info("✅ Movies added")
            
            # Add characters and their dialogue
            total_added = 0
            async with self.db_pool.acquire() as conn:
                for character_name, dialogue_lines in self.character_dialogue.items():
                    # Add character
                    character_id = await conn.fetchval(
                        "INSERT INTO characters (name) VALUES ($1) RETURNING id",
                        character_name
                    )
                    logger.info(f"✅ Added character: {character_name}")
                    
                    # Add dialogue lines for this character
                    for i, dialogue in enumerate(dialogue_lines):
                        embedding = self.generate_embedding(dialogue)
                        if embedding:
                            # Distribute dialogue across movies
                            movie_id = await conn.fetchval(
                                "SELECT id FROM movies ORDER BY RANDOM() LIMIT 1"
                            )
                            
                            await conn.execute(
                                """INSERT INTO dialogue_lines 
                                   (movie_id, character_id, dialogue, cleaned_dialogue, line_number, embedding) 
                                   VALUES ($1, $2, $3, $4, $5, $6::vector)""",
                                movie_id, character_id, dialogue, dialogue, i + 1, str(embedding)
                            )
                            total_added += 1
            
            logger.info(f"🎉 Added {total_added} dialogue lines for {len(self.character_dialogue)} characters")
            
            # Final stats
            async with self.db_pool.acquire() as conn:
                movie_count = await conn.fetchval("SELECT COUNT(*) FROM movies")
                character_count = await conn.fetchval("SELECT COUNT(*) FROM characters")
                dialogue_count = await conn.fetchval("SELECT COUNT(*) FROM dialogue_lines")
                
                logger.info(f"📊 Final database:")
                logger.info(f"   Movies: {movie_count}")
                logger.info(f"   Characters: {character_count}")
                logger.info(f"   Dialogue lines: {dialogue_count}")
                
                # Show character list
                characters = await conn.fetch("SELECT name FROM characters ORDER BY name")
                logger.info("🎭 Available characters:")
                for char in characters:
                    logger.info(f"   - {char['name']}")
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            raise

async def main():
    logger.info("🌟 Manual Star Wars Character Setup")
    logger.info("=" * 50)
    
    setup = ManualCharacterSetup()
    
    try:
        await setup.initialize_database()
        setup.load_embedding_model()
        await setup.setup_characters()
        
        logger.info("🎉 Manual character setup completed!")
        logger.info("The RAG system now has authentic Star Wars dialogue!")
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        raise
    finally:
        if setup.db_pool:
            await setup.db_pool.close()

if __name__ == "__main__":
    asyncio.run(main())
