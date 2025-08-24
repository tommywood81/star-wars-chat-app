#!/usr/bin/env python3
"""
Fixed dialogue extraction that properly identifies character names vs scene descriptions.
"""

import asyncio
import asyncpg
import logging
import os
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel
import torch
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedDialogueProcessor:
    def __init__(self):
        self.embedding_model = None
        self.tokenizer = None
        self.db_pool = None
        
        # Scene/location keywords to filter out
        self.scene_keywords = {
            'INT.', 'EXT.', 'FADE', 'CUT', 'DISSOLVE', 'CONTINUED', 'DAY', 'NIGHT',
            'COCKPIT', 'ROOM', 'HALLWAY', 'CORRIDOR', 'PASSAGEWAY', 'BRIDGE',
            'BATTLEFIELD', 'SPACE', 'PLANET', 'STAR', 'FALCON', 'FIGHTER',
            'DEATH', 'STAR', 'HOTH', 'TATOOINE', 'DAGOBAH', 'BESPIN', 'ENDOR',
            'GENERATORS', 'GUN', 'TOWERS', 'OUTPOST', 'WAR', 'LEADER', 'THREE',
            'MILLENNIUM', 'X-WING', 'SNOWSPEEDER', 'ROGUE', 'AROUND', 'THE'
        }
        
        # Real character names (these are the ones we want)
        self.character_names = {
            'LUKE', 'VADER', 'YODA', 'HAN', 'LEIA', 'OBI-WAN', 'THREEPIO', 'ARTOO',
            'CHEWBACCA', 'LANDO', 'EMPEROR', 'TARKIN', 'PIETT', 'VEERS', 'JERJERROD',
            'TROOPER', 'IMPERIAL', 'BIGGS', 'WEDGE', 'ACKBAR', 'MON', 'OWEN', 'BERU',
            'JABBA', 'BOBA', 'FIXER', 'CAMIE', 'DEAK', 'WINDY', 'WOMAN', 'MAN',
            'BOY', 'GIRL', 'CHILD', 'CHILDREN', 'SOLDIER', 'OFFICER', 'PILOT',
            'CAPTAIN', 'COMMANDER', 'GENERAL', 'ADMIRAL', 'GUARD', 'CREW', 'DROID',
            'ROBOT', 'VOICE', 'ANNOUNCER', 'COMPUTER', 'INTERCOM', 'REBEL', 'CHIEF',
            'PRINCESS', 'DARK', 'LORD', 'SITH', 'JEDI', 'MASTER', 'APPRENTICE',
            'SKYWALKER', 'SOLO', 'ORGANA', 'KENOBI', 'CALRISSIAN', 'PALPATINE',
            'FETT', 'HUTT', 'ANTILLES', 'MOTHMA', 'ACKBAR', 'PIETT', 'VEERS',
            'JERJERROD', 'TARKIN', 'DARTH', 'ANAKIN', 'PADME', 'AMIDALA', 'QUI-GON',
            'JINN', 'MACE', 'WINDU', 'KIT', 'FISTO', 'PLO', 'KOON', 'KI-ADI-MUNDI',
            'SHAK', 'TI', 'AYLA', 'SECURA', 'BARISS', 'OFFEE', 'LUMINARA', 'UNDULI',
            'DEPA', 'BILLABA', 'COLEMAN', 'TREBOR', 'EVEN', 'PIELL', 'OPPO', 'RANCISIS',
            'SAESEE', 'TIIN', 'ADDI', 'GALLIA', 'STASS', 'ALLIE', 'AGEN', 'KOLAR',
            'EETH', 'KOTH', 'YAREL', 'POOF', 'MAS', 'AMEDDA', 'SIO', 'BIBBLE',
            'RIC', 'OLIE', 'BRAVO', 'FIVE', 'BRAVO', 'SIX', 'BRAVO', 'SEVEN',
            'BRAVO', 'EIGHT', 'BRAVO', 'NINE', 'BRAVO', 'TEN', 'BRAVO', 'ELEVEN',
            'BRAVO', 'TWELVE', 'GOLD', 'FIVE', 'GOLD', 'SIX', 'GOLD', 'SEVEN',
            'GOLD', 'EIGHT', 'GOLD', 'NINE', 'GOLD', 'TEN', 'GOLD', 'ELEVEN',
            'GOLD', 'TWELVE', 'RED', 'FIVE', 'RED', 'SIX', 'RED', 'SEVEN',
            'RED', 'EIGHT', 'RED', 'NINE', 'RED', 'TEN', 'RED', 'ELEVEN', 'RED', 'TWELVE'
        }
        
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
            'PIETT': 'Admiral Piett',
            'VEERS': 'General Veers',
            'JERJERROD': 'Moff Jerjerrod',
            'TROOPER': 'Stormtrooper',
            'IMPERIAL': 'Imperial Officer',
            'BIGGS': 'Biggs Darklighter',
            'WEDGE': 'Wedge Antilles',
            'ACKBAR': 'Admiral Ackbar',
            'MON': 'Mon Mothma',
            'OWEN': 'Uncle Owen',
            'BERU': 'Aunt Beru',
            'JABBA': 'Jabba the Hutt',
            'BOBA': 'Boba Fett',
            'FIXER': 'The Fixer',
            'CAMIE': 'Camie',
            'DEAK': 'Deak',
            'WINDY': 'Windy',
            'REBEL': 'Rebel Soldier',
            'DARTH': 'Darth Vader',
            'PRINCESS': 'Princess Leia',
            'SKYWALKER': 'Luke Skywalker',
            'SOLO': 'Han Solo',
            'ORGANA': 'Princess Leia',
            'KENOBI': 'Obi-Wan Kenobi',
            'CALRISSIAN': 'Lando Calrissian',
            'PALPATINE': 'Emperor Palpatine',
            'HUTT': 'Jabba the Hutt',
            'ANTILLES': 'Wedge Antilles',
            'MOTHMA': 'Mon Mothma',
            'FETT': 'Boba Fett'
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
    
    def generate_embedding(self, text: str) -> List[float]:
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
    
    def is_character_name(self, name: str) -> bool:
        """Check if a name is actually a character name, not a scene description."""
        name = name.strip().upper()
        
        # Skip if it's a scene keyword
        if name in self.scene_keywords:
            return False
        
        # Skip if it contains scene-related words
        for keyword in self.scene_keywords:
            if keyword in name:
                return False
        
        # Skip if it's too long (likely a scene description)
        if len(name) > 20:
            return False
        
        # Skip if it contains punctuation that suggests it's a location
        if any(char in name for char in ['–', '-', ':', '(', ')', '.']):
            return False
        
        # Check if it's a known character name
        if name in self.character_names:
            return True
        
        # Check if it's a variation of known names
        for char_name in self.character_names:
            if char_name in name or name in char_name:
                return True
        
        # If it's short and doesn't contain scene keywords, it might be a character
        if len(name) <= 15 and not any(keyword in name for keyword in self.scene_keywords):
            return True
        
        return False
    
    def normalize_character_name(self, name: str) -> str:
        """Normalize character name using mappings."""
        name = name.strip().upper()
        
        # Check direct mapping
        if name in self.character_mappings:
            return self.character_mappings[name]
        
        # Handle variations
        if 'THREEPIO' in name or 'C-3PO' in name:
            return 'C-3PO'
        if 'ARTOO' in name or 'R2-D2' in name:
            return 'R2-D2'
        if 'STORM' in name or 'TROOPER' in name:
            return 'Stormtrooper'
        if 'IMPERIAL' in name:
            return 'Imperial Officer'
        if 'REBEL' in name:
            return 'Rebel Soldier'
        if 'DARTH' in name:
            return 'Darth Vader'
        if 'PRINCESS' in name:
            return 'Princess Leia'
        if 'SKYWALKER' in name:
            return 'Luke Skywalker'
        if 'SOLO' in name:
            return 'Han Solo'
        if 'KENOBI' in name:
            return 'Obi-Wan Kenobi'
        if 'CALRISSIAN' in name:
            return 'Lando Calrissian'
        if 'PALPATINE' in name:
            return 'Emperor Palpatine'
        if 'HUTT' in name:
            return 'Jabba the Hutt'
        if 'FETT' in name:
            return 'Boba Fett'
        
        # Default: return original name
        return name.title()
    
    def extract_dialogue_from_script(self, script_path: Path) -> List[Dict[str, Any]]:
        dialogue_lines = []
        
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            current_character = None
            current_dialogue = []
            line_number = 0
            
            for line in lines:
                line_number += 1
                line = line.strip()
                
                if not line:
                    continue
                
                # Check if line is a character name (ALL CAPS and not a scene description)
                if (line.isupper() and len(line) > 2 and self.is_character_name(line)):
                    
                    # Save previous dialogue
                    if current_character and current_dialogue:
                        dialogue_text = ' '.join(current_dialogue).strip()
                        if dialogue_text:
                            dialogue_lines.append({
                                'character': current_character,
                                'dialogue': dialogue_text,
                                'line_number': line_number - len(current_dialogue)
                            })
                    
                    # Start new character
                    current_character = self.normalize_character_name(line)
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
            
            logger.info(f"📝 Extracted {len(dialogue_lines)} character dialogue lines from {script_path.name}")
            return dialogue_lines
            
        except Exception as e:
            logger.error(f"❌ Failed to extract from {script_path}: {e}")
            return []
    
    async def process_all_characters(self):
        """Process all dialogue and identify top 30 characters."""
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
            
            # Process all scripts and collect character counts
            all_dialogue = []
            character_counts = Counter()
            
            script_dir = Path("data/raw")
            for script_file in script_dir.glob("*.txt"):
                if script_file.name not in self.movie_mappings:
                    continue
                
                logger.info(f"🎬 Processing {script_file.name}...")
                dialogue_lines = self.extract_dialogue_from_script(script_file)
                
                for dialogue in dialogue_lines:
                    all_dialogue.append(dialogue)
                    character_counts[dialogue['character']] += 1
            
            # Get top 30 characters
            top_characters = character_counts.most_common(30)
            logger.info("🏆 Top 30 characters by dialogue count:")
            for i, (char, count) in enumerate(top_characters, 1):
                logger.info(f"  {i:2d}. {char}: {count} lines")
            
            # Add top characters to database
            async with self.db_pool.acquire() as conn:
                for char_name, _ in top_characters:
                    await conn.execute(
                        "INSERT INTO characters (name) VALUES ($1) ON CONFLICT (name) DO NOTHING",
                        char_name
                    )
                logger.info(f"✅ Added {len(top_characters)} top characters")
            
            # Process dialogue for top characters only
            total_added = 0
            async with self.db_pool.acquire() as conn:
                for script_file in script_dir.glob("*.txt"):
                    if script_file.name not in self.movie_mappings:
                        continue
                    
                    movie_id = await conn.fetchval(
                        "SELECT id FROM movies WHERE title = $1",
                        self.movie_mappings[script_file.name]['title']
                    )
                    
                    dialogue_lines = self.extract_dialogue_from_script(script_file)
                    
                    for dialogue_data in dialogue_lines:
                        char_name = dialogue_data['character']
                        
                        # Only process top 30 characters
                        if char_name not in [char for char, _ in top_characters]:
                            continue
                        
                        character_id = await conn.fetchval(
                            "SELECT id FROM characters WHERE name = $1",
                            char_name
                        )
                        
                        if character_id:
                            embedding = self.generate_embedding(dialogue_data['dialogue'])
                            if embedding:
                                await conn.execute(
                                    """INSERT INTO dialogue_lines 
                                       (movie_id, character_id, dialogue, cleaned_dialogue, line_number, embedding) 
                                       VALUES ($1, $2, $3, $4, $5, $6::vector)""",
                                    movie_id, character_id, dialogue_data['dialogue'],
                                    dialogue_data['dialogue'], dialogue_data['line_number'], str(embedding)
                                )
                                total_added += 1
            
            logger.info(f"🎉 Added {total_added} dialogue lines for top 30 characters")
            
            # Final stats
            async with self.db_pool.acquire() as conn:
                movie_count = await conn.fetchval("SELECT COUNT(*) FROM movies")
                character_count = await conn.fetchval("SELECT COUNT(*) FROM characters")
                dialogue_count = await conn.fetchval("SELECT COUNT(*) FROM dialogue_lines")
                
                logger.info(f"📊 Final database:")
                logger.info(f"   Movies: {movie_count}")
                logger.info(f"   Characters: {character_count}")
                logger.info(f"   Dialogue lines: {dialogue_count}")
            
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            raise

async def main():
    logger.info("🌟 Fixed Dialogue Processing")
    logger.info("=" * 50)
    
    processor = FixedDialogueProcessor()
    
    try:
        await processor.initialize_database()
        processor.load_embedding_model()
        await processor.process_all_characters()
        
        logger.info("🎉 Fixed dialogue processing completed!")
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        raise
    finally:
        if processor.db_pool:
            await processor.db_pool.close()

if __name__ == "__main__":
    asyncio.run(main())
