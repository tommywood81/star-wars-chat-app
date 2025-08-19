#!/usr/bin/env python3
"""
Complete Star Wars RAG Data Processing Pipeline

This module orchestrates the full pipeline from raw scripts to embedded,
context-enhanced dialogue using Phi-2 LLM for intelligent preprocessing.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.star_wars_rag.data_processor import DialogueProcessor
from src.star_wars_rag.llm_script_preprocessor import LLMScriptPreprocessor
from src.star_wars_rag.app import StarWarsRAGApp

logger = logging.getLogger(__name__)

class StarWarsCompletePipeline:
    """Complete pipeline for processing Star Wars scripts with LLM enhancement."""
    
    def __init__(self, 
                 raw_scripts_dir: str = "data/raw",
                 output_dir: str = "data/processed_complete",
                 model_path: Optional[str] = None):
        
        self.raw_scripts_dir = Path(raw_scripts_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.dialogue_processor = DialogueProcessor()
        self.llm_preprocessor = LLMScriptPreprocessor(
            model_path=model_path,
            raw_scripts_dir=raw_scripts_dir,
            output_dir=str(self.output_dir)
        )
        
        # Pipeline statistics
        self.stats = {
            'total_lines_extracted': 0,
            'total_lines_enhanced': 0,
            'movies_processed': 0,
            'characters_found': 0,
            'processing_time': 0
        }
    
    def run_complete_pipeline(self, use_llm_enhancement: bool = True) -> Dict[str, Any]:
        """Run the complete data processing pipeline.
        
        Args:
            use_llm_enhancement: Whether to use Phi-2 LLM for context enhancement
            
        Returns:
            Pipeline statistics and results
        """
        start_time = time.time()
        logger.info("🚀 Starting complete Star Wars RAG pipeline...")
        
        # Step 1: Extract ALL dialogue lines from all movies
        logger.info("📖 Step 1: Extracting dialogue from all Original Trilogy scripts...")
        all_dialogue_data = self._extract_all_dialogue()
        
        # Step 2: Apply LLM enhancement if enabled
        if use_llm_enhancement and self.llm_preprocessor.llm:
            logger.info("🤖 Step 2: Applying Phi-2 LLM enhancement...")
            enhanced_data = self._enhance_with_llm(all_dialogue_data)
        else:
            logger.info("📝 Step 2: Using rule-based context extraction...")
            enhanced_data = self._enhance_with_rules(all_dialogue_data)
        
        # Step 3: Save processed data
        logger.info("💾 Step 3: Saving complete processed dataset...")
        self._save_complete_dataset(enhanced_data)
        
        # Step 4: Generate embeddings and test RAG system
        logger.info("🔍 Step 4: Testing RAG system with complete dataset...")
        rag_stats = self._test_rag_system()
        
        # Final statistics
        total_time = time.time() - start_time
        self.stats['processing_time'] = total_time
        
        logger.info(f"✅ Pipeline complete in {total_time:.2f} seconds!")
        logger.info(f"📊 Total dialogue lines processed: {self.stats['total_lines_extracted']:,}")
        logger.info(f"🎭 Characters found: {self.stats['characters_found']}")
        logger.info(f"🎬 Movies processed: {self.stats['movies_processed']}")
        
        return {
            'pipeline_stats': self.stats,
            'rag_stats': rag_stats,
            'success': True
        }
    
    def _extract_all_dialogue(self) -> List[Dict]:
        """Extract dialogue from all Original Trilogy scripts."""
        trilogy_files = [
            "STAR WARS A NEW HOPE.txt",
            "THE EMPIRE STRIKES BACK.txt", 
            "STAR WARS THE RETURN OF THE JEDI.txt"
        ]
        
        all_dialogue = []
        
        for filename in trilogy_files:
            script_path = self.raw_scripts_dir / filename
            if not script_path.exists():
                logger.warning(f"Script not found: {filename}")
                continue
            
            try:
                # Load script content
                script_content = self.dialogue_processor.load_script(script_path)
                movie_name = filename.replace('.txt', '').replace('_', ' ').title()
                
                # Extract dialogue using robust processor
                dialogue_lines = self.dialogue_processor.extract_dialogue_lines(
                    script_content, movie_name
                )
                
                logger.info(f"📄 {movie_name}: {len(dialogue_lines)} dialogue lines extracted")
                all_dialogue.extend(dialogue_lines)
                self.stats['movies_processed'] += 1
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
        
        # Update statistics
        self.stats['total_lines_extracted'] = len(all_dialogue)
        
        # Count unique characters
        characters = set()
        for line in all_dialogue:
            char = self.dialogue_processor.normalize_character_name(line['character'])
            if char:
                characters.add(char)
        self.stats['characters_found'] = len(characters)
        
        logger.info(f"🎯 Total dialogue extracted: {len(all_dialogue):,} lines")
        logger.info(f"👥 Unique characters: {len(characters)}")
        
        return all_dialogue
    
    def _enhance_with_llm(self, dialogue_data: List[Dict]) -> List[Dict]:
        """Enhance dialogue with Phi-2 LLM context analysis."""
        enhanced_data = []
        
        logger.info("🤖 Using Phi-2 LLM for intelligent context enhancement...")
        
        for i, line in enumerate(dialogue_data):
            if i % 100 == 0:  # Progress logging
                logger.info(f"Processing line {i+1}/{len(dialogue_data)}")
            
            try:
                # Use LLM to enhance context
                character = line['character']
                dialogue = line['dialogue']
                basic_scene = line.get('scene', 'Unknown')
                
                # Create context chunk (simulate surrounding script)
                context_chunk = f"{basic_scene}\n\n{character} {dialogue}"
                
                # Get LLM-enhanced context
                enhanced_context = self.llm_preprocessor.extract_scene_context_with_llm(
                    context_chunk, character, dialogue
                )
                
                # Get character insights
                enhanced_line = self.llm_preprocessor.enhance_dialogue_with_llm(
                    character, dialogue, enhanced_context
                )
                
                # Merge with original data
                enhanced_line.update({
                    'line_number': line['line_number'],
                    'scene': line['scene'],
                    'movie': line['movie'],
                    'character_normalized': self.dialogue_processor.normalize_character_name(character)
                })
                
                enhanced_data.append(enhanced_line)
                
            except Exception as e:
                logger.warning(f"LLM enhancement failed for line {i}: {e}")
                # Fallback to rule-based enhancement
                enhanced_data.append(self._enhance_line_with_rules(line))
        
        self.stats['total_lines_enhanced'] = len(enhanced_data)
        return enhanced_data
    
    def _enhance_with_rules(self, dialogue_data: List[Dict]) -> List[Dict]:
        """Enhance dialogue with rule-based context extraction."""
        enhanced_data = []
        
        logger.info("📝 Using rule-based context enhancement...")
        
        for line in dialogue_data:
            enhanced_line = self._enhance_line_with_rules(line)
            enhanced_data.append(enhanced_line)
        
        self.stats['total_lines_enhanced'] = len(enhanced_data)
        return enhanced_data
    
    def _enhance_line_with_rules(self, line: Dict) -> Dict:
        """Apply rule-based enhancement to a single dialogue line."""
        character = line['character']
        dialogue = line['dialogue']
        scene = line.get('scene', 'Unknown')
        
        # Basic scene context
        context = f"Scene: {scene}"
        
        # Simple character state inference based on dialogue patterns
        character_state = "Unknown"
        dialogue_type = "Unknown"
        
        if '!' in dialogue:
            character_state = "Excited or Urgent"
            dialogue_type = "Exclamation"
        elif '?' in dialogue:
            character_state = "Inquisitive"
            dialogue_type = "Question"
        elif dialogue.isupper():
            character_state = "Angry or Commanding"
            dialogue_type = "Command"
        else:
            character_state = "Calm"
            dialogue_type = "Statement"
        
        return {
            'character': character,
            'character_normalized': self.dialogue_processor.normalize_character_name(character),
            'dialogue': dialogue,
            'context': context,
            'character_state': character_state,
            'dialogue_type': dialogue_type,
            'motivation': "Context-derived",
            'line_number': line['line_number'],
            'scene': line['scene'],
            'movie': line['movie']
        }
    
    def _save_complete_dataset(self, enhanced_data: List[Dict]) -> None:
        """Save the complete enhanced dataset."""
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(enhanced_data)
        
        # Save as CSV
        csv_path = self.output_dir / "complete_enhanced_dialogue.csv"
        df.to_csv(csv_path, index=False)
        
        # Save as text format
        txt_path = self.output_dir / "complete_enhanced_dialogue.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("# Complete Enhanced Star Wars Original Trilogy Dialogue\n")
            f.write("# Format: CHARACTER | DIALOGUE | CONTEXT | CHARACTER_STATE | DIALOGUE_TYPE | MOVIE\n\n")
            
            for line in enhanced_data:
                f.write(f"{line['character_normalized']} | {line['dialogue']} | "
                       f"{line['context']} | {line['character_state']} | "
                       f"{line['dialogue_type']} | {line['movie']}\n")
        
        logger.info(f"💾 Saved complete dataset: {len(enhanced_data):,} lines")
        logger.info(f"📄 CSV: {csv_path}")
        logger.info(f"📄 TXT: {txt_path}")
    
    def _test_rag_system(self) -> Dict[str, Any]:
        """Test the RAG system with the complete dataset."""
        try:
            # Initialize RAG app
            rag_app = StarWarsRAGApp()
            
            # Load the complete processed data
            csv_path = self.output_dir / "complete_enhanced_dialogue.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                
                # Load into RAG system
                rag_app.retriever.load_dialogue_data(df)
                rag_app.is_loaded = True
                
                # Get system statistics
                stats = rag_app.get_system_stats()
                
                logger.info(f"🔍 RAG System Loaded:")
                logger.info(f"   📊 Total dialogue lines: {stats['total_dialogue_lines']:,}")
                logger.info(f"   👥 Characters: {stats['num_characters']}")
                logger.info(f"   🎬 Movies: {stats['num_movies']}")
                
                return stats
            else:
                logger.error("Complete dataset CSV not found for RAG testing")
                return {}
                
        except Exception as e:
            logger.error(f"RAG system testing failed: {e}")
            return {}


def main():
    """Main function to run the complete pipeline."""
    logging.basicConfig(level=logging.INFO)
    
    print("🌟 Star Wars RAG Complete Pipeline")
    print("=" * 50)
    
    # Check for Phi-2 model
    model_path = "models/phi-2-q4.gguf"
    if not Path(model_path).exists():
        print(f"⚠️  Phi-2 model not found at {model_path}")
        print("📝 Running with rule-based enhancement")
        model_path = None
    else:
        print(f"🤖 Using Phi-2 LLM: {model_path}")
    
    # Initialize and run pipeline
    pipeline = StarWarsCompletePipeline(model_path=model_path)
    
    try:
        results = pipeline.run_complete_pipeline(use_llm_enhancement=(model_path is not None))
        
        print("\n✅ Pipeline Results:")
        print("-" * 30)
        for key, value in results['pipeline_stats'].items():
            print(f"{key}: {value:,}" if isinstance(value, (int, float)) else f"{key}: {value}")
        
        if results['rag_stats']:
            print(f"\n🔍 RAG System Ready:")
            print(f"Total lines: {results['rag_stats']['total_dialogue_lines']:,}")
            print(f"Characters: {results['rag_stats']['num_characters']}")
            
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
