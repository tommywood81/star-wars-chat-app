#!/usr/bin/env python3
"""
Enhanced Embedding Generator for LLM-Processed Dialogue

This script creates embeddings that include both the dialogue and its rich context.
Follows best practices for embedding generation and storage.
"""

import sys
import json
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add src to path
sys.path.append('src')

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedEmbeddingGenerator:
    """Generate embeddings that include dialogue + rich context for better semantic search."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required: pip install sentence-transformers")
        
        self.model_name = model_name
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"🤖 Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
    
    def create_enhanced_text(self, dialogue_item: Dict[str, Any]) -> str:
        """Create enhanced text that includes dialogue + context for better embeddings.
        
        Args:
            dialogue_item: Dictionary containing dialogue and context fields
            
        Returns:
            Enhanced text string for embedding
        """
        speaker = dialogue_item.get('speaker', 'Unknown')
        dialogue = dialogue_item.get('dialogue', '')
        movie = dialogue_item.get('movie', '')
        
        # Core dialogue
        enhanced_text = f"{speaker}: {dialogue}"
        
        # Add contextual information for richer embeddings
        context_parts = []
        
        # Who they're talking to
        addressee = dialogue_item.get('addressee', '')
        if addressee and addressee not in ['unknown', 'others']:
            context_parts.append(f"speaking to {addressee}")
        
        # Emotional context
        emotion = dialogue_item.get('emotion', '')
        if emotion and emotion not in ['neutral', 'unknown']:
            context_parts.append(f"feeling {emotion}")
        
        # Location context
        location = dialogue_item.get('location', '')
        if location and location not in ['scene_location', 'unknown']:
            context_parts.append(f"at {location}")
        
        # Stakes/situation
        stakes = dialogue_item.get('stakes', '')
        if stakes and stakes not in ['story_progression', 'unknown']:
            context_parts.append(f"stakes: {stakes}")
        
        # Scene context
        scene_context = dialogue_item.get('context', '')
        if scene_context and scene_context != 'Context unavailable':
            context_parts.append(f"scene: {scene_context}")
        
        # Add movie context
        if movie:
            context_parts.append(f"from {movie}")
        
        # Combine all context
        if context_parts:
            enhanced_text += f" ({', '.join(context_parts)})"
        
        return enhanced_text
    
    def generate_embeddings(self, dialogue_data: List[Dict[str, Any]], 
                          batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for enhanced dialogue texts.
        
        Args:
            dialogue_data: List of dialogue items with context
            batch_size: Batch size for embedding generation
            
        Returns:
            Numpy array of embeddings
        """
        logger.info(f"🔄 Generating enhanced embeddings for {len(dialogue_data)} items...")
        
        # Create enhanced texts
        enhanced_texts = []
        for i, item in enumerate(dialogue_data):
            if i % 100 == 0:
                logger.info(f"📝 Processing text {i+1}/{len(dialogue_data)}")
            
            enhanced_text = self.create_enhanced_text(item)
            enhanced_texts.append(enhanced_text)
        
        # Generate embeddings in batches
        logger.info("🤖 Generating embeddings...")
        embeddings = self.model.encode(
            enhanced_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        logger.info(f"✅ Generated embeddings: shape {embeddings.shape}")
        return embeddings
    
    def save_enhanced_data(self, dialogue_data: List[Dict[str, Any]], 
                          embeddings: np.ndarray, output_dir: str = "data/enhanced_embeddings"):
        """Save enhanced dialogue data and embeddings.
        
        Args:
            dialogue_data: List of dialogue items with context
            embeddings: Generated embeddings
            output_dir: Output directory for saved data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        embeddings_file = output_path / "enhanced_embeddings.npy"
        np.save(embeddings_file, embeddings)
        logger.info(f"💾 Saved embeddings: {embeddings_file}")
        
        # Save enhanced dialogue data
        dialogue_file = output_path / "enhanced_dialogue_data.json"
        enhanced_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_items": len(dialogue_data),
                "embedding_model": self.model_name,
                "embedding_dimension": embeddings.shape[1],
                "description": "Enhanced dialogue data with rich context for better semantic search"
            },
            "dialogue_items": dialogue_data
        }
        
        with open(dialogue_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Saved dialogue data: {dialogue_file}")
        
        # Save sample enhanced texts for inspection
        sample_file = output_path / "enhanced_texts_sample.txt"
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write("Enhanced Text Samples (Dialogue + Context)\n")
            f.write("=" * 60 + "\n\n")
            
            for i, item in enumerate(dialogue_data[:20]):  # First 20 samples
                enhanced_text = self.create_enhanced_text(item)
                f.write(f"Sample {i+1}:\n")
                f.write(f"Original: {item.get('speaker', 'Unknown')}: {item.get('dialogue', '')}\n")
                f.write(f"Enhanced: {enhanced_text}\n")
                f.write("-" * 40 + "\n")
        
        logger.info(f"💾 Saved text samples: {sample_file}")


def main():
    """Main function to generate enhanced embeddings from pipeline data."""
    print("🌟 Enhanced Embedding Generator")
    print("=" * 50)
    
    # Load pipeline-processed data
    pipeline_file = Path("data/pipeline_processed/pipeline_enhanced_complete.json")
    
    if not pipeline_file.exists():
        print(f"❌ Pipeline data not found: {pipeline_file}")
        print("💡 Run smart_pipeline_preprocessor.py first to generate enhanced data")
        return 1
    
    try:
        print(f"📖 Loading pipeline data: {pipeline_file}")
        with open(pipeline_file, 'r', encoding='utf-8') as f:
            pipeline_data = json.load(f)
        
        dialogue_lines = pipeline_data.get('dialogue_lines', [])
        if not dialogue_lines:
            print("❌ No dialogue lines found in pipeline data")
            return 1
        
        print(f"✅ Loaded {len(dialogue_lines)} dialogue lines")
        
        # Check if any lines have LLM processing
        llm_processed = sum(1 for item in dialogue_lines 
                           if item.get('processing_method') == 'llm_enhanced')
        
        print(f"📊 LLM-enhanced lines: {llm_processed}")
        print(f"📊 Rule-based lines: {len(dialogue_lines) - llm_processed}")
        
        # Initialize embedding generator
        generator = EnhancedEmbeddingGenerator()
        
        # Generate enhanced embeddings
        embeddings = generator.generate_embeddings(dialogue_lines)
        
        # Save enhanced data
        generator.save_enhanced_data(dialogue_lines, embeddings)
        
        print(f"\n🎉 Enhanced embeddings created successfully!")
        print(f"📁 Output: data/enhanced_embeddings/")
        print(f"📊 Embeddings shape: {embeddings.shape}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error generating enhanced embeddings: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
