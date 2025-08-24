#!/usr/bin/env python3
"""
Process real Star Wars dialogue from script files.

This script uses the improved script processor to extract authentic dialogue
from the Star Wars script files and prepare it for use in the RAG system.
"""

import logging
from pathlib import Path
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from star_wars_rag.improved_script_processor import process_all_scripts, ImprovedScriptProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to process all Star Wars scripts."""
    # Define paths
    scripts_dir = Path("data/raw")
    output_dir = Path("data/processed")
    
    # Check if scripts directory exists
    if not scripts_dir.exists():
        logger.error(f"Scripts directory not found: {scripts_dir}")
        logger.info("Please ensure the script files are in the data/raw directory")
        return
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    logger.info("Starting Star Wars script processing...")
    logger.info(f"Input directory: {scripts_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Process all scripts
    try:
        dialogue_lines = process_all_scripts(scripts_dir, output_dir)
        
        logger.info(f"Successfully processed {len(dialogue_lines)} dialogue lines")
        
        # Show some sample output
        if dialogue_lines:
            logger.info("\nSample dialogue lines:")
            for i, line in enumerate(dialogue_lines[:5]):
                logger.info(f"  {i+1}. {line.normalized_character}: {line.dialogue[:80]}...")
        
        # Show character statistics
        processor = ImprovedScriptProcessor()
        stats = processor.get_character_statistics(dialogue_lines)
        
        logger.info(f"\nTop 10 characters by dialogue lines:")
        for char, count in sorted(stats.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {char}: {count} lines")
        
        logger.info(f"\nProcessing complete! Output files:")
        logger.info(f"  - {output_dir / 'all_dialogue.json'}")
        logger.info(f"  - {output_dir / 'character_stats.json'}")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()
