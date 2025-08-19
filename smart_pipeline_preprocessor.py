#!/usr/bin/env python3
"""
Smart Pipeline-Based LLM Preprocessor with Progress Tracking

Features:
- Pipeline architecture with individual movie processing
- Progress indicators and detailed logging
- Smart caching (checks for existing preprocessed data)
- Estimated time calculations
- Best practices for data pipeline management
"""

import sys
import os
import logging
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import hashlib

# Add src to path
sys.path.append('src')

try:
    from star_wars_rag.llm import LocalLLM
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline_processing.log')
    ]
)
logger = logging.getLogger(__name__)

class SmartPipelinePreprocessor:
    """Pipeline-based preprocessor with smart caching and progress tracking."""
    
    def __init__(self, model_path: Optional[str] = None, output_dir: str = "data/pipeline_processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Cache directory for intermediate results
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        self.llm = None
        self.processing_stats = {
            'total_start_time': None,
            'movies_processed': 0,
            'total_lines_processed': 0,
            'llm_analyses': 0,
            'cache_hits': 0,
            'processing_rate': 0
        }
        
        # Initialize LLM if available
        if model_path and os.path.exists(model_path) and LLM_AVAILABLE:
            try:
                logger.info(f"🤖 Initializing Phi-2 LLM: {model_path}")
                self.llm = LocalLLM(
                    model_path=model_path,
                    n_ctx=1024,  # Balanced context size for speed
                    n_threads=2,
                    verbose=False
                )
                logger.info("✅ LLM successfully loaded")
            except Exception as e:
                logger.error(f"❌ LLM loading failed: {e}")
        else:
            logger.info("📝 No LLM model - using rule-based processing")
    
    def get_file_hash(self, file_path: Path) -> str:
        """Generate hash of file for cache validation."""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def check_cache(self, script_path: Path) -> Optional[Dict[str, Any]]:
        """Check if preprocessed data exists and is current."""
        cache_file = self.cache_dir / f"{script_path.stem}_processed.json"
        
        if not cache_file.exists():
            logger.info(f"📄 No cache found for {script_path.name}")
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Check if source file has changed
            current_hash = self.get_file_hash(script_path)
            cached_hash = cached_data.get('metadata', {}).get('source_hash', '')
            
            if current_hash == cached_hash:
                logger.info(f"✅ Cache hit for {script_path.name} ({len(cached_data.get('dialogue_lines', []))} lines)")
                self.processing_stats['cache_hits'] += 1
                return cached_data
            else:
                logger.info(f"🔄 Cache outdated for {script_path.name} - will reprocess")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Cache read error for {script_path.name}: {e}")
            return None
    
    def save_to_cache(self, script_path: Path, processed_data: List[Dict], metadata: Dict) -> None:
        """Save processed data to cache."""
        cache_file = self.cache_dir / f"{script_path.stem}_processed.json"
        
        cache_data = {
            'metadata': {
                'source_file': str(script_path),
                'source_hash': self.get_file_hash(script_path),
                'processed_date': datetime.now().isoformat(),
                'processing_method': 'llm' if self.llm else 'rule_based',
                'total_lines': len(processed_data),
                **metadata
            },
            'dialogue_lines': processed_data
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Cached {len(processed_data)} lines for {script_path.name}")
        except Exception as e:
            logger.error(f"❌ Cache save error: {e}")
    
    def estimate_processing_time(self, script_path: Path) -> Tuple[int, str]:
        """Estimate processing time for a script."""
        try:
            with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Rough estimate: count potential dialogue lines
            lines = content.split('\n')
            potential_dialogue = 0
            
            for line in lines:
                line = line.strip()
                if line and re.match(r'^[A-Z][A-Z\s\-\']+\s+.+$', line) and len(line) < 200:
                    potential_dialogue += 1
            
            # Time estimates (based on testing)
            if self.llm:
                seconds_per_line = 0.5  # LLM processing is slower but more accurate
                method = "LLM"
            else:
                seconds_per_line = 0.01  # Rule-based is much faster
                method = "Rule-based"
            
            total_seconds = potential_dialogue * seconds_per_line
            
            if total_seconds < 60:
                time_str = f"{total_seconds:.0f} seconds"
            elif total_seconds < 3600:
                time_str = f"{total_seconds/60:.1f} minutes"
            else:
                time_str = f"{total_seconds/3600:.1f} hours"
            
            logger.info(f"📊 Estimated {potential_dialogue} lines, ~{time_str} ({method})")
            return potential_dialogue, time_str
            
        except Exception as e:
            logger.error(f"❌ Estimation error: {e}")
            return 0, "unknown"
    
    def analyze_dialogue_with_progress(self, script_chunk: str, character: str, dialogue: str, 
                                     line_idx: int, total_lines: int) -> Dict[str, Any]:
        """Analyze dialogue with progress reporting."""
        
        # Progress reporting every 25 lines
        if line_idx % 25 == 0:
            progress = (line_idx / total_lines) * 100
            logger.info(f"🔄 Progress: {progress:.1f}% ({line_idx}/{total_lines} lines)")
        
        if not self.llm:
            return self._rule_based_analysis(character, dialogue)
        
        try:
            # Streamlined LLM prompt for faster processing
            prompt = f"""Analyze this Star Wars dialogue for context:

DIALOGUE: {character}: "{dialogue}"
CONTEXT: {script_chunk[-500:]}  

Extract JSON:
{{"speaker": "{character}", "addressee": "who_spoken_to", "emotion": "emotion", "location": "where", "stakes": "what_at_stake"}}

JSON:"""

            response = self.llm.generate(
                prompt=prompt,
                max_tokens=150,  # Reduced for speed
                temperature=0.1,  # Very low for consistency
                stop=["\n\n", "DIALOGUE:", "}"]
            )
            
            self.processing_stats['llm_analyses'] += 1
            
            # Quick JSON extraction
            response_text = response.get('response', '').strip()
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                try:
                    json_text = response_text[json_start:json_end]
                    analysis = json.loads(json_text)
                    
                    # Validate and return
                    if 'speaker' in analysis and 'addressee' in analysis:
                        analysis['processing_method'] = 'llm'
                        return analysis
                except json.JSONDecodeError:
                    pass
            
            # Fallback to rule-based
            logger.debug(f"LLM parsing failed for {character}, using fallback")
            return self._rule_based_analysis(character, dialogue)
            
        except Exception as e:
            logger.debug(f"LLM error for {character}: {e}")
            return self._rule_based_analysis(character, dialogue)
    
    def _rule_based_analysis(self, character: str, dialogue: str) -> Dict[str, Any]:
        """Fast rule-based analysis for fallback."""
        dialogue_lower = dialogue.lower()
        
        # Quick emotion detection
        if '!' in dialogue or dialogue.isupper():
            emotion = "urgent"
        elif '?' in dialogue:
            emotion = "questioning"
        elif any(word in dialogue_lower for word in ['please', 'help']):
            emotion = "pleading"
        else:
            emotion = "neutral"
        
        # Character-based addressee guessing
        if 'luke' in character.lower():
            addressee = "allies"
        elif 'vader' in character.lower():
            addressee = "subordinates"
        else:
            addressee = "others"
        
        return {
            "speaker": character,
            "addressee": addressee,
            "emotion": emotion,
            "location": "scene_location",
            "stakes": "story_progression",
            "processing_method": "rule_based"
        }
    
    def process_single_movie(self, script_path: Path) -> Dict[str, Any]:
        """Pipeline stage: Process a single movie with full progress tracking."""
        
        movie_name = script_path.stem.replace('_', ' ').title()
        logger.info(f"🎬 PIPELINE: Processing {movie_name}")
        logger.info(f"=" * 60)
        
        # Stage 1: Check cache
        logger.info(f"🔍 Stage 1: Checking cache...")
        cached_data = self.check_cache(script_path)
        if cached_data:
            self.processing_stats['movies_processed'] += 1
            return cached_data
        
        # Stage 2: Estimate processing time
        logger.info(f"📊 Stage 2: Estimating processing time...")
        estimated_lines, time_estimate = self.estimate_processing_time(script_path)
        
        # Stage 3: Load and parse script
        logger.info(f"📖 Stage 3: Loading script ({script_path.name})...")
        try:
            with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"❌ Failed to load {script_path}: {e}")
            return {'dialogue_lines': [], 'metadata': {'error': str(e)}}
        
        # Stage 4: Extract dialogue lines
        logger.info(f"🎭 Stage 4: Extracting dialogue lines...")
        start_time = time.time()
        
        lines = content.split('\n')
        raw_dialogue = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            char_match = re.match(r'^([A-Z][A-Z\s\-\']+?)\s+(.+)$', line)
            
            if char_match and len(char_match.group(1)) < 50:
                character = char_match.group(1).strip()
                dialogue = char_match.group(2).strip()
                
                # Filter for quality dialogue
                if (len(dialogue) > 10 and 
                    not dialogue.startswith('(') and
                    not any(indicator in dialogue.upper() for indicator in ['INT.', 'EXT.', 'CUT TO:'])):
                    
                    # Clean dialogue
                    clean_dialogue = re.sub(r'\([^)]*\)', '', dialogue).strip()
                    clean_dialogue = re.sub(r'\s+', ' ', clean_dialogue).strip()
                    
                    if len(clean_dialogue) > 5:
                        raw_dialogue.append({
                            'character': character,
                            'dialogue': clean_dialogue,
                            'line_number': i + 1
                        })
        
        logger.info(f"✅ Extracted {len(raw_dialogue)} dialogue lines")
        
        # Stage 5: Context analysis with progress
        logger.info(f"🤖 Stage 5: Analyzing context ({len(raw_dialogue)} lines)...")
        if len(raw_dialogue) > 100:
            logger.info(f"⚠️ Large dataset detected - this may take {time_estimate}")
        
        processed_dialogue = []
        analysis_start = time.time()
        
        for idx, item in enumerate(raw_dialogue):
            # Get surrounding context
            context_start = max(0, item['line_number'] - 10)
            context_end = min(len(lines), item['line_number'] + 5)
            context_chunk = '\n'.join(lines[context_start:context_end])
            
            # Analyze with progress
            analysis = self.analyze_dialogue_with_progress(
                context_chunk, 
                item['character'], 
                item['dialogue'],
                idx + 1,
                len(raw_dialogue)
            )
            
            # Add metadata
            analysis.update({
                'dialogue': item['dialogue'],
                'movie': movie_name,
                'line_number': item['line_number'],
                'word_count': len(item['dialogue'].split())
            })
            
            processed_dialogue.append(analysis)
        
        analysis_time = time.time() - analysis_start
        total_time = time.time() - start_time
        
        # Stage 6: Save to cache
        logger.info(f"💾 Stage 6: Saving to cache...")
        metadata = {
            'processing_time_seconds': total_time,
            'analysis_time_seconds': analysis_time,
            'lines_per_second': len(processed_dialogue) / analysis_time if analysis_time > 0 else 0,
            'estimated_lines': estimated_lines,
            'actual_lines': len(processed_dialogue)
        }
        
        self.save_to_cache(script_path, processed_dialogue, metadata)
        
        # Update stats
        self.processing_stats['movies_processed'] += 1
        self.processing_stats['total_lines_processed'] += len(processed_dialogue)
        
        logger.info(f"✅ {movie_name} complete: {len(processed_dialogue)} lines in {total_time:.2f}s")
        logger.info(f"📊 Rate: {len(processed_dialogue) / total_time:.1f} lines/second")
        
        return {
            'dialogue_lines': processed_dialogue,
            'metadata': metadata
        }
    
    def run_complete_pipeline(self, raw_scripts_dir: str = "data/raw") -> Dict[str, Any]:
        """Run the complete pipeline with all stages."""
        
        logger.info(f"🚀 SMART PIPELINE STARTING")
        logger.info(f"=" * 60)
        
        self.processing_stats['total_start_time'] = time.time()
        
        # Pipeline Configuration
        trilogy_files = [
            "STAR WARS A NEW HOPE.txt",
            "THE EMPIRE STRIKES BACK.txt", 
            "STAR WARS THE RETURN OF THE JEDI.txt"
        ]
        
        scripts_dir = Path(raw_scripts_dir)
        all_results = {}
        all_dialogue = []
        
        # Process each movie individually
        for i, filename in enumerate(trilogy_files, 1):
            script_path = scripts_dir / filename
            
            if not script_path.exists():
                logger.warning(f"⚠️ Script not found: {filename}")
                continue
            
            logger.info(f"\n🎬 MOVIE {i}/{len(trilogy_files)}: {filename}")
            
            # Process single movie through pipeline
            result = self.process_single_movie(script_path)
            all_results[filename] = result
            all_dialogue.extend(result.get('dialogue_lines', []))
            
            # Inter-movie progress
            logger.info(f"🎯 Pipeline Progress: {i}/{len(trilogy_files)} movies complete")
        
        # Final consolidation
        total_time = time.time() - self.processing_stats['total_start_time']
        
        logger.info(f"\n🎉 PIPELINE COMPLETE!")
        logger.info(f"=" * 60)
        logger.info(f"📊 Total movies processed: {self.processing_stats['movies_processed']}")
        logger.info(f"📝 Total dialogue lines: {len(all_dialogue)}")
        logger.info(f"🤖 LLM analyses: {self.processing_stats['llm_analyses']}")
        logger.info(f"⚡ Cache hits: {self.processing_stats['cache_hits']}")
        logger.info(f"⏱️ Total time: {total_time:.2f} seconds")
        logger.info(f"📈 Overall rate: {len(all_dialogue) / total_time:.1f} lines/second")
        
        # Save combined results
        self.save_final_results(all_dialogue, all_results)
        
        return {
            'dialogue_lines': all_dialogue,
            'movie_results': all_results,
            'pipeline_stats': self.processing_stats,
            'total_time': total_time
        }
    
    def save_final_results(self, all_dialogue: List[Dict], movie_results: Dict) -> None:
        """Save final consolidated results."""
        
        # Save complete dataset
        final_path = self.output_dir / "pipeline_enhanced_complete.json"
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'processing_date': datetime.now().isoformat(),
                    'total_lines': len(all_dialogue),
                    'movies': list(movie_results.keys()),
                    'pipeline_stats': self.processing_stats
                },
                'dialogue_lines': all_dialogue
            }, f, indent=2, ensure_ascii=False)
        
        # Save readable format
        readable_path = self.output_dir / "pipeline_enhanced_readable.txt"
        with open(readable_path, 'w', encoding='utf-8') as f:
            f.write(f"# Pipeline-Enhanced Star Wars Dialogue\n")
            f.write(f"# Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total Lines: {len(all_dialogue):,}\n\n")
            
            for item in all_dialogue:
                f.write(f"SPEAKER: {item['speaker']} → {item.get('addressee', 'unknown')}\n")
                f.write(f"DIALOGUE: \"{item['dialogue']}\"\n")
                f.write(f"EMOTION: {item.get('emotion', 'unknown')}\n")
                f.write(f"LOCATION: {item.get('location', 'unknown')}\n")
                f.write(f"MOVIE: {item['movie']}\n")
                f.write("-" * 60 + "\n")
        
        logger.info(f"💾 Final results saved:")
        logger.info(f"   📄 JSON: {final_path}")
        logger.info(f"   📄 TXT: {readable_path}")


def main():
    """Main pipeline execution with command line argument support."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Smart Pipeline-Based Star Wars Preprocessor")
    parser.add_argument("--model-path", type=str, help="Path to the LLM model file")
    parser.add_argument("--force-reprocess", action="store_true", help="Force reprocessing even if cache exists")
    args = parser.parse_args()
    
    print("🌟 Smart Pipeline-Based Star Wars Preprocessor")
    print("=" * 60)
    
    # Model detection - use command line arg if provided
    model_path = None
    
    if args.model_path:
        if os.path.exists(args.model_path):
            model_path = args.model_path
            print(f"🤖 Using specified model: {model_path}")
        else:
            print(f"❌ Specified model not found: {args.model_path}")
            print("🔍 Searching for default models...")
    
    # If no command line model or it doesn't exist, search for defaults
    if not model_path:
        model_paths = [
            "models/phi-2.Q4_K_M.gguf",  # Actual file we have
            "models/phi-2-q4.gguf", 
            "models/phi-2.gguf",
            "models/phi-2.Q4_K_S.gguf"
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                print(f"🤖 Found LLM model: {model_path}")
                break
    
    if not model_path:
        print("📝 No LLM model found - using rule-based processing")
    
    # Initialize pipeline
    processor = SmartPipelinePreprocessor(model_path=model_path)
    
    # Clear cache if force reprocessing is requested
    if args.force_reprocess:
        print("🗑️ Force reprocessing: clearing cache...")
        cache_dir = Path("data/pipeline_processed/cache")
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            print("✅ Cache cleared")
    
    try:
        results = processor.run_complete_pipeline()
        
        print(f"\n✅ PIPELINE SUCCESS!")
        print(f"   📝 Processed: {len(results['dialogue_lines']):,} lines")
        print(f"   ⏱️ Time: {results['total_time']:.2f} seconds")
        print(f"   📁 Output: data/pipeline_processed/")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
