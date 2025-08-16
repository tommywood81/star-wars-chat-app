#!/usr/bin/env python3
"""
Demo script for Star Wars RAG Chat Application.

This script demonstrates the complete pipeline from script processing to character chat.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from star_wars_rag import StarWarsRAGApp

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run the Star Wars RAG demo."""
    print("🌟 Star Wars RAG Chat Application Demo")
    print("=" * 50)
    
    # Initialize the application
    print("\n1. Initializing RAG application...")
    app = StarWarsRAGApp()
    
    # Check for script data
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print("❌ Error: data/raw directory not found!")
        print("Please ensure you have Star Wars script files in data/raw/")
        return
    
    script_files = list(data_dir.glob("*.txt"))
    if not script_files:
        print("❌ Error: No script files found in data/raw/")
        print("Please add Star Wars script files (*.txt) to data/raw/")
        return
    
    print(f"Found {len(script_files)} script files:")
    for script in script_files[:5]:  # Show first 5
        print(f"  - {script.name}")
    if len(script_files) > 5:
        print(f"  ... and {len(script_files) - 5} more")
    
    # Load the system (use just one script for demo speed)
    print(f"\n2. Loading dialogue data from scripts...")
    try:
        # Use A New Hope for demo (or first available script)
        demo_script = None
        for script in script_files:
            if "NEW HOPE" in script.name.upper():
                demo_script = script
                break
        
        if demo_script is None:
            demo_script = script_files[0]  # Use first available
        
        print(f"Processing: {demo_script.name}")
        
        # Create a temporary directory with just this script
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_script_dir = Path(temp_dir) / "scripts"
            temp_script_dir.mkdir()
            
            # Copy the demo script
            temp_script = temp_script_dir / demo_script.name
            temp_script.write_text(demo_script.read_text(encoding='utf-8'), encoding='utf-8')
            
            # Load the system
            app.load_from_scripts(temp_script_dir, pattern=demo_script.name)
        
    except Exception as e:
        print(f"❌ Error loading scripts: {e}")
        print("Trying to load from processed data instead...")
        
        # Try to load from previously processed data
        processed_file = Path("data/processed/a_new_hope_dialogue.csv")
        if processed_file.exists():
            app.load_from_processed_data(processed_file)
        else:
            print("❌ No processed data found either. Please check your data files.")
            return
    
    # Display system statistics
    print("\n3. System Statistics:")
    try:
        stats = app.get_system_stats()
        print(f"  📊 Total dialogue lines: {stats['total_dialogue_lines']:,}")
        print(f"  👥 Characters found: {stats['num_characters']}")
        print(f"  🎬 Movies: {stats['num_movies']}")
        print(f"  🤖 Embedding model: {stats['embedding_model']}")
        print(f"  📏 Embedding dimension: {stats['embedding_dimension']}")
        
        print(f"\n  🌟 Top Characters:")
        for char, count in list(stats['top_characters'].items())[:5]:
            print(f"    - {char}: {count} lines")
        
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        return
    
    # Test retrieval quality
    print("\n4. Testing Retrieval Quality:")
    try:
        quality_results = app.test_retrieval_quality()
        print(f"  🔍 Test queries: {len(quality_results['test_queries'])}")
        print(f"  📈 Total results found: {quality_results['total_results_found']}")
        print(f"  📊 Average results per query: {quality_results['average_results_per_query']:.1f}")
        
        # Show best result for each query
        print(f"\n  🎯 Sample Results:")
        for query, result in list(quality_results['query_results'].items())[:3]:
            if result['results']:
                best = result['results'][0]
                print(f"    Query: '{query}'")
                print(f"    → [{best['similarity']:.3f}] {best['character']}: {best['dialogue'][:60]}...")
                print()
        
    except Exception as e:
        print(f"❌ Error testing quality: {e}")
    
    # Interactive character chat demo
    print("\n5. Character Chat Demo:")
    try:
        characters = app.retriever.get_available_characters()[:5]  # Top 5 characters
        
        demo_queries = [
            "Tell me about the Force",
            "I need help with something dangerous", 
            "What should I do in this situation?",
            "Can you help me understand this?"
        ]
        
        for i, query in enumerate(demo_queries[:2]):  # Show 2 examples
            print(f"\n  💬 Demo Chat {i+1}:")
            print(f"  Query: '{query}'")
            
            # Try with first available character
            if characters:
                char = characters[0]
                try:
                    response = app.chat_with_character(query, char)
                    print(f"  {char}: {response['response']}")
                    print(f"  (Similarity: {response['similarity']:.3f}, Movie: {response.get('movie', 'Unknown')})")
                except Exception as e:
                    print(f"  ❌ Error chatting with {char}: {e}")
            
            # Show general search results
            try:
                search_results = app.search_dialogue(query, top_k=2)
                if search_results:
                    print(f"  🔍 Other relevant dialogue:")
                    for result in search_results:
                        print(f"    - [{result['similarity']:.3f}] {result['character']}: {result['dialogue'][:50]}...")
            except Exception as e:
                print(f"  ❌ Error in search: {e}")
    
    except Exception as e:
        print(f"❌ Error in character chat demo: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 Demo completed successfully!")
    print("\nThis demo showed:")
    print("  ✅ Script processing and dialogue extraction")
    print("  ✅ Semantic embedding generation")
    print("  ✅ Similarity-based dialogue retrieval")
    print("  ✅ Character-specific responses")
    print("  ✅ System statistics and quality metrics")
    
    print(f"\n🚀 The Star Wars RAG system is ready for:")
    print("  - Character-based chatbots")
    print("  - Dialogue search and retrieval")
    print("  - Quote finding and similarity matching")
    print("  - Integration with larger language models")
    
    print(f"\n📖 Next Steps:")
    print("  - Run tests: python -m pytest tests/ -v")
    print("  - Try custom queries with the loaded system")
    print("  - Integrate with LLM for full chat functionality")
    print("  - Deploy as web application or API")


if __name__ == "__main__":
    main()
