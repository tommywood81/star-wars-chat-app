#!/usr/bin/env python3
"""
Enhanced Star Wars Chat Demo with Local LLM Integration.

This demo showcases the complete RAG + LLM chat system with character interactions.
"""

import sys
import logging
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from star_wars_rag import StarWarsChatApp

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def display_typing_effect(text: str, delay: float = 0.03) -> None:
    """Display text with typing effect."""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()


def main():
    """Run the enhanced Star Wars chat demo."""
    print("🌟 Star Wars RAG + LLM Chat Demo")
    print("=" * 50)
    print("This demo shows the complete pipeline:")
    print("📖 Script Processing → 🧠 Embeddings → 🔍 RAG Retrieval → 🤖 Local LLM → 💬 Character Chat")
    print()
    
    # Initialize the chat application
    print("1. Initializing Enhanced Chat Application...")
    try:
        app = StarWarsChatApp(auto_download=True)
        print("✅ Chat application initialized")
    except Exception as e:
        print(f"❌ Failed to initialize chat app: {e}")
        print("🔧 This might be due to missing llama-cpp-python. Install with:")
        print("   pip install llama-cpp-python")
        return
    
    # Check LLM status
    print("\n2. Checking LLM Status...")
    llm_info = app.get_llm_info()
    
    if llm_info.get('llm_loaded'):
        print("✅ LLM loaded successfully")
        if 'model_name' in llm_info:
            print(f"📱 Model: {llm_info['model_name']}")
        if 'model_size_mb' in llm_info:
            print(f"💾 Size: {llm_info['model_size_mb']:.1f} MB")
    else:
        print("⚠️ Using MockLLM (llama-cpp-python not available or no model)")
        print("💡 For full functionality, ensure llama-cpp-python is installed")
    
    # Load dialogue data
    print("\n3. Loading Star Wars Dialogue Data...")
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print("❌ Error: data/raw directory not found!")
        return
    
    script_files = list(data_dir.glob("*.txt"))
    if not script_files:
        print("❌ Error: No script files found!")
        return
    
    try:
        # Use A New Hope for demo
        demo_script = None
        for script in script_files:
            if "NEW HOPE" in script.name.upper():
                demo_script = script
                break
        
        if demo_script is None:
            demo_script = script_files[0]
        
        print(f"📖 Processing: {demo_script.name}")
        
        # Create temp directory and load
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_script_dir = Path(temp_dir) / "scripts"
            temp_script_dir.mkdir()
            
            temp_script = temp_script_dir / demo_script.name
            temp_script.write_text(demo_script.read_text(encoding='utf-8'), encoding='utf-8')
            
            app.load_from_scripts(temp_script_dir, pattern=demo_script.name)
        
        print("✅ Dialogue data loaded successfully")
        
    except Exception as e:
        print(f"❌ Failed to load dialogue data: {e}")
        return
    
    # Display system statistics
    print("\n4. System Statistics:")
    try:
        stats = app.get_system_stats()
        print(f"📊 Total dialogue lines: {stats['total_dialogue_lines']:,}")
        print(f"👥 Characters: {stats['num_characters']}")
        print(f"🎬 Movies: {stats['num_movies']}")
        
        print(f"\n🌟 Available Characters:")
        for char in stats['characters'][:8]:  # Show first 8
            print(f"  - {char}")
        if len(stats['characters']) > 8:
            print(f"  ... and {len(stats['characters']) - 8} more")
            
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        return
    
    # Interactive chat demo
    print("\n5. 🎭 Character Chat Demonstration")
    print("-" * 40)
    
    # Demo conversations
    demo_conversations = [
        {
            "character": "Luke Skywalker",
            "user_message": "How can I learn to use the Force?",
            "context": "Luke learning about the Force"
        },
        {
            "character": "Darth Vader", 
            "user_message": "What makes someone truly powerful?",
            "context": "Vader discussing power and the dark side"
        },
        {
            "character": "Obi-Wan Kenobi",
            "user_message": "I'm feeling lost and need guidance",
            "context": "Obi-Wan offering wisdom and guidance"
        },
        {
            "character": "Princess Leia",
            "user_message": "How do we fight against overwhelming odds?",
            "context": "Leia discussing rebellion and hope"
        }
    ]
    
    for i, demo in enumerate(demo_conversations, 1):
        character = demo["character"]
        user_message = demo["user_message"]
        
        # Check if character is available
        available_chars = app.get_available_characters()
        if character not in available_chars:
            print(f"\n❌ {character} not available in current dataset")
            continue
        
        print(f"\n💬 Demo Chat {i}: {demo['context']}")
        print(f"🎭 Character: {character}")
        print(f"👤 User: {user_message}")
        print(f"🎬 {character}: ", end="")
        
        try:
            start_time = time.time()
            
            # Generate character response
            response = app.chat_with_character(
                user_message=user_message,
                character=character,
                max_context_lines=4,
                temperature=0.7,
                max_tokens=120
            )
            
            response_time = time.time() - start_time
            
            # Display response with typing effect
            response_text = response['response']
            display_typing_effect(response_text, delay=0.02)
            
            # Show metadata
            print(f"📈 Response generated in {response_time:.2f}s")
            print(f"📊 Context lines used: {response['conversation_metadata']['context_lines_used']}")
            
            if 'llm_metadata' in response and response['llm_metadata']:
                llm_meta = response['llm_metadata']
                if 'tokens_per_second' in llm_meta:
                    print(f"⚡ LLM speed: {llm_meta['tokens_per_second']:.1f} tokens/sec")
            
        except Exception as e:
            print(f"❌ Error in chat: {e}")
            continue
        
        time.sleep(1)  # Brief pause between demos
    
    # Test chat quality
    print(f"\n6. 🧪 Chat Quality Assessment")
    print("-" * 30)
    
    try:
        quality_results = app.test_chat_quality([
            "Tell me about hope in dark times",
            "What does it mean to be a Jedi?",
            "How do you face your fears?"
        ])
        
        summary = quality_results['test_summary']
        print(f"📊 Tests completed: {summary['successful_tests']}/{summary['total_tests']}")
        print(f"✅ Success rate: {summary['success_rate']*100:.1f}%")
        print(f"⏱️ Average response time: {summary['average_time_seconds']:.2f}s")
        
        # Show sample results
        print(f"\n🎯 Sample Quality Results:")
        for character, results in list(quality_results['character_results'].items())[:2]:
            print(f"\n🎭 {character}:")
            for result in results[:2]:  # Show first 2 tests
                if result['success']:
                    print(f"  Q: {result['query']}")
                    print(f"  A: {result['response']}")
                    print(f"  ⏱️ {result['time_seconds']}s")
                else:
                    print(f"  ❌ Failed: {result['query']}")
        
    except Exception as e:
        print(f"❌ Quality test failed: {e}")
    
    # Show model information
    print(f"\n7. 🤖 Model Information")
    print("-" * 25)
    
    try:
        model_info = app.list_available_models()
        
        if 'current_model' in model_info:
            current = model_info['current_model']
            print("📱 Current LLM:")
            for key, value in current.items():
                if key != 'model_file_info':  # Skip nested info for brevity
                    print(f"  {key}: {value}")
        
        downloaded = model_info.get('downloaded_models', [])
        if downloaded:
            print(f"\n💾 Downloaded Models: {len(downloaded)}")
            for model in downloaded[:3]:  # Show first 3
                print(f"  - {model['filename']} ({model['size_mb']} MB)")
        
    except Exception as e:
        print(f"❌ Error getting model info: {e}")
    
    # Interactive chat section
    print(f"\n8. 🎮 Interactive Chat (Optional)")
    print("-" * 35)
    print("You can now try chatting with characters yourself!")
    print("Available characters:", ", ".join(app.get_available_characters()[:5]))
    print()
    
    try:
        response = input("Would you like to try interactive chat? (y/n): ").strip().lower()
        
        if response == 'y':
            print("\n🎭 Interactive Star Wars Chat")
            print("Type 'quit' to exit, 'switch <character>' to change character")
            
            current_character = app.get_available_characters()[0]
            print(f"Currently chatting with: {current_character}")
            
            while True:
                try:
                    user_input = input(f"\n👤 You: ").strip()
                    
                    if user_input.lower() == 'quit':
                        break
                    
                    if user_input.lower().startswith('switch '):
                        new_char = user_input[7:].strip()
                        available = app.get_available_characters()
                        
                        # Find matching character
                        matched_char = None
                        for char in available:
                            if new_char.lower() in char.lower():
                                matched_char = char
                                break
                        
                        if matched_char:
                            current_character = matched_char
                            print(f"🎭 Switched to: {current_character}")
                        else:
                            print(f"❌ Character not found. Available: {', '.join(available[:5])}")
                        continue
                    
                    if not user_input:
                        continue
                    
                    print(f"🎬 {current_character}: ", end="")
                    
                    # Generate response
                    response = app.chat_with_character(
                        user_message=user_input,
                        character=current_character,
                        temperature=0.8,
                        max_tokens=100
                    )
                    
                    display_typing_effect(response['response'], delay=0.03)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Chat error: {e}")
                    continue
            
            print("\n👋 Thanks for chatting!")
    
    except KeyboardInterrupt:
        pass
    
    # Summary
    print(f"\n" + "=" * 50)
    print("🎉 Enhanced Chat Demo Completed!")
    print("\nThis demo showcased:")
    print("  ✅ RAG-based context retrieval")
    print("  ✅ Character-specific prompt engineering")
    print("  ✅ Local LLM text generation")
    print("  ✅ Safety filtering and response validation")
    print("  ✅ Real-time chat with Star Wars characters")
    
    print(f"\n🚀 Ready for production deployment:")
    print("  - FastAPI backend integration")
    print("  - Web interface development")
    print("  - PostgreSQL + pgvector database")
    print("  - Docker containerization")
    print("  - DigitalOcean droplet deployment")
    
    print(f"\n📖 Next: Follow instructions for FastAPI and deployment setup!")


if __name__ == "__main__":
    main()
