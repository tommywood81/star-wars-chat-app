"""
Streamlit web interface for Star Wars RAG chat application.

This module provides a complete web interface for chatting with Star Wars characters
using the RAG + LLM system with real-time interactions and history management.
"""

import streamlit as st
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

from star_wars_rag.chat import StarWarsChatApp

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Star Wars RAG Chat",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Star Wars theme
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #FFD700;
        font-size: 3rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        margin-bottom: 2rem;
    }
    
    .character-header {
        color: #4169E1;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    
    .character-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 20%;
    }
    
    .system-stats {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .sidebar-content {
        background: #1e1e1e;
        color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_chat_app():
    """Initialize and cache the chat application."""
    try:
        with st.spinner("🤖 Initializing Star Wars Chat System..."):
            # Initialize with MockLLM for faster web demo
            app = StarWarsChatApp(auto_download=False)
            
            # Load data if available
            data_dir = Path("data/raw")
            if data_dir.exists():
                script_files = list(data_dir.glob("*.txt"))
                if script_files:
                    # Load A New Hope by default
                    default_script = None
                    for script in script_files:
                        if "NEW HOPE" in script.name.upper():
                            default_script = script
                            break
                    
                    if default_script is None:
                        default_script = script_files[0]
                    
                    st.info(f"📖 Loading {default_script.name}...")
                    
                    # Create temp directory for loading
                    import tempfile
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_script_dir = Path(temp_dir) / "scripts"
                        temp_script_dir.mkdir()
                        
                        temp_script = temp_script_dir / default_script.name
                        temp_script.write_text(
                            default_script.read_text(encoding='utf-8'), 
                            encoding='utf-8'
                        )
                        
                        app.load_from_scripts(temp_script_dir, pattern=default_script.name)
                    
                    st.success("✅ Data loaded successfully!")
            
            return app
    except Exception as e:
        st.error(f"❌ Failed to initialize chat system: {e}")
        return None


def display_system_stats(app: StarWarsChatApp):
    """Display system statistics in the sidebar."""
    if not app or not app.is_loaded:
        st.sidebar.warning("⚠️ System not loaded")
        return
    
    try:
        stats = app.get_system_stats()
        llm_info = app.get_llm_info()
        
        st.sidebar.markdown("### 📊 System Status")
        
        # Main stats
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Dialogue Lines", stats.get('total_dialogue_lines', 0))
            st.metric("Characters", stats.get('num_characters', 0))
        
        with col2:
            st.metric("Movies", stats.get('num_movies', 0))
            st.metric("LLM Status", "✅" if llm_info.get('llm_loaded') else "❌")
        
        # Model info
        st.sidebar.markdown("### 🤖 Models")
        st.sidebar.text(f"Embedding: {stats.get('embedding_model', 'Unknown')}")
        st.sidebar.text(f"LLM: {llm_info.get('model_name', 'Mock')}")
        
        # Top characters
        if 'top_characters' in stats:
            st.sidebar.markdown("### 🌟 Top Characters")
            top_chars = list(stats['top_characters'].items())[:5]
            for char, count in top_chars:
                st.sidebar.text(f"{char}: {count} lines")
                
    except Exception as e:
        st.sidebar.error(f"❌ Error loading stats: {e}")


def display_chat_history(messages: List[Dict[str, Any]]):
    """Display chat message history."""
    for message in messages:
        if message["role"] == "user":
            st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong><br>
                    {message["content"]}
                </div>
            """, unsafe_allow_html=True)
        else:
            character = message.get("character", "Character")
            st.markdown(f"""
                <div class="chat-message character-message">
                    <strong>🎭 {character}:</strong><br>
                    {message["content"]}
                </div>
            """, unsafe_allow_html=True)


def display_context_used(context: List[Dict[str, Any]]):
    """Display the context used for response generation."""
    if not context:
        return
    
    with st.expander(f"📖 Context Used ({len(context)} lines)"):
        for i, ctx in enumerate(context, 1):
            character = ctx.get('character', 'Unknown')
            dialogue = ctx.get('dialogue', '')
            movie = ctx.get('movie', '')
            similarity = ctx.get('similarity', 0)
            
            st.markdown(f"""
                **{i}.** {character} ({movie}) - Similarity: {similarity:.3f}
                > *{dialogue[:150]}{"..." if len(dialogue) > 150 else ""}*
            """)


def generate_character_response(app: StarWarsChatApp, 
                              user_message: str, 
                              character: str,
                              conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """Generate character response with progress indication."""
    try:
        with st.spinner(f"🎭 {character} is thinking..."):
            start_time = time.time()
            
            # Generate response
            response = app.chat_with_character(
                user_message=user_message,
                character=character,
                conversation_history=conversation_history,
                temperature=0.8,
                max_tokens=150
            )
            
            response_time = time.time() - start_time
            response["generation_time"] = response_time
            
            return response
            
    except Exception as e:
        st.error(f"❌ Error generating response: {e}")
        return {
            "response": "I'm having trouble responding right now. Perhaps we could talk about something else?",
            "character": character,
            "context_used": [],
            "error": str(e)
        }


def display_character_selector(app: StarWarsChatApp) -> str:
    """Display character selection interface."""
    if not app or not app.is_loaded:
        st.warning("⚠️ Please wait for the system to load...")
        return ""
    
    try:
        available_characters = app.get_available_characters()
        
        if not available_characters:
            st.error("❌ No characters available")
            return ""
        
        # Character selection
        st.markdown("### 🎭 Choose Your Character")
        
        # Create character cards
        cols = st.columns(min(4, len(available_characters)))
        
        # Main character options
        main_characters = ["Luke Skywalker", "Darth Vader", "Obi-Wan Kenobi", "Princess Leia", "Han Solo"]
        available_main = [char for char in main_characters if char in available_characters]
        other_characters = [char for char in available_characters if char not in main_characters]
        
        # Display main characters first
        character_options = available_main + other_characters
        
        selected_character = st.selectbox(
            "Select a character to chat with:",
            character_options,
            index=0 if character_options else None,
            help="Choose which Star Wars character you'd like to have a conversation with"
        )
        
        if selected_character:
            # Display character info
            try:
                stats = app.get_system_stats()
                char_count = stats.get('top_characters', {}).get(selected_character, 0)
                
                st.info(f"🎭 **{selected_character}** has {char_count} dialogue lines available")
                
                # Show sample dialogue
                sample_results = app.retrieve_similar_dialogue(
                    "hello", 
                    character_filter=selected_character, 
                    top_k=2
                )
                
                if sample_results:
                    with st.expander(f"Sample dialogue from {selected_character}"):
                        for result in sample_results:
                            st.markdown(f"> *{result['dialogue']}*")
                            
            except Exception as e:
                st.warning(f"Could not load character info: {e}")
        
        return selected_character
        
    except Exception as e:
        st.error(f"❌ Error in character selection: {e}")
        return ""


def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">🌟 Star Wars RAG Chat 🌟</h1>', unsafe_allow_html=True)
    st.markdown("#### Chat with your favorite Star Wars characters using AI!")
    
    # Initialize app
    app = initialize_chat_app()
    
    # Sidebar with system info
    with st.sidebar:
        st.markdown("## 🚀 System Dashboard")
        display_system_stats(app)
        
        # Settings
        st.markdown("## ⚙️ Chat Settings")
        
        show_context = st.checkbox("Show context used", value=False, 
                                 help="Display the dialogue context used to generate responses")
        
        show_metadata = st.checkbox("Show response metadata", value=False,
                                  help="Display technical details about response generation")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    if not app:
        st.error("❌ Chat system failed to initialize. Please refresh the page.")
        return
    
    # Character selection
    selected_character = display_character_selector(app)
    
    if not selected_character:
        st.warning("⚠️ Please select a character to start chatting!")
        return
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "current_character" not in st.session_state:
        st.session_state.current_character = selected_character
    
    # Check if character changed
    if st.session_state.current_character != selected_character:
        st.session_state.current_character = selected_character
        st.session_state.messages = []  # Clear history when switching characters
        st.success(f"🎭 Switched to {selected_character}. Chat history cleared.")
    
    # Chat interface
    st.markdown(f'<div class="character-header">💬 Chatting with {selected_character}</div>', 
                unsafe_allow_html=True)
    
    # Display chat history
    if st.session_state.messages:
        st.markdown("### 📜 Conversation History")
        display_chat_history(st.session_state.messages)
    else:
        st.markdown(f"### 👋 Start a conversation with {selected_character}!")
        
        # Suggested questions
        suggestions = [
            "Tell me about your greatest challenge",
            "What drives you to keep going?", 
            "Share some wisdom with me",
            "What do you think about the Force?",
            "Tell me about your journey"
        ]
        
        st.markdown("💡 **Suggested questions:**")
        suggestion_cols = st.columns(len(suggestions))
        for i, suggestion in enumerate(suggestions):
            if suggestion_cols[i].button(f"💬 {suggestion}", key=f"suggestion_{i}"):
                # Auto-fill the suggestion
                st.session_state.user_input = suggestion
                st.rerun()
    
    # Chat input
    st.markdown("### ✍️ Your Message")
    
    # Text input with session state
    user_input = st.text_area(
        "Type your message here...",
        value=st.session_state.get("user_input", ""),
        height=100,
        placeholder=f"Ask {selected_character} anything...",
        key="chat_input"
    )
    
    # Send button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        send_button = st.button("🚀 Send Message", type="primary", use_container_width=True)
    
    # Process message
    if send_button and user_input.strip():
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": time.time()
        })
        
        # Prepare conversation history for the model
        conversation_history = []
        recent_messages = st.session_state.messages[-6:]  # Last 6 messages for context
        
        for i in range(0, len(recent_messages), 2):
            if i < len(recent_messages):
                user_msg = recent_messages[i]
                char_msg = recent_messages[i + 1] if i + 1 < len(recent_messages) else None
                
                conv_turn = {"user": user_msg["content"], "character_name": selected_character}
                if char_msg:
                    conv_turn["character"] = char_msg["content"]
                
                conversation_history.append(conv_turn)
        
        # Generate response
        response_data = generate_character_response(
            app, user_input, selected_character, conversation_history
        )
        
        # Add character response to history
        st.session_state.messages.append({
            "role": "character",
            "content": response_data["response"],
            "character": selected_character,
            "timestamp": time.time(),
            "context_used": response_data.get("context_used", []),
            "metadata": response_data.get("conversation_metadata", {})
        })
        
        # Clear input
        st.session_state.user_input = ""
        
        # Show success message
        st.success("✅ Response generated!")
        
        # Display context if enabled
        if show_context and response_data.get("context_used"):
            display_context_used(response_data["context_used"])
        
        # Display metadata if enabled
        if show_metadata and response_data.get("conversation_metadata"):
            with st.expander("🔧 Response Metadata"):
                metadata = response_data["conversation_metadata"]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Context Lines", metadata.get("context_lines_used", 0))
                    st.metric("Retrieval Results", metadata.get("retrieval_results", 0))
                
                with col2:
                    st.metric("Response Time", f"{metadata.get('total_time_seconds', 0):.2f}s")
                    if "generation_time" in response_data:
                        st.metric("Generation Time", f"{response_data['generation_time']:.2f}s")
        
        # Auto-rerun to update the display
        st.rerun()
    
    elif send_button and not user_input.strip():
        st.warning("⚠️ Please enter a message before sending!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            🌟 Star Wars RAG Chat System | Powered by Local LLM & Semantic Search<br>
            <em>May the Force be with you!</em> ⚔️
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
