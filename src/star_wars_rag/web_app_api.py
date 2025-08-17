"""
Streamlit web interface for Star Wars RAG chat application using API client.

This module provides a web interface that communicates with the FastAPI backend
instead of running its own LLM, making it much more efficient and scalable.
"""

import streamlit as st
import requests
import time
import uuid
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://star-wars-api:8000"  # Internal Docker network
API_TIMEOUT = 30

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
    
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=10)  # Cache for 10 seconds
def get_api_health():
    """Get API health status."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "models_loaded": False, "database_connected": False, "data_loaded": False}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "error", "models_loaded": False, "database_connected": False, "data_loaded": False}


@st.cache_data(ttl=60)  # Cache for 1 minute
def get_available_characters():
    """Get list of available characters from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/characters", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except Exception as e:
        logger.error(f"Failed to get characters: {e}")
        return []


def display_system_stats():
    """Display system statistics in the sidebar."""
    health = get_api_health()
    
    st.markdown("### 🎯 System Status")
    
    # Overall status
    if health["status"] == "healthy":
        st.markdown('**Status:** <span class="status-healthy">🟢 Healthy</span>', unsafe_allow_html=True)
    else:
        st.markdown('**Status:** <span class="status-error">🔴 Error</span>', unsafe_allow_html=True)
    
    # Component status
    st.markdown("#### 📊 Components")
    
    # LLM Status
    if health.get("models_loaded", False):
        st.markdown("**🤖 LLM:** <span class='status-healthy'>✅ Loaded</span>", unsafe_allow_html=True)
    else:
        st.markdown("**🤖 LLM:** <span class='status-error'>❌ Not Loaded</span>", unsafe_allow_html=True)
    
    # Database Status
    if health.get("database_connected", False):
        st.markdown("**🗄️ Database:** <span class='status-healthy'>✅ Connected</span>", unsafe_allow_html=True)
    else:
        st.markdown("**🗄️ Database:** <span class='status-error'>❌ Disconnected</span>", unsafe_allow_html=True)
    
    # Data Status
    if health.get("data_loaded", False):
        st.markdown("**📚 Data:** <span class='status-healthy'>✅ Loaded</span>", unsafe_allow_html=True)
    else:
        st.markdown("**📚 Data:** <span class='status-error'>❌ Not Loaded</span>", unsafe_allow_html=True)


def display_character_selector() -> Optional[str]:
    """Display character selection interface."""
    characters = get_available_characters()
    
    if not characters:
        st.error("❌ Could not load characters. Please check API connection.")
        return None
    
    st.markdown('<h2 class="character-header">🎭 Select Your Character</h2>', unsafe_allow_html=True)
    
    # Character selection
    character_names = [char.get("name", "Unknown") for char in characters]
    
    selected_name = st.selectbox(
        "Choose a character to chat with:",
        character_names,
        index=0 if character_names else None,
        key="character_selector"
    )
    
    if selected_name:
        # Find the selected character data
        selected_char = next((char for char in characters if char.get("name") == selected_name), None)
        
        if selected_char:
            # Display character info
            with st.expander(f"ℹ️ About {selected_name}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("📝 Dialogue Lines", selected_char.get("dialogue_count", 0))
                
                with col2:
                    st.metric("🎬 Appearances", selected_char.get("appearances", "Unknown"))
                
                # Sample dialogue
                if selected_char.get("sample_dialogue"):
                    st.markdown("**💬 Sample Dialogue:**")
                    st.info(f'"{selected_char["sample_dialogue"]}"')
        
        return selected_name
    
    return None


def chat_with_api(character: str, message: str, session_id: str) -> Dict[str, Any]:
    """Send chat request to API."""
    try:
        payload = {
            "character": character,
            "message": message,
            "session_id": session_id,
            "max_tokens": 150,
            "temperature": 0.8
        }
        
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            timeout=API_TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "response": "I'm having trouble responding right now. The API returned an error.",
                "character": character,
                "error": f"API returned status {response.status_code}"
            }
            
    except requests.exceptions.Timeout:
        return {
            "response": "I'm taking too long to respond. Please try again.",
            "character": character,
            "error": "Request timeout"
        }
    except Exception as e:
        logger.error(f"Chat API error: {e}")
        return {
            "response": "I'm having trouble responding right now. Perhaps we could talk about something else?",
            "character": character,
            "error": str(e)
        }


def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">🌟 Star Wars RAG Chat 🌟</h1>', unsafe_allow_html=True)
    st.markdown("#### Chat with your favorite Star Wars characters using AI!")
    
    # Initialize session ID
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Sidebar with system info
    with st.sidebar:
        st.markdown("## 🚀 System Dashboard")
        display_system_stats()
        
        # Settings
        st.markdown("## ⚙️ Chat Settings")
        
        show_metadata = st.checkbox("Show response metadata", value=False,
                                  help="Display technical details about response generation")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())  # New session
            st.rerun()
        
        # Connection info
        st.markdown("## 🔗 Connection Info")
        st.markdown(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
    
    # Check API health
    health = get_api_health()
    if health["status"] != "healthy":
        st.error("❌ API is not healthy. Please check the backend service.")
        st.stop()
    
    # Character selection
    selected_character = display_character_selector()
    
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
        st.session_state.session_id = str(uuid.uuid4())  # New session
        st.success(f"🎭 Switched to {selected_character}. Chat history cleared.")
    
    # Display chat history
    st.markdown(f'<h2 class="character-header">💬 Chat with {selected_character}</h2>', unsafe_allow_html=True)
    
    # Chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🧑 You:</strong> {message["content"]}
                </div>
            """, unsafe_allow_html=True)
        else:
            character_name = message.get("character", selected_character)
            st.markdown(f"""
                <div class="chat-message character-message">
                    <strong>🎭 {character_name}:</strong> {message["content"]}
                </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    st.markdown("### 💬 Send a Message")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your message:",
            key="user_input",
            placeholder=f"Ask {selected_character} anything about Star Wars...",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("🚀 Send", type="primary", use_container_width=True)
    
    # Process message
    if send_button and user_input.strip():
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": time.time()
        })
        
        # Generate response
        with st.spinner(f"🎭 {selected_character} is thinking..."):
            response_data = chat_with_api(selected_character, user_input, st.session_state.session_id)
        
        # Add character response to history
        st.session_state.messages.append({
            "role": "character",
            "content": response_data["response"],
            "character": selected_character,
            "timestamp": time.time(),
            "metadata": response_data
        })
        
        # Show success message
        if "error" not in response_data:
            st.success("✅ Response generated!")
        else:
            st.warning(f"⚠️ Response generated with issues: {response_data['error']}")
        
        # Display metadata if enabled
        if show_metadata and response_data:
            with st.expander("🔧 Response Metadata"):
                st.json(response_data)
        
        # Clear input and rerun
        st.session_state.user_input = ""
        st.rerun()
    
    elif send_button and not user_input.strip():
        st.warning("⚠️ Please enter a message before sending!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            🌟 Star Wars RAG Chat System | Powered by FastAPI & Local LLM<br>
            <em>May the Force be with you!</em> ⚔️
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
