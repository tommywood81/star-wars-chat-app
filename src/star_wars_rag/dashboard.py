"""
Enhanced SaaS-grade Star Wars RAG Dashboard

This module provides a professional dashboard interface with explainability features,
dark Star Wars theme, analytics, and advanced RAG visualization capabilities.
"""

import streamlit as st
import requests
import time
import uuid
import json
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import logging
from collections import Counter
import pandas as pd

logger = logging.getLogger(__name__)

# Configuration
import os
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8002")  # Environment configurable
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))

# Page configuration
st.set_page_config(
    page_title="Star Wars RAG Dashboard",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Dark Star Wars Theme CSS
st.markdown("""
<style>
    /* Global Dark Theme */
    .stApp {
        background: linear-gradient(180deg, #000000 0%, #0D1117 50%, #1a1a2e 100%);
        color: #ffffff;
    }
    
    .main .block-container {
        padding: 1rem 1rem 10rem;
        max-width: 100%;
    }
    
    /* Header Styling */
    .main-header {
        text-align: center;
        background: linear-gradient(45deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: bold;
        text-shadow: 0 0 20px rgba(255, 215, 0, 0.5);
        margin-bottom: 2rem;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 20px rgba(255, 215, 0, 0.5); }
        to { text-shadow: 0 0 30px rgba(255, 215, 0, 0.8); }
    }
    
    /* Character Persona Header */
    .character-persona {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border: 2px solid #4169E1;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(65, 105, 225, 0.3);
    }
    
    .character-avatar {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        border: 3px solid #FFD700;
        background: linear-gradient(45deg, #667eea, #764ba2);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        margin-right: 1rem;
    }
    
    /* Chat Interface */
    .chat-container {
        background: rgba(13, 17, 23, 0.8);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid #30363d;
        backdrop-filter: blur(10px);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0 0.5rem 20%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .character-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 20% 0.5rem 0;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
    }
    
    /* Explainability Panels */
    .explainability-panel {
        background: rgba(48, 54, 61, 0.6);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        backdrop-filter: blur(5px);
    }
    
    .context-line {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #4169E1;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .context-line.highlighted {
        border-left-color: #FFD700;
        background: rgba(255, 215, 0, 0.1);
    }
    
    /* Pipeline Visualization */
    .pipeline-step {
        background: rgba(13, 17, 23, 0.9);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .pipeline-step.active {
        border-color: #FFD700;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.3);
    }
    
    .pipeline-step.complete {
        border-color: #28a745;
        background: rgba(40, 167, 69, 0.1);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #0d1117 0%, #1a1a2e 100%);
    }
    
    /* Metrics and Stats */
    .metric-card {
        background: rgba(13, 17, 23, 0.8);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    /* Character Selector */
    .character-card {
        background: rgba(13, 17, 23, 0.8);
        border: 2px solid #30363d;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .character-card:hover {
        border-color: #4169E1;
        box-shadow: 0 8px 25px rgba(65, 105, 225, 0.3);
        transform: translateY(-2px);
    }
    
    .character-card.selected {
        border-color: #FFD700;
        background: rgba(255, 215, 0, 0.1);
    }
    
    /* Animations for Loading */
    .loading-pulse {
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Search Bar */
    .search-container {
        background: rgba(13, 17, 23, 0.8);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=10)
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


@st.cache_data(ttl=30)
def get_system_info():
    """Get system information from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/system/info", timeout=API_TIMEOUT)
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        logger.error(f"System info failed: {e}")
        return {}


@st.cache_data(ttl=60)
def get_characters():
    """Get available characters from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/characters", timeout=API_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            return data.get("characters", [])
        return []
    except Exception as e:
        logger.error(f"Characters fetch failed: {e}")
        return []


def chat_with_api(character: str, message: str, session_id: str) -> Dict[str, Any]:
    """Send chat request to API."""
    try:
        payload = {
            "character": character,
            "message": message,
            "session_id": session_id,
            "temperature": 0.8,
            "max_tokens": 150
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
                "response": "I'm having trouble responding right now.",
                "error": f"API error: {response.status_code}"
            }
    except Exception as e:
        logger.error(f"Chat API error: {e}")
        return {
            "response": "I'm experiencing technical difficulties.",
            "error": str(e)
        }


def display_character_persona_header(character: str, character_data: Dict):
    """Display enhanced character persona header with avatar and traits."""
    dialogue_count = character_data.get('dialogue_count', 0)
    movies = character_data.get('movies', [])
    sample_dialogue = character_data.get('sample_dialogue', [])
    
    # Character emoji mapping
    character_emojis = {
        "Luke Skywalker": "👨‍🚀",
        "Darth Vader": "🎭",
        "Princess Leia": "👸",
        "Han Solo": "🤵",
        "Obi-Wan Kenobi": "🧙‍♂️",
        "Yoda": "🐸",
        "C-3PO": "🤖",
        "R2-D2": "🤖",
        "Chewbacca": "🐻",
        "Lando Calrissian": "🎩",
        "Emperor Palpatine": "👑",
        "Boba Fett": "🚀"
    }
    
    emoji = character_emojis.get(character, "🎭")
    
    st.markdown(f"""
    <div class="character-persona">
        <div style="display: flex; align-items: center;">
            <div class="character-avatar">{emoji}</div>
            <div>
                <h2 style="margin: 0; color: #FFD700;">{character}</h2>
                <p style="margin: 0.5rem 0; color: #b3b3b3;">
                    {dialogue_count} dialogue lines across {len(movies)} movies
                </p>
                <div style="display: flex; gap: 1rem; margin-top: 0.5rem;">
                    <span style="background: rgba(65, 105, 225, 0.2); padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.8rem;">
                        📚 {len(movies)} Films
                    </span>
                    <span style="background: rgba(255, 215, 0, 0.2); padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.8rem;">
                        💬 {dialogue_count} Lines
                    </span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_pipeline_visualization(step: str = "idle"):
    """Display animated RAG pipeline visualization."""
    steps = [
        ("User Query", "📝"),
        ("Embedding", "🔮"),
        ("Vector Search", "🔍"),
        ("Retrieved Chunks", "📊"),
        ("LLM Generation", "🤖"),
        ("Response", "💬")
    ]
    
    cols = st.columns(len(steps))
    
    for i, (step_name, icon) in enumerate(steps):
        with cols[i]:
            if step == "idle":
                class_name = "pipeline-step"
            elif i < len(steps) - 1:
                class_name = "pipeline-step complete"
            elif step_name.lower().replace(" ", "_") == step:
                class_name = "pipeline-step active"
            else:
                class_name = "pipeline-step"
            
            st.markdown(f"""
            <div class="{class_name}">
                <div style="font-size: 2rem;">{icon}</div>
                <div style="font-size: 0.9rem; margin-top: 0.5rem;">{step_name}</div>
            </div>
            """, unsafe_allow_html=True)


def display_retrieved_context_panel(context_used: List[Dict], show_highlights: bool = True):
    """Display retrieved script lines with explainability."""
    if not context_used:
        return
    
    st.markdown("### 📖 Retrieved Script Context")
    
    with st.expander(f"View {len(context_used)} retrieved dialogue lines", expanded=False):
        for i, ctx in enumerate(context_used, 1):
            character = ctx.get('character', 'Unknown')
            dialogue = ctx.get('dialogue', '')
            movie = ctx.get('movie', 'Unknown')
            similarity = ctx.get('similarity', 0)
            
            # Determine if this line was likely used (high similarity)
            highlight_class = "highlighted" if similarity > 0.4 else ""
            
            st.markdown(f"""
            <div class="context-line {highlight_class}">
                <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 0.5rem;">
                    <strong>#{i} {character}</strong>
                    <div style="margin-left: auto;">
                        <span style="background: rgba(255,255,255,0.1); padding: 0.2rem 0.5rem; border-radius: 10px; font-size: 0.8rem;">
                            📺 {movie}
                        </span>
                        <span style="background: rgba(65, 105, 225, 0.2); padding: 0.2rem 0.5rem; border-radius: 10px; font-size: 0.8rem; margin-left: 0.5rem;">
                            🎯 {similarity:.3f}
                        </span>
                    </div>
                </div>
                <div style="font-style: italic; color: #e6e6e6;">
                    "{dialogue}"
                </div>
            </div>
            """, unsafe_allow_html=True)


def display_llm_info_panel(metadata: Dict):
    """Display LLM processing information and metrics."""
    st.markdown("### 🤖 Model Processing Info")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Retrieval Results",
            metadata.get('retrieval_results', 0),
            help="Number of similar dialogue lines found"
        )
    
    with col2:
        st.metric(
            "Context Lines Used", 
            metadata.get('context_lines_used', 0),
            help="Lines sent to LLM for context"
        )
    
    with col3:
        total_time = metadata.get('total_time_seconds', 0)
        st.metric(
            "Total Time",
            f"{total_time:.2f}s",
            help="End-to-end response time"
        )
    
    with col4:
        prompt_length = metadata.get('prompt_length', 0)
        st.metric(
            "Prompt Length",
            f"{prompt_length} chars",
            help="Total prompt size sent to LLM"
        )


def display_character_analytics():
    """Display character analytics with charts."""
    st.markdown("### 📊 Character Analytics")
    
    characters = get_characters()
    if not characters:
        st.warning("No character data available")
        return
    
    # Prepare data
    char_names = []
    dialogue_counts = []
    movies_data = []
    
    for char in characters:
        if isinstance(char, dict) and char.get('dialogue_count', 0) > 0:
            char_names.append(char['name'])
            dialogue_counts.append(char['dialogue_count'])
            movies_data.extend(char.get('movies', []))
    
    if not char_names:
        st.info("Analytics will appear when characters have dialogue data")
        return
    
    col1, col2 = st.columns(2)
    
    # Top characters pie chart
    with col1:
        st.markdown("#### 🎭 Top Characters by Dialogue")
        
        # Get top 10 characters
        top_chars = list(zip(char_names, dialogue_counts))
        top_chars.sort(key=lambda x: x[1], reverse=True)
        top_chars = top_chars[:10]
        
        if top_chars:
            fig = px.pie(
                values=[count for _, count in top_chars],
                names=[name for name, _ in top_chars],
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Movies distribution
    with col2:
        st.markdown("#### 📺 Content by Movie")
        
        movie_counts = Counter(movies_data)
        if movie_counts:
            fig = px.bar(
                x=list(movie_counts.keys()),
                y=list(movie_counts.values()),
                color_discrete_sequence=['#FFD700', '#4169E1', '#DC143C']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                xaxis_title="Movie",
                yaxis_title="Character Appearances"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)


def display_session_history_sidebar():
    """Display session history in sidebar."""
    st.sidebar.markdown("## 📚 Session History")
    
    if "session_history" not in st.session_state:
        st.session_state.session_history = []
    
    if st.session_state.session_history:
        for i, session in enumerate(reversed(st.session_state.session_history[-10:])):
            with st.sidebar.expander(f"Session {len(st.session_state.session_history) - i}"):
                st.markdown(f"**Character:** {session.get('character', 'Unknown')}")
                st.markdown(f"**Messages:** {session.get('message_count', 0)}")
                st.markdown(f"**Time:** {session.get('timestamp', '')}")
                
                if st.button(f"Export Session {len(st.session_state.session_history) - i}", key=f"export_{i}"):
                    # Export functionality would go here
                    st.success("Export feature coming soon!")
    else:
        st.sidebar.info("No session history yet")
    
    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.session_history = []
        st.rerun()


def main():
    """Main dashboard application."""
    # Header with Star Wars styling
    st.markdown('<h1 class="main-header">⭐ STAR WARS RAG DASHBOARD ⭐</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #b3b3b3;">
        Professional-grade RAG system with explainable AI and real-time analytics
    </div>
    """, unsafe_allow_html=True)
    
    # Check API health
    health = get_api_health()
    system_info = get_system_info()
    
    if health.get("status") != "healthy":
        st.error("🚨 API is not responding. Please ensure the backend is running.")
        st.stop()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_character" not in st.session_state:
        st.session_state.current_character = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Sidebar with system stats and history
    with st.sidebar:
        st.markdown("## 🚀 System Dashboard")
        
        # System health indicators
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Dialogue Lines", system_info.get('dialogue_lines', 0))
            st.metric("🎭 Characters", system_info.get('characters_count', 0))
        
        with col2:
            st.metric("🎬 Movies", system_info.get('movies_count', 0))
            status_emoji = "✅" if health.get("models_loaded") else "❌"
            st.metric("🤖 LLM Status", status_emoji)
        
        # Model information
        st.markdown("### 🔧 Model Info")
        st.text(f"Embedding: {system_info.get('embedding_model', 'Unknown')}")
        st.text(f"LLM: {system_info.get('llm_model', 'Unknown')}")
        
        # Session history
        display_session_history_sidebar()
        
        # Settings
        st.markdown("## ⚙️ Dashboard Settings")
        show_explainability = st.checkbox("Show Explainability Panels", value=True)
        show_pipeline = st.checkbox("Show Processing Pipeline", value=True)
        show_analytics = st.checkbox("Show Character Analytics", value=False)
    
    # Main layout: Character selector and chat
    char_col, chat_col = st.columns([1, 2])
    
    # Character Selection Panel
    with char_col:
        st.markdown("### 🎭 Character Selection")
        
        characters = get_characters()
        if not characters:
            st.error("No characters available")
            st.stop()
        
        # Get top 20 characters by dialogue count
        top_characters = [char for char in characters if isinstance(char, dict) and char.get('dialogue_count', 0) > 0]
        top_characters.sort(key=lambda x: x.get('dialogue_count', 0), reverse=True)
        top_characters = top_characters[:20]
        
        character_names = [char['name'] for char in top_characters]
        
        # Character dropdown
        selected_character = st.selectbox(
            "Choose your character:",
            character_names,
            index=0 if character_names else None,
            help="Select from the top 20 characters with the most dialogue"
        )
        
        if selected_character and selected_character != st.session_state.current_character:
            st.session_state.current_character = selected_character
            st.session_state.messages = []  # Clear when switching
            st.success(f"Switched to {selected_character}")
        
        # Display character info
        if selected_character:
            character_data = next((char for char in top_characters if char['name'] == selected_character), {})
            display_character_persona_header(selected_character, character_data)
    
    # Main Chat Interface
    with chat_col:
        if not selected_character:
            st.info("👈 Please select a character to start chatting")
            return
        
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Pipeline visualization
        if show_pipeline:
            display_pipeline_visualization("idle")
        
        # Chat messages
        if st.session_state.messages:
            st.markdown("### 💬 Conversation")
            
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="user-message">
                        <strong>👤 You:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    character = message.get("character", "Character")
                    st.markdown(f"""
                    <div class="character-message">
                        <strong>🎭 {character}:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show explainability panels for the last message
                    if (show_explainability and 
                        message == st.session_state.messages[-1] and 
                        message.get("context_used")):
                        
                        with st.expander("🔍 Explainability Details", expanded=False):
                            # Retrieved context
                            display_retrieved_context_panel(message["context_used"])
                            
                            # LLM info
                            if message.get("metadata"):
                                display_llm_info_panel(message["metadata"])
        else:
            st.markdown(f"### 👋 Start a conversation with {selected_character}!")
            
            # Suggested questions
            suggestions = [
                "Tell me about your greatest challenge",
                "What motivates you?",
                "Share wisdom with me",
                "Tell me about the Force",
                "What's your story?"
            ]
            
            st.markdown("💡 **Quick starts:**")
            suggestion_cols = st.columns(len(suggestions))
            for i, suggestion in enumerate(suggestions):
                if suggestion_cols[i].button(f"💭 {suggestion}", key=f"suggestion_{i}"):
                    st.session_state.user_input = suggestion
                    st.rerun()
        
        # Chat input
        st.markdown("### ✍️ Your Message")
        user_input = st.text_area(
            "Type your message:",
            value=st.session_state.get("user_input", ""),
            height=100,
            placeholder=f"Ask {selected_character} anything...",
            key="chat_input"
        )
        
        # Send button
        if st.button("🚀 Send Message", type="primary", use_container_width=True):
            if user_input.strip():
                # Add user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": time.time()
                })
                
                # Show pipeline in action
                if show_pipeline:
                    with st.spinner("Processing..."):
                        for step in ["embedding", "vector_search", "llm_generation"]:
                            display_pipeline_visualization(step)
                            time.sleep(0.5)
                
                # Generate response
                with st.spinner(f"🎭 {selected_character} is thinking..."):
                    response_data = chat_with_api(
                        selected_character, 
                        user_input, 
                        st.session_state.session_id
                    )
                
                # Add character response
                st.session_state.messages.append({
                    "role": "character",
                    "content": response_data.get("response", "I couldn't respond."),
                    "character": selected_character,
                    "timestamp": time.time(),
                    "context_used": response_data.get("context_used", []),
                    "metadata": response_data.get("metadata", {})
                })
                
                # Clear input
                st.session_state.user_input = ""
                st.success("✅ Response generated!")
                st.rerun()
            else:
                st.warning("Please enter a message!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Character Analytics (bottom panel)
    if show_analytics:
        st.markdown("---")
        display_character_analytics()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        ⭐ <strong>Star Wars RAG Dashboard</strong> | Professional-grade AI system with explainable retrieval<br>
        <em style="color: #FFD700;">May the Force be with you!</em> ⚔️
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
