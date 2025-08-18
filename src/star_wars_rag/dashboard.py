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
    page_title="Millennium Falcon Control Interface",
    page_icon="⬛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Millennium Falcon Cockpit Theme CSS
st.markdown("""
<style>
    /* Millennium Falcon Cockpit Theme */
    .stApp {
        background: radial-gradient(ellipse at center, #1a1a1a 0%, #0a0a0a 50%, #000000 100%);
        color: #e0e0e0;
        font-family: 'Courier New', monospace;
    }
    
    .main .block-container {
        padding: 1rem 1rem 10rem;
        max-width: 100%;
    }
    
    /* Cockpit Header */
    .cockpit-header {
        text-align: center;
        color: #00ff41;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 0 0 15px #00ff41;
        margin-bottom: 2rem;
        letter-spacing: 3px;
        border: 2px solid #333333;
        border-radius: 20px;
        padding: 1rem;
        background: linear-gradient(45deg, #1a1a1a, #2a2a2a);
        box-shadow: inset 0 0 20px rgba(0, 255, 65, 0.1);
    }
    
    /* Control Panel */
    .control-panel {
        background: linear-gradient(145deg, #2a2a2a, #1a1a1a);
        border: 2px solid #555555;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.5), 0 2px 10px rgba(255, 255, 255, 0.1);
    }
    
    /* Character Status Display */
    .character-status {
        background: linear-gradient(145deg, #333333, #1a1a1a);
        border: 2px solid #666666;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        position: relative;
    }
    
    .character-status::before {
        content: '';
        position: absolute;
        top: 10px;
        right: 10px;
        width: 12px;
        height: 12px;
        background: #00ff41;
        border-radius: 50%;
        box-shadow: 0 0 10px #00ff41;
        animation: statusBlink 2s infinite;
    }
    
    @keyframes statusBlink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0.3; }
    }
    
    .character-avatar {
        width: 70px;
        height: 70px;
        border-radius: 8px;
        border: 2px solid #00ff41;
        background: linear-gradient(45deg, #333333, #555555);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin-right: 1rem;
        color: #00ff41;
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.3);
    }
    
    /* Communication Interface */
    .comm-interface {
        background: linear-gradient(145deg, #1a1a1a, #0a0a0a);
        border: 2px solid #444444;
        border-radius: 8px;
        padding: 1.5rem;
        border-left: 4px solid #00ff41;
    }
    
    .user-transmission {
        background: linear-gradient(135deg, #2a4a2a, #1a3a1a);
        color: #e0e0e0;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0 0.5rem 20%;
        border-left: 4px solid #00ff41;
        box-shadow: 0 2px 8px rgba(0, 255, 65, 0.2);
    }
    
    .character-transmission {
        background: linear-gradient(135deg, #4a2a2a, #3a1a1a);
        color: #e0e0e0;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 20% 0.5rem 0;
        border-left: 4px solid #ff6b6b;
        box-shadow: 0 2px 8px rgba(255, 107, 107, 0.2);
    }
    
    /* Data Readouts */
    .data-readout {
        background: rgba(0, 0, 0, 0.7);
        border: 1px solid #555555;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
    }
    
    .data-line {
        background: rgba(0, 0, 0, 0.5);
        border-left: 3px solid #00ff41;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0 5px 5px 0;
        font-family: 'Courier New', monospace;
    }
    
    .data-line.priority {
        border-left-color: #ff6b6b;
        background: rgba(255, 107, 107, 0.1);
    }
    
    /* System Status Indicators */
    .system-indicator {
        background: linear-gradient(145deg, #1a1a1a, #0a0a0a);
        border: 1px solid #444444;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        transition: all 0.3s ease;
        font-family: 'Courier New', monospace;
    }
    
    .system-indicator.active {
        border-color: #00ff41;
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.3);
    }
    
    .system-indicator.complete {
        border-color: #ff6b6b;
        background: rgba(255, 107, 107, 0.1);
    }
    
    /* Sidebar Control Panel */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a1a 0%, #0a0a0a 100%);
        border-right: 2px solid #333333;
    }
    
    /* Status Displays */
    .status-display {
        background: linear-gradient(145deg, #1a1a1a, #0a0a0a);
        border: 1px solid #444444;
        border-radius: 5px;
        padding: 1rem;
        text-align: center;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.5);
        font-family: 'Courier New', monospace;
    }
    
    /* Navigation Controls */
    .nav-control {
        background: linear-gradient(145deg, #1a1a1a, #0a0a0a);
        border: 2px solid #333333;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        font-family: 'Courier New', monospace;
    }
    
    .nav-control:hover {
        border-color: #00ff41;
        box-shadow: 0 5px 15px rgba(0, 255, 65, 0.3);
        transform: translateY(-1px);
    }
    
    .nav-control.selected {
        border-color: #ff6b6b;
        background: rgba(255, 107, 107, 0.1);
    }
    
    /* Terminal Loading */
    .terminal-loading {
        animation: terminalPulse 1.5s ease-in-out infinite;
    }
    
    @keyframes terminalPulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Input Terminal */
    .input-terminal {
        background: linear-gradient(145deg, #0a0a0a, #1a1a1a);
        border: 1px solid #333333;
        border-radius: 5px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
    }
    
    /* Control Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #333333, #555555);
        color: #00ff41;
        border: 2px solid #00ff41;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
        font-family: 'Courier New', monospace;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        box-shadow: 0 5px 15px rgba(0, 255, 65, 0.4);
        transform: translateY(-2px);
        background: linear-gradient(45deg, #555555, #777777);
    }
    
    /* Remove all scrollbars styling for cleaner cockpit look */
    ::-webkit-scrollbar {
        width: 8px;
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #333333;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555555;
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
    """Display character status display with cockpit-style readout."""
    dialogue_count = character_data.get('dialogue_count', 0)
    movies = character_data.get('movies', [])
    sample_dialogue = character_data.get('sample_dialogue', [])
    
    # Character code mapping for cockpit display
    character_codes = {
        "Luke Skywalker": "LSW",
        "Darth Vader": "DVR", 
        "Princess Leia": "PLO",
        "Han Solo": "HSO",
        "Obi-Wan Kenobi": "OWK",
        "Yoda": "YDA",
        "C-3PO": "C3P",
        "R2-D2": "R2D",
        "Chewbacca": "CHW",
        "Lando Calrissian": "LCR",
        "Emperor Palpatine": "EMP",
        "Boba Fett": "BFT"
    }
    
    char_code = character_codes.get(character, "UNK")
    
    st.markdown(f"""
    <div class="character-status">
        <div style="display: flex; align-items: center;">
            <div class="character-avatar">{char_code}</div>
            <div>
                <h2 style="margin: 0; color: #00ff41; font-family: 'Courier New', monospace;">{character}</h2>
                <p style="margin: 0.5rem 0; color: #cccccc; font-family: 'Courier New', monospace;">
                    STATUS: ACTIVE | LINES: {dialogue_count} | FILMS: {len(movies)}
                </p>
                <div style="display: flex; gap: 1rem; margin-top: 0.5rem;">
                    <span style="background: rgba(0, 255, 65, 0.2); padding: 0.3rem 0.8rem; border-radius: 3px; font-size: 0.8rem; font-family: 'Courier New', monospace;">
                        FILMS: {len(movies)}
                    </span>
                    <span style="background: rgba(255, 107, 107, 0.2); padding: 0.3rem 0.8rem; border-radius: 3px; font-size: 0.8rem; font-family: 'Courier New', monospace;">
                        DIALOGUE: {dialogue_count}
                    </span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_pipeline_visualization(step: str = "idle"):
    """Display system process monitoring with cockpit-style indicators."""
    steps = [
        ("USER_INPUT", "01"),
        ("EMBEDDING", "02"),
        ("VECTOR_SCAN", "03"),
        ("DATA_RETRIEVAL", "04"),
        ("LLM_PROCESS", "05"),
        ("OUTPUT", "06")
    ]
    
    cols = st.columns(len(steps))
    
    for i, (step_name, code) in enumerate(steps):
        with cols[i]:
            if step == "idle":
                class_name = "system-indicator"
            elif i < len(steps) - 1:
                class_name = "system-indicator complete"
            elif step_name.lower().replace("_", "_") == step:
                class_name = "system-indicator active"
            else:
                class_name = "system-indicator"
            
            st.markdown(f"""
            <div class="{class_name}">
                <div style="font-size: 1.5rem; color: #00ff41; font-family: 'Courier New', monospace;">{code}</div>
                <div style="font-size: 0.8rem; margin-top: 0.5rem; font-family: 'Courier New', monospace;">{step_name}</div>
            </div>
            """, unsafe_allow_html=True)


def display_retrieved_context_panel(context_used: List[Dict], show_highlights: bool = True):
    """Display retrieved script data with cockpit-style readout."""
    if not context_used:
        return
    
    st.markdown("### DATA_RETRIEVAL: Script Context", unsafe_allow_html=True)
    
    with st.expander(f"VIEW {len(context_used)} RETRIEVED DIALOGUE ENTRIES", expanded=False):
        for i, ctx in enumerate(context_used, 1):
            character = ctx.get('character', 'Unknown')
            dialogue = ctx.get('dialogue', '')
            movie = ctx.get('movie', 'Unknown')
            similarity = ctx.get('similarity', 0)
            
            # Determine if this line was priority data (high similarity)
            highlight_class = "priority" if similarity > 0.4 else ""
            
            st.markdown(f"""
            <div class="data-line {highlight_class}">
                <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 0.5rem;">
                    <strong style="font-family: 'Courier New', monospace; color: #00ff41;">#{i:02d} {character}</strong>
                    <div style="margin-left: auto;">
                        <span style="background: rgba(255,255,255,0.1); padding: 0.2rem 0.5rem; border-radius: 3px; font-size: 0.8rem; font-family: 'Courier New', monospace;">
                            SRC: {movie}
                        </span>
                        <span style="background: rgba(0, 255, 65, 0.2); padding: 0.2rem 0.5rem; border-radius: 3px; font-size: 0.8rem; margin-left: 0.5rem; font-family: 'Courier New', monospace;">
                            SIM: {similarity:.3f}
                        </span>
                    </div>
                </div>
                <div style="font-style: italic; color: #cccccc; font-family: 'Courier New', monospace;">
                    "{dialogue}"
                </div>
            </div>
            """, unsafe_allow_html=True)


def display_llm_info_panel(metadata: Dict):
    """Display LLM processing telemetry with system readouts."""
    st.markdown("### LLM_PROCESS: System Telemetry")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "RETRIEVAL_COUNT",
            metadata.get('retrieval_results', 0),
            help="Number of similar dialogue lines found"
        )
    
    with col2:
        st.metric(
            "CONTEXT_LINES", 
            metadata.get('context_lines_used', 0),
            help="Lines sent to LLM for context"
        )
    
    with col3:
        total_time = metadata.get('total_time_seconds', 0)
        st.metric(
            "PROC_TIME",
            f"{total_time:.2f}s",
            help="End-to-end response time"
        )
    
    with col4:
        prompt_length = metadata.get('prompt_length', 0)
        st.metric(
            "PROMPT_SIZE",
            f"{prompt_length} chars",
            help="Total prompt size sent to LLM"
        )


def display_character_analytics():
    """Display character analytics with tactical readouts."""
    st.markdown("### TACTICAL_ANALYTICS: Character Data")
    
    characters = get_characters()
    if not characters:
        st.warning("WARNING: No character data available")
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
        st.info("STANDBY: Analytics will appear when characters have dialogue data")
        return
    
    col1, col2 = st.columns(2)
    
    # Top characters distribution
    with col1:
        st.markdown("#### DIALOGUE_DISTRIBUTION: Top Characters")
        
        # Get top 10 characters
        top_chars = list(zip(char_names, dialogue_counts))
        top_chars.sort(key=lambda x: x[1], reverse=True)
        top_chars = top_chars[:10]
        
        if top_chars:
            fig = px.pie(
                values=[count for _, count in top_chars],
                names=[name for name, _ in top_chars],
                color_discrete_sequence=['#00ff41', '#ff6b6b', '#ffff41', '#41a7ff', '#ff41ff', '#41ffff', '#ff8c41', '#8c41ff', '#41ff8c', '#ffc741']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e0e0e0',
                font_family='Courier New'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Movies distribution
    with col2:
        st.markdown("#### SOURCE_ANALYSIS: Content by Film")
        
        movie_counts = Counter(movies_data)
        if movie_counts:
            fig = px.bar(
                x=list(movie_counts.keys()),
                y=list(movie_counts.values()),
                color_discrete_sequence=['#00ff41', '#ff6b6b', '#ffff41']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e0e0e0',
                font_family='Courier New',
                xaxis_title="FILM_SOURCE",
                yaxis_title="CHARACTER_COUNT"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)


def display_session_history_sidebar():
    """Display session logs in control panel."""
    st.sidebar.markdown("## SESSION_LOGS: Communication History")
    
    if "session_history" not in st.session_state:
        st.session_state.session_history = []
    
    if st.session_state.session_history:
        for i, session in enumerate(reversed(st.session_state.session_history[-10:])):
            with st.sidebar.expander(f"LOG {len(st.session_state.session_history) - i:03d}"):
                st.markdown(f"**CHARACTER:** {session.get('character', 'Unknown')}")
                st.markdown(f"**MESSAGES:** {session.get('message_count', 0)}")
                st.markdown(f"**TIMESTAMP:** {session.get('timestamp', '')}")
                
                if st.button(f"EXPORT LOG {len(st.session_state.session_history) - i:03d}", key=f"export_{i}"):
                    # Export functionality would go here
                    st.success("EXPORT: Feature in development")
    else:
        st.sidebar.info("STANDBY: No session history logged")
    
    if st.sidebar.button("PURGE_LOGS: Clear History"):
        st.session_state.session_history = []
        st.rerun()


def main():
    """Main Millennium Falcon cockpit interface."""
    # Cockpit Header
    st.markdown('<h1 class="cockpit-header">MILLENNIUM FALCON CONTROL INTERFACE</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #cccccc; font-family: 'Courier New', monospace;">
        RAG_SYSTEM v2.1 | NEURAL_NETWORK ACTIVE | DIALOGUE_DATABASE ONLINE
    </div>
    """, unsafe_allow_html=True)
    
    # Check API health
    health = get_api_health()
    system_info = get_system_info()
    
    if health.get("status") != "healthy":
        st.error("ALERT: API connection failed. Ensure backend systems are operational.")
        st.stop()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_character" not in st.session_state:
        st.session_state.current_character = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Control Panel Sidebar
    with st.sidebar:
        st.markdown("## SYSTEM_STATUS: Control Panel")
        
        # System health indicators
        col1, col2 = st.columns(2)
        with col1:
            st.metric("DIALOGUE_DB", system_info.get('dialogue_lines', 0))
            st.metric("CHAR_COUNT", system_info.get('characters_count', 0))
        
        with col2:
            st.metric("FILM_DATA", system_info.get('movies_count', 0))
            status_indicator = "ONLINE" if health.get("models_loaded") else "OFFLINE"
            st.metric("LLM_STATUS", status_indicator)
        
        # Model information
        st.markdown("### MODEL_CONFIG: Neural Networks")
        st.text(f"EMBEDDING: {system_info.get('embedding_model', 'Unknown')}")
        st.text(f"LLM: {system_info.get('llm_model', 'Unknown')}")
        
        # Session history
        display_session_history_sidebar()
        
        # Settings
        st.markdown("## INTERFACE_CONFIG: Display Settings")
        show_explainability = st.checkbox("Enable Data Analysis Panels", value=True)
        show_pipeline = st.checkbox("Enable Process Monitoring", value=True)
        show_analytics = st.checkbox("Enable Tactical Analytics", value=False)
    
    # Main layout: Character selector and chat
    char_col, chat_col = st.columns([1, 2])
    
    # Character Selection Panel
    with char_col:
        st.markdown("### CHARACTER_SELECT: Personnel Database")
        
        characters = get_characters()
        if not characters:
            st.error("ERROR: No character data available")
            st.stop()
        
        # Get top 20 characters by dialogue count
        top_characters = [char for char in characters if isinstance(char, dict) and char.get('dialogue_count', 0) > 0]
        top_characters.sort(key=lambda x: x.get('dialogue_count', 0), reverse=True)
        top_characters = top_characters[:20]
        
        character_names = [char['name'] for char in top_characters]
        
        # Character dropdown
        selected_character = st.selectbox(
            "SELECT_CHARACTER: Choose target for communication",
            character_names,
            index=0 if character_names else None,
            help="Select from the top 20 characters with the most dialogue data"
        )
        
        if selected_character and selected_character != st.session_state.current_character:
            st.session_state.current_character = selected_character
            st.session_state.messages = []  # Clear when switching
            st.success(f"CONNECTION_ESTABLISHED: {selected_character}")
        
        # Display character info
        if selected_character:
            character_data = next((char for char in top_characters if char['name'] == selected_character), {})
            display_character_persona_header(selected_character, character_data)
    
    # Communication Interface
    with chat_col:
        if not selected_character:
            st.info("SELECT CHARACTER: Please choose a target from the personnel database")
            return
        
        st.markdown('<div class="comm-interface">', unsafe_allow_html=True)
        
        # Pipeline visualization
        if show_pipeline:
            display_pipeline_visualization("idle")
        
        # Communication Log
        if st.session_state.messages:
            st.markdown("### COMM_LOG: Active Transmission")
            
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="user-transmission">
                        <strong>USER_INPUT:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    character = message.get("character", "Character")
                    st.markdown(f"""
                    <div class="character-transmission">
                        <strong>RESPONSE_FROM {character.upper()}:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show explainability panels for the last message
                    if (show_explainability and 
                        message == st.session_state.messages[-1] and 
                        message.get("context_used")):
                        
                        with st.expander("DATA_ANALYSIS: Transmission Details", expanded=False):
                            # Retrieved context
                            display_retrieved_context_panel(message["context_used"])
                            
                            # LLM info
                            if message.get("metadata"):
                                display_llm_info_panel(message["metadata"])
        else:
            st.markdown(f"### COMM_READY: Begin transmission with {selected_character}")
            
            # Suggested questions
            suggestions = [
                "Tell me about your greatest challenge",
                "What motivates you?", 
                "Share wisdom with me",
                "Tell me about the Force",
                "What's your story?"
            ]
            
            st.markdown("QUICK_START: Pre-programmed queries")
            suggestion_cols = st.columns(len(suggestions))
            for i, suggestion in enumerate(suggestions):
                if suggestion_cols[i].button(f"QUERY: {suggestion}", key=f"suggestion_{i}"):
                    st.session_state.user_input = suggestion
                    st.rerun()
        
        # Input Terminal
        st.markdown("### INPUT_TERMINAL: Message Composition")
        user_input = st.text_area(
            "COMPOSE_MESSAGE: Enter transmission data",
            value=st.session_state.get("user_input", ""),
            height=100,
            placeholder=f"Enter query for {selected_character}...",
            key="chat_input"
        )
        
        # Send button
        if st.button("TRANSMIT: Send Message", type="primary", use_container_width=True):
            if user_input.strip():
                # Add user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": time.time()
                })
                
                # Show pipeline in action
                if show_pipeline:
                    with st.spinner("PROCESSING: System modules activating..."):
                        for step in ["embedding", "vector_scan", "llm_process"]:
                            display_pipeline_visualization(step)
                            time.sleep(0.5)
                
                # Generate response
                with st.spinner(f"PROCESSING: {selected_character} neural network active..."):
                    response_data = chat_with_api(
                        selected_character, 
                        user_input, 
                        st.session_state.session_id
                    )
                
                # Add character response
                st.session_state.messages.append({
                    "role": "character",
                    "content": response_data.get("response", "TRANSMISSION_ERROR: No response generated."),
                    "character": selected_character,
                    "timestamp": time.time(),
                    "context_used": response_data.get("context_used", []),
                    "metadata": response_data.get("metadata", {})
                })
                
                # Clear input
                st.session_state.user_input = ""
                st.success("TRANSMISSION_COMPLETE: Response received!")
                st.rerun()
            else:
                st.warning("INPUT_ERROR: Please enter a message!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Character Analytics (bottom panel)
    if show_analytics:
        st.markdown("---")
        display_character_analytics()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem; font-family: 'Courier New', monospace;">
        <strong>MILLENNIUM FALCON CONTROL INTERFACE</strong> | Neural RAG System v2.1 | Real-time Dialogue Processing<br>
        <em style="color: #00ff41;">SYSTEM STATUS: OPERATIONAL | HYPERSPACE COORDINATES LOCKED</em>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
