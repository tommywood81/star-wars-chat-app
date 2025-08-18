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
    page_title="Star Wars RAG Analytics Dashboard",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern SaaS Dashboard Theme CSS
st.markdown("""
<style>
    /* Modern SaaS Dashboard Theme */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        color: #1e293b;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
    }
    
    .main .block-container {
        padding: 2rem 2rem 3rem;
        max-width: 100%;
    }
    
    /* Dashboard Header */
    .dashboard-header {
        text-align: center;
        color: #1e40af;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 2rem;
        padding: 2rem;
        background: white;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
    }
    
    /* Card Components */
    .saas-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
    }
    
    .saas-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transform: translateY(-1px);
    }
    
    /* Character Profile Card */
    .character-profile {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
    }
    
    .character-avatar {
        width: 60px;
        height: 60px;
        border-radius: 12px;
        border: 2px solid #3b82f6;
        background: linear-gradient(135deg, #dbeafe, #bfdbfe);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        margin-right: 1rem;
        color: #1e40af;
        font-weight: 600;
    }
    
    /* Chat Interface */
    .chat-interface {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0 0.5rem 20%;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.2);
    }
    
    .character-message {
        background: linear-gradient(135deg, #f1f5f9, #e2e8f0);
        color: #1e293b;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 20% 0.5rem 0;
        border: 1px solid #cbd5e1;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    /* Data Panels */
    .data-panel {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 3px solid #3b82f6;
    }
    
    .context-line {
        background: white;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #3b82f6;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        transition: all 0.2s ease;
    }
    
    .context-line:hover {
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .context-line.priority {
        border-left-color: #f59e0b;
        background: #fffbeb;
    }
    
    /* Process Indicators */
    .process-indicator {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    .process-indicator.active {
        border-color: #3b82f6;
        background: #dbeafe;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    .process-indicator.complete {
        border-color: #10b981;
        background: #d1fae5;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        transition: all 0.2s ease;
        font-weight: 500;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.2);
        transform: translateY(-1px);
    }
    
    /* Input Styling */
    .stTextArea > div > div > textarea {
        background: white;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 0.75rem;
        transition: all 0.2s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Selectbox Styling */
    .stSelectbox > div > div {
        background: white;
        border: 1px solid #d1d5db;
        border-radius: 8px;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #1e293b;
        font-weight: 600;
    }
    
    /* Clean Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    
    /* Success/Info/Warning Messages */
    .stSuccess {
        background: #d1fae5;
        border: 1px solid #10b981;
        color: #065f46;
    }
    
    .stInfo {
        background: #dbeafe;
        border: 1px solid #3b82f6;
        color: #1e40af;
    }
    
    .stWarning {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        color: #92400e;
    }
    
    .stError {
        background: #fee2e2;
        border: 1px solid #ef4444;
        color: #dc2626;
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
    """Display character profile card with modern SaaS styling."""
    dialogue_count = character_data.get('dialogue_count', 0)
    movies = character_data.get('movies', [])
    sample_dialogue = character_data.get('sample_dialogue', [])
    
    # Character initials for clean display
    char_initials = ''.join([name[0] for name in character.split()[:2]])
    
    st.markdown(f"""
    <div class="character-profile">
        <div style="display: flex; align-items: center;">
            <div class="character-avatar">{char_initials}</div>
            <div>
                <h2 style="margin: 0; color: #1e40af;">{character}</h2>
                <p style="margin: 0.5rem 0; color: #64748b;">
                    {dialogue_count} dialogue lines across {len(movies)} films
                </p>
                <div style="display: flex; gap: 1rem; margin-top: 0.5rem;">
                    <span style="background: #dbeafe; color: #1e40af; padding: 0.3rem 0.8rem; border-radius: 6px; font-size: 0.8rem; font-weight: 500;">
                        {len(movies)} Films
                    </span>
                    <span style="background: #fef3c7; color: #92400e; padding: 0.3rem 0.8rem; border-radius: 6px; font-size: 0.8rem; font-weight: 500;">
                        {dialogue_count} Lines
                    </span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_pipeline_visualization(step: str = "idle"):
    """Display processing pipeline with modern SaaS indicators."""
    steps = [
        ("Query Processing", "1"),
        ("Text Embedding", "2"),
        ("Vector Search", "3"),
        ("Context Retrieval", "4"),
        ("LLM Generation", "5"),
        ("Response Output", "6")
    ]
    
    cols = st.columns(len(steps))
    
    for i, (step_name, number) in enumerate(steps):
        with cols[i]:
            if step == "idle":
                class_name = "process-indicator"
            elif i < len(steps) - 1:
                class_name = "process-indicator complete"
            elif step_name.lower().replace(" ", "_") == step.replace("_", " "):
                class_name = "process-indicator active"
            else:
                class_name = "process-indicator"
            
            st.markdown(f"""
            <div class="{class_name}">
                <div style="font-size: 1.5rem; color: #3b82f6; font-weight: 600;">{number}</div>
                <div style="font-size: 0.8rem; margin-top: 0.5rem; color: #64748b;">{step_name}</div>
            </div>
            """, unsafe_allow_html=True)


def display_retrieved_context_panel(context_used: List[Dict], show_highlights: bool = True):
    """Display retrieved context with modern SaaS styling."""
    if not context_used:
        return
    
    st.markdown("### Retrieved Context")
    
    with st.expander(f"View {len(context_used)} retrieved dialogue lines", expanded=False):
        for i, ctx in enumerate(context_used, 1):
            character = ctx.get('character', 'Unknown')
            dialogue = ctx.get('dialogue', '')
            movie = ctx.get('movie', 'Unknown')
            similarity = ctx.get('similarity', 0)
            
            # Determine if this line was high priority (high similarity)
            highlight_class = "priority" if similarity > 0.4 else ""
            
            st.markdown(f"""
            <div class="context-line {highlight_class}">
                <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 0.5rem;">
                    <strong style="color: #1e40af;">#{i} {character}</strong>
                    <div style="margin-left: auto;">
                        <span style="background: #f1f5f9; color: #64748b; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8rem;">
                            {movie}
                        </span>
                        <span style="background: #dbeafe; color: #1e40af; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8rem; margin-left: 0.5rem;">
                            {similarity:.3f}
                        </span>
                    </div>
                </div>
                <div style="font-style: italic; color: #475569;">
                    "{dialogue}"
                </div>
            </div>
            """, unsafe_allow_html=True)


def display_llm_info_panel(metadata: Dict):
    """Display LLM processing metrics with modern styling."""
    st.markdown("### Model Processing Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Context Found",
            metadata.get('retrieval_results', 0),
            help="Number of similar dialogue lines found"
        )
    
    with col2:
        st.metric(
            "Lines Used", 
            metadata.get('context_lines_used', 0),
            help="Lines sent to LLM for context"
        )
    
    with col3:
        total_time = metadata.get('total_time_seconds', 0)
        st.metric(
            "Response Time",
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
    """Display character analytics with modern SaaS styling."""
    st.markdown("### Character Analytics")
    
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
    
    # Top characters distribution
    with col1:
        st.markdown("#### Top Characters by Dialogue")
        
        # Get top 10 characters
        top_chars = list(zip(char_names, dialogue_counts))
        top_chars.sort(key=lambda x: x[1], reverse=True)
        top_chars = top_chars[:10]
        
        if top_chars:
            fig = px.pie(
                values=[count for _, count in top_chars],
                names=[name for name, _ in top_chars],
                color_discrete_sequence=['#3b82f6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#84cc16', '#f97316', '#6366f1']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#1e293b',
                font_family='system-ui'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Movies distribution
    with col2:
        st.markdown("#### Content by Film")
        
        movie_counts = Counter(movies_data)
        if movie_counts:
            fig = px.bar(
                x=list(movie_counts.keys()),
                y=list(movie_counts.values()),
                color_discrete_sequence=['#3b82f6', '#06b6d4', '#10b981']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#1e293b',
                font_family='system-ui',
                xaxis_title="Film",
                yaxis_title="Character Count"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)


def display_session_history_sidebar():
    """Display session history in sidebar."""
    st.sidebar.markdown("## Session History")
    
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
    
    if st.sidebar.button("Clear History"):
        st.session_state.session_history = []
        st.rerun()


def main():
    """Main Star Wars RAG dashboard interface."""
    # Dashboard Header
    st.markdown('<h1 class="dashboard-header">Star Wars RAG Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #64748b;">
        Advanced RAG system with real-time character interaction and explainable AI
    </div>
    """, unsafe_allow_html=True)
    
    # Check API health
    health = get_api_health()
    system_info = get_system_info()
    
    if health.get("status") != "healthy":
        st.error("API connection failed. Please ensure the backend service is running.")
        st.stop()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_character" not in st.session_state:
        st.session_state.current_character = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Dashboard Sidebar
    with st.sidebar:
        st.markdown("## System Overview")
        
        # System metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Dialogue Lines", system_info.get('dialogue_lines', 0))
            st.metric("Characters", system_info.get('characters_count', 0))
        
        with col2:
            st.metric("Films", system_info.get('movies_count', 0))
            status_indicator = "Online" if health.get("models_loaded") else "Offline"
            st.metric("Model Status", status_indicator)
        
        # Model information
        st.markdown("### Model Configuration")
        st.text(f"Embedding: {system_info.get('embedding_model', 'Unknown')}")
        st.text(f"LLM: {system_info.get('llm_model', 'Unknown')}")
        
        # Session history
        display_session_history_sidebar()
        
        # Settings
        st.markdown("## Display Settings")
        show_explainability = st.checkbox("Show Explainability Panels", value=True)
        show_pipeline = st.checkbox("Show Processing Pipeline", value=True)
        show_analytics = st.checkbox("Show Character Analytics", value=False)
    
    # Main layout: Character selector and chat
    char_col, chat_col = st.columns([1, 2])
    
    # Character Selection Panel
    with char_col:
        st.markdown("### Character Selection")
        
        characters = get_characters()
        if not characters:
            st.error("No character data available")
            st.stop()
        
        # Get top 20 characters by dialogue count
        top_characters = [char for char in characters if isinstance(char, dict) and char.get('dialogue_count', 0) > 0]
        top_characters.sort(key=lambda x: x.get('dialogue_count', 0), reverse=True)
        top_characters = top_characters[:20]
        
        character_names = [char['name'] for char in top_characters]
        
        # Character dropdown
        selected_character = st.selectbox(
            "Choose a character to chat with:",
            character_names,
            index=0 if character_names else None,
            help="Select from the top 20 characters with the most dialogue"
        )
        
        if selected_character and selected_character != st.session_state.current_character:
            st.session_state.current_character = selected_character
            st.session_state.messages = []  # Clear when switching
            st.success(f"Now chatting with {selected_character}")
        
        # Display character info
        if selected_character:
            character_data = next((char for char in top_characters if char['name'] == selected_character), {})
            display_character_persona_header(selected_character, character_data)
    
    # Chat Interface
    with chat_col:
        if not selected_character:
            st.info("Please select a character to start chatting")
            return
        
        st.markdown('<div class="chat-interface">', unsafe_allow_html=True)
        
        # Pipeline visualization
        if show_pipeline:
            display_pipeline_visualization("idle")
        
        # Chat Messages
        if st.session_state.messages:
            st.markdown("### Conversation")
            
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="user-message">
                        <strong>You:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    character = message.get("character", "Character")
                    st.markdown(f"""
                    <div class="character-message">
                        <strong>{character}:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show explainability panels for the last message
                    if (show_explainability and 
                        message == st.session_state.messages[-1] and 
                        message.get("context_used")):
                        
                        with st.expander("Explainability Details", expanded=False):
                            # Retrieved context
                            display_retrieved_context_panel(message["context_used"])
                            
                            # LLM info
                            if message.get("metadata"):
                                display_llm_info_panel(message["metadata"])
        else:
            st.markdown(f"### Start a conversation with {selected_character}!")
            
            # Suggested questions
            suggestions = [
                "Tell me about your greatest challenge",
                "What motivates you?", 
                "Share wisdom with me",
                "Tell me about the Force",
                "What's your story?"
            ]
            
            st.markdown("**Quick start suggestions:**")
            suggestion_cols = st.columns(len(suggestions))
            for i, suggestion in enumerate(suggestions):
                if suggestion_cols[i].button(suggestion, key=f"suggestion_{i}"):
                    st.session_state.user_input = suggestion
                    st.rerun()
        
        # Chat Input
        st.markdown("### Your Message")
        user_input = st.text_area(
            "Type your message:",
            value=st.session_state.get("user_input", ""),
            height=100,
            placeholder=f"Ask {selected_character} anything...",
            key="chat_input"
        )
        
        # Send button
        if st.button("Send Message", type="primary", use_container_width=True):
            if user_input.strip():
                # Add user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": time.time()
                })
                
                # Show pipeline in action
                if show_pipeline:
                    with st.spinner("Processing your message..."):
                        for step in ["query processing", "text embedding", "llm generation"]:
                            display_pipeline_visualization(step)
                            time.sleep(0.5)
                
                # Generate response
                with st.spinner(f"{selected_character} is thinking..."):
                    response_data = chat_with_api(
                        selected_character, 
                        user_input, 
                        st.session_state.session_id
                    )
                
                # Add character response
                st.session_state.messages.append({
                    "role": "character",
                    "content": response_data.get("response", "I couldn't respond right now."),
                    "character": selected_character,
                    "timestamp": time.time(),
                    "context_used": response_data.get("context_used", []),
                    "metadata": response_data.get("metadata", {})
                })
                
                # Clear input
                st.session_state.user_input = ""
                st.success("Response generated!")
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
    <div style="text-align: center; color: #64748b; padding: 2rem;">
        <strong>Star Wars RAG Analytics Dashboard</strong> | Advanced AI system with explainable retrieval<br>
        <em style="color: #3b82f6;">Powered by neural embeddings and large language models</em>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

