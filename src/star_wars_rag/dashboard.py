"""
Simple Star Wars RAG Chat Dashboard with Voice Features
"""

import streamlit as st
import requests
import json
import time
import uuid
from pathlib import Path
import base64
from io import BytesIO
import speech_recognition as sr
import pyttsx3
import threading
import queue

# Page configuration
st.set_page_config(
    page_title="Star Wars Chat",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean design
st.markdown("""
<style>
    .main {
        background-image: url('data:image/jpeg;base64,{}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    
    .stApp {
        background: rgba(0, 0, 0, 0.8);
    }
    
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .character-selector {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    
    .voice-controls {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .info-button {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
    }
    
    .stButton > button {
        border-radius: 25px;
        border: 2px solid #FFD700;
        background: linear-gradient(90deg, #FFD700, #FFA500);
        color: #000;
        font-weight: bold;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(255, 215, 0, 0.3);
    }
    
    .voice-button {
        background: linear-gradient(90deg, #4CAF50, #45a049) !important;
        border: 2px solid #4CAF50 !important;
    }
    
    .voice-button:hover {
        box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# Load background image
def load_background_image():
    """Load and encode background image."""
    try:
        background_path = Path("static/images/background.jpeg")
        if background_path.exists():
            with open(background_path, "rb") as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode()
    except Exception as e:
        st.error(f"Error loading background image: {e}")
    return ""

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'selected_character' not in st.session_state:
    st.session_state.selected_character = None
if 'voice_enabled' not in st.session_state:
    st.session_state.voice_enabled = False
if 'tts_enabled' not in st.session_state:
    st.session_state.tts_enabled = False

# API configuration
API_BASE_URL = "http://localhost:8000"

def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_available_characters():
    """Get available characters from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/characters", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [char['name'] for char in data.get('characters', [])]
    except Exception as e:
        st.error(f"Error fetching characters: {e}")
    return []

def chat_with_character(message, character):
    """Send chat message to API."""
    try:
        payload = {
            "character": character,
            "message": message,
            "session_id": st.session_state.session_id,
            "max_tokens": 150,
            "temperature": 0.7
        }
        
        response = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error communicating with API: {e}")
        return None

def display_info_modal():
    """Display information about the application."""
    if st.button("ℹ️ Info", key="info_button", help="Click for app information"):
        st.info("""
        ## Star Wars RAG Chat Application
        
        **How it works:**
        1. **Text Processing**: Raw Star Wars script data is processed and cleaned
        2. **Embedding Generation**: Dialogue lines are converted to vector embeddings
        3. **Vector Search**: When you ask a question, the system finds the 6 most relevant dialogue lines
        4. **Context Retrieval**: These lines provide context about how the character speaks
        5. **LLM Generation**: A local Phi-2 model generates responses based on the context
        6. **Response Output**: The character responds in their authentic voice
        
        **Technology Stack:**
        - **RAG (Retrieval-Augmented Generation)**: Combines search with AI generation
        - **Local LLM**: Phi-2 model runs entirely on your device
        - **Vector Database**: PostgreSQL with pgvector for similarity search
        - **Embeddings**: Sentence transformers for semantic understanding
        
        **Voice Features:**
        - **Speech-to-Text**: Use your microphone to speak to characters
        - **Text-to-Speech**: Characters can respond through your speakers
        
        **Data Source:**
        - Original Star Wars Trilogy scripts (Episodes IV, V, VI)
        - Clean, authentic dialogue from the movies
        
        This system ensures responses stay true to each character's personality and speaking style!
        """)

def voice_input():
    """Handle voice input using speech recognition."""
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak now!")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            st.warning("No speech detected. Please try again.")
            return None
        except sr.UnknownValueError:
            st.warning("Could not understand audio. Please try again.")
            return None
        except Exception as e:
            st.error(f"Voice input error: {e}")
            return None

def text_to_speech(text, character):
    """Convert text to speech."""
    try:
        engine = pyttsx3.init()
        
        # Set voice properties based on character
        if character == "VADER":
            engine.setProperty('rate', 120)
            engine.setProperty('volume', 0.8)
        elif character == "YODA":
            engine.setProperty('rate', 100)
            engine.setProperty('volume', 0.7)
        else:
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
        
        # Run TTS in a separate thread to avoid blocking
        def speak():
            engine.say(text)
            engine.runAndWait()
        
        thread = threading.Thread(target=speak)
        thread.start()
        
        return True
    except Exception as e:
        st.error(f"TTS error: {e}")
        return False

def main():
    """Main dashboard function."""
    # Load background
    background_b64 = load_background_image()
    if background_b64:
        st.markdown(f"""
        <style>
        .main {{
            background-image: url('data:image/jpeg;base64,{background_b64}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """, unsafe_allow_html=True)
    
    # Header
    st.title("⭐ Star Wars Character Chat")
    st.markdown("Chat with your favorite Star Wars characters using AI and voice!")
    
    # Info button
    display_info_modal()
    
    # Check API health
    if not check_api_health():
        st.error("🚨 API is not responding. Please ensure the backend is running.")
        st.stop()
    
    # Character selection
    st.markdown("### Choose Your Character")
    characters = get_available_characters()
    
    if characters:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_character = st.selectbox(
                "Select a character to chat with:",
                characters,
                key="character_selector"
            )
            st.session_state.selected_character = selected_character
        
        with col2:
            if st.button("🔄 Refresh Characters", key="refresh_char"):
                st.rerun()
    
    else:
        st.warning("No characters available. Please check the API connection.")
        st.stop()
    
    # Voice controls
    st.markdown("### Voice Controls")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎤 Enable Voice Input", key="voice_input_btn", 
                    help="Click to enable microphone input"):
            st.session_state.voice_enabled = not st.session_state.voice_enabled
    
    with col2:
        if st.button("🔊 Enable Voice Output", key="voice_output_btn",
                    help="Click to enable text-to-speech"):
            st.session_state.tts_enabled = not st.session_state.tts_enabled
    
    # Show voice status
    voice_status = "✅ Enabled" if st.session_state.voice_enabled else "❌ Disabled"
    tts_status = "✅ Enabled" if st.session_state.tts_enabled else "❌ Disabled"
    
    st.markdown(f"**Voice Input:** {voice_status} | **Voice Output:** {tts_status}")
    
    # Chat interface
    st.markdown("### Chat with " + selected_character)
    
    # Display conversation history
    if st.session_state.conversation:
        st.markdown("#### Conversation History")
        for i, (user_msg, char_msg) in enumerate(st.session_state.conversation):
            with st.container():
                st.markdown(f"**You:** {user_msg}")
                st.markdown(f"**{selected_character}:** {char_msg}")
                st.divider()
    
    # Input section
    st.markdown("#### Send Message")
    
    # Voice input button
    if st.session_state.voice_enabled:
        if st.button("🎤 Speak", key="speak_btn", help="Click and speak"):
            with st.spinner("Listening..."):
                voice_text = voice_input()
                if voice_text:
                    st.session_state.voice_input = voice_text
                    st.success(f"🎤 Heard: {voice_text}")
    
    # Text input
    user_message = st.text_area(
        "Type your message:",
        value=st.session_state.get('voice_input', ''),
        height=100,
        key="message_input"
    )
    
    # Send button
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("🚀 Send", key="send_btn", disabled=not user_message.strip()):
            if user_message.strip() and selected_character:
                with st.spinner(f"Getting response from {selected_character}..."):
                    response = chat_with_character(user_message, selected_character)
                    
                    if response:
                        character_response = response.get('response', 'No response received.')
                        
                        # Add to conversation history
                        st.session_state.conversation.append((user_message, character_response))
                        
                        # Text-to-speech if enabled
                        if st.session_state.tts_enabled:
                            text_to_speech(character_response, selected_character)
                        
                        # Clear input
                        st.session_state.voice_input = ""
                        st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Chat", key="clear_btn"):
            st.session_state.conversation = []
            st.session_state.voice_input = ""
            st.rerun()
    
    # Quick start suggestions
    st.markdown("### Quick Start Suggestions")
    suggestions = [
        "Tell me about the Force",
        "What's your opinion on droids?",
        "How did you become a Jedi?",
        "What's your favorite ship?",
        "Tell me about your lightsaber",
        "What's the most important lesson you've learned?"
    ]
    
    cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(suggestion, key=f"suggestion_{i}"):
                st.session_state.voice_input = suggestion
                st.rerun()

if __name__ == "__main__":
    main()

