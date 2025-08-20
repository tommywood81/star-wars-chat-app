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
# Voice features will be added later
# import speech_recognition as sr
# import pyttsx3
# import threading
# import queue

# Page configuration
st.set_page_config(
    page_title="Star Wars Chat",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for sleek red and white theme
st.markdown("""
<style>
    /* Global styles - Force override */
    .main {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
        color: #1f2937 !important;
    }
    
    .stApp {
        background: transparent !important;
    }
    
    /* Text styling - Force override */
    h1, h2, h3, h4, h5, h6 {
        color: #1f2937 !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
    }
    
    p, div, span {
        color: #374151 !important;
    }
    
    /* Force all text to be visible */
    .stMarkdown, .stText, .stMarkdownContainer {
        color: #1f2937 !important;
    }
    
    /* Override Streamlit's default text colors */
    .stMarkdown p {
        color: #1f2937 !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #1f2937 !important;
    }
    
    /* Card containers */
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
    }
    
    .character-selector {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .voice-controls {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Info button */
    .info-button {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        border: none;
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: white;
        font-weight: 500;
        padding: 12px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(220, 38, 38, 0.4);
        background: linear-gradient(135deg, #b91c1c 0%, #991b1b 100%);
    }
    
    .stButton > button:disabled {
        background: #bdc3c7;
        color: #7f8c8d;
        box-shadow: none;
        transform: none;
    }
    
    /* Voice button special styling */
    .voice-button {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%) !important;
        box-shadow: 0 4px 12px rgba(39, 174, 96, 0.3) !important;
    }
    
    .voice-button:hover {
        box-shadow: 0 8px 20px rgba(39, 174, 96, 0.4) !important;
        background: linear-gradient(135deg, #229954 0%, #27ae60 100%) !important;
    }
    
    /* Streamlit specific elements */
    .stSelectbox > div > div {
        background: white;
        border-radius: 8px;
        border: 1px solid #e1e8ed;
    }
    
    .stTextArea > div > div > textarea {
        background: white;
        border-radius: 8px;
        border: 1px solid #e1e8ed;
        color: #2c3e50;
    }
    
    /* Conversation styling */
    .conversation-message {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        border-left: 4px solid #dc2626;
    }
    
    /* Info box styling */
    .stAlert {
        background: rgba(52, 152, 219, 0.1);
        border: 1px solid rgba(52, 152, 219, 0.3);
        border-radius: 8px;
        color: #2c3e50;
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #dc2626, transparent);
        margin: 24px 0;
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
if 'last_response_data' not in st.session_state:
    st.session_state.last_response_data = None
# Voice features will be added later
# if 'voice_enabled' not in st.session_state:
#     st.session_state.voice_enabled = False
# if 'tts_enabled' not in st.session_state:
#     st.session_state.tts_enabled = False

# API configuration
API_BASE_URL = "http://star-wars-api:8000"

def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_available_characters():
    """Get available characters from API - only top 15 by line count."""
    try:
        response = requests.get(f"{API_BASE_URL}/characters", timeout=5)
        if response.status_code == 200:
            data = response.json()
            characters = data.get('characters', [])
            # Sort by dialogue count and get top 15
            sorted_chars = sorted(characters, key=lambda x: x.get('dialogue_count', 0), reverse=True)
            return [char['name'] for char in sorted_chars[:15]]
    except Exception as e:
        st.error(f"Error fetching characters: {e}")
    return []

def get_character_suggestions(character_name):
    """Get custom quick start suggestions for a specific character."""
    suggestions = {
        "Han Solo": [
            "What's your honest opinion about the Force?",
            "Tell me about that time you shot first",
            "How do you really feel about Princess Leia?"
        ],
        "Luke Skywalker": [
            "Tell me about your father and the dark side",
            "What was it like training with Yoda on Dagobah?",
            "How did you feel when you found out about Leia?"
        ],
        "C-3PO": [
            "What's your protocol for dealing with droids?",
            "Tell me about your relationship with R2-D2",
            "What's the most dangerous situation you've been in?"
        ],
        "Princess Leia": [
            "What's it like being a princess and a rebel?",
            "How do you handle being called 'Your Worship'?",
            "What's your strategy for dealing with the Empire?"
        ],
        "Darth Vader": [
            "What do you think about your son's potential?",
            "How do you feel about the Emperor's plans?",
            "What's your opinion on the Death Star's effectiveness?"
        ],
        "Obi-Wan Kenobi": [
            "What was it like training Anakin Skywalker?",
            "How do you feel about becoming a Force ghost?",
            "What's your advice for young Jedi?"
        ],
        "Lando Calrissian": [
            "What was it like running Cloud City?",
            "How do you feel about your deal with Vader?",
            "What's your strategy for winning at sabacc?"
        ],
        "Yoda": [
            "What's the most important lesson you teach?",
            "How do you feel about Luke's training?",
            "What's your philosophy on fear and anger?"
        ]
    }
    return suggestions.get(character_name, [
        "Tell me about your experiences",
        "What's your opinion on the current situation?",
        "How do you feel about the Force?"
    ])

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

def display_explainability_panel(response_data):
    """Display the full prompt sent to the LLM for explainability."""
    if st.button("🔍 Show Full Prompt", key="explainability_btn", help="Click to see the complete prompt sent to the LLM"):
        if response_data and 'conversation_metadata' in response_data:
            metadata = response_data['conversation_metadata']
            full_prompt = metadata.get('full_prompt', 'No prompt data available')
            
            st.markdown("### 🔍 Full Prompt Sent to LLM")
            st.markdown("**This is the exact prompt that was sent to the Phi-2 model:**")
            
            # Display the full prompt in a code block for better readability
            st.code(full_prompt, language="text")
            
            # Also show the retrieved context separately
            if 'retrieved_context' in metadata:
                st.markdown("### 📚 Retrieved Context")
                st.markdown("**These are the 6 most relevant dialogue lines used as context:**")
                context = metadata['retrieved_context']
                for i, item in enumerate(context, 1):
                    char = item.get('character', 'Unknown')
                    dialogue = item.get('dialogue', '')
                    movie = item.get('movie', '')
                    similarity = item.get('similarity', 0)
                    
                    st.markdown(f"**{i}.** **{char}** (from {movie}, similarity: {similarity:.3f})")
                    st.markdown(f"   *\"{dialogue}\"*")
                    st.markdown("")
        else:
            st.warning("No prompt data available. Try sending a message first to see the explainability.")

# Voice features will be added later
# def voice_input():
#     """Handle voice input using speech recognition."""
#     recognizer = sr.Recognizer()
#     
#     with sr.Microphone() as source:
#         st.info("🎤 Listening... Speak now!")
#         try:
#             audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
#             text = recognizer.recognize_google(audio)
#             return text
#         except sr.WaitTimeoutError:
#             st.warning("No speech detected. Please try again.")
#             return None
#         except sr.UnknownValueError:
#             st.warning("Could not understand audio. Please try again.")
#             return None
#         except Exception as e:
#             st.error(f"Voice input error: {e}")
#             return None
# 
# def text_to_speech(text, character):
#     """Convert text to speech."""
#     try:
#         engine = pyttsx3.init()
#         
#         # Set voice properties based on character
#         if character == "VADER":
#             engine.setProperty('rate', 120)
#             engine.setProperty('volume', 0.8)
#         elif character == "YODA":
#             engine.setProperty('rate', 100)
#             engine.setProperty('volume', 0.7)
#         else:
#             engine.setProperty('rate', 150)
#             engine.setProperty('volume', 0.9)
#         
#         # Run TTS in a separate thread to avoid blocking
#         def speak():
#             engine.say(text)
#             engine.runAndWait()
#         
#         thread = threading.Thread(target=speak)
#         thread.start()
#         
#         return True
#     except Exception as e:
#         st.error(f"TTS error: {e}")
#         return False

def main():
    """Main dashboard function."""
    # Sleek Jinja2-style theme is applied via CSS
    
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
    
    # Voice controls (coming soon)
    st.markdown("### Voice Controls")
    st.info("🎤 Voice features coming soon! For now, enjoy the text-based chat experience.")
    
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
    
    # Explainability panel
    if st.session_state.last_response_data:
        display_explainability_panel(st.session_state.last_response_data)
    
    # Input section
    st.markdown("#### Send Message")
    
    # Voice input button (coming soon)
    # if st.session_state.voice_enabled:
    #     if st.button("🎤 Speak", key="speak_btn", help="Click and speak"):
    #         with st.spinner("Listening..."):
    #             voice_text = voice_input()
    #             if voice_text:
    #                 st.session_state.voice_input = voice_text
    #                 st.success(f"🎤 Heard: {voice_text}")
    
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
                        
                        # Store the full response data for explainability
                        st.session_state.last_response_data = response
                        
                        # Add to conversation history
                        st.session_state.conversation.append((user_message, character_response))
                        
                        # Text-to-speech if enabled (coming soon)
                        # if st.session_state.tts_enabled:
                        #     text_to_speech(character_response, selected_character)
                        
                        # Clear input
                        st.session_state.voice_input = ""
                        st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Chat", key="clear_btn"):
            st.session_state.conversation = []
            st.session_state.voice_input = ""
            st.session_state.last_response_data = None
            st.rerun()
    
    # Quick start suggestions - customized for selected character
    st.markdown("### Quick Start Suggestions")
    suggestions = get_character_suggestions(selected_character)
    
    cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(suggestion, key=f"suggestion_{i}"):
                st.session_state.voice_input = suggestion
                st.rerun()

if __name__ == "__main__":
    main()

