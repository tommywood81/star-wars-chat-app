"""
Star Wars RAG Chat Application

A retrieval-augmented generation (RAG) system for chatting with Star Wars characters
using dialogue from the original scripts.
"""

__version__ = "0.1.0"
__author__ = "Tommy Wood"

# Core services - these are needed for individual service containers
from .api import app as fastapi_app

# Conditional imports for individual services
try:
    from .stt_service import app as stt_app, STTService
except ImportError:
    stt_app = None
    STTService = None

try:
    from .tts_service import app as tts_app, TTSService
except ImportError:
    tts_app = None
    TTSService = None

try:
    from .llm_service import app as llm_app, LLMService
except ImportError:
    llm_app = None
    LLMService = None

# Legacy imports - commented out to avoid dependency issues in individual containers
# from .data_processor import DialogueProcessor
# from .embeddings import StarWarsEmbedder
# from .retrieval import DialogueRetriever
# from .app import StarWarsRAGApp
# from .chat import StarWarsChatApp
# from .llm import LocalLLM, create_llm
# from .prompt import StarWarsPromptBuilder
# from .models import ModelManager
# from .database import DatabaseManager, setup_database

__all__ = [
    # Core services
    "fastapi_app",
    "stt_app",
    "tts_app", 
    "llm_app",
    "STTService",
    "TTSService",
    "LLMService"
]
