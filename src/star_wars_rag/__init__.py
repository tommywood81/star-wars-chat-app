"""
Star Wars RAG Chat Application

A retrieval-augmented generation (RAG) system for chatting with Star Wars characters
using dialogue from the original scripts.
"""

__version__ = "0.1.0"
__author__ = "Tommy Wood"

# from .data_processor import DialogueProcessor  # Removed for fast deployment
# from .embeddings import StarWarsEmbedder  # Removed for fast deployment
# from .retrieval import DialogueRetriever  # Removed for fast deployment
# from .app import StarWarsRAGApp  # Removed for fast deployment
# from .chat import StarWarsChatApp  # Removed for fast deployment
# from .llm import LocalLLM, create_llm  # Removed for fast deployment
# from .prompt import StarWarsPromptBuilder  # Removed for fast deployment
# from .models import ModelManager  # Removed for fast deployment
# from .database import DatabaseManager, setup_database  # Removed for fast deployment
from .api import app as fastapi_app
from .stt_service import app as stt_app, STTService
from .tts_service import app as tts_app, TTSService
from .llm_service import app as llm_app, LLMService

__all__ = [
    # "DialogueProcessor",  # Removed for fast deployment
    # "StarWarsEmbedder",  # Removed for fast deployment
    # "DialogueRetriever",  # Removed for fast deployment
    # "StarWarsRAGApp",  # Removed for fast deployment
    # "StarWarsChatApp",  # Removed for fast deployment
    # "LocalLLM",  # Removed for fast deployment
    # "create_llm",  # Removed for fast deployment
    # "StarWarsPromptBuilder",  # Removed for fast deployment
    # "ModelManager",  # Removed for fast deployment
    # "DatabaseManager",  # Removed for fast deployment
    # "setup_database",  # Removed for fast deployment
    "fastapi_app",
    "stt_app",
    "tts_app", 
    "llm_app",
    "STTService",
    "TTSService",
    "LLMService"
]
