"""
Star Wars RAG Chat Application

A retrieval-augmented generation (RAG) system for chatting with Star Wars characters
using dialogue from the original scripts.
"""

__version__ = "0.1.0"
__author__ = "Tommy Wood"

from .data_processor import DialogueProcessor
from .embeddings import StarWarsEmbedder
from .retrieval import DialogueRetriever
from .app import StarWarsRAGApp
from .chat import StarWarsChatApp
from .llm import LocalLLM, create_llm
from .prompt import StarWarsPromptBuilder
from .models import ModelManager

__all__ = [
    "DialogueProcessor",
    "StarWarsEmbedder", 
    "DialogueRetriever",
    "StarWarsRAGApp",
    "StarWarsChatApp",
    "LocalLLM",
    "create_llm",
    "StarWarsPromptBuilder",
    "ModelManager"
]
