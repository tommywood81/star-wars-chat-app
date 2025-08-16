"""
Star Wars RAG Chat Application

A retrieval-augmented generation (RAG) system for chatting with Star Wars characters
using dialogue from the original scripts.
"""

__version__ = "0.1.0"
__author__ = "Tommy Wood"

from star_wars_rag.data_processor import DialogueProcessor
from star_wars_rag.embeddings import StarWarsEmbedder
from star_wars_rag.retrieval import DialogueRetriever
from star_wars_rag.app import StarWarsRAGApp
from star_wars_rag.chat import StarWarsChatApp
from star_wars_rag.llm import LocalLLM, create_llm
from star_wars_rag.prompt import StarWarsPromptBuilder
from star_wars_rag.models import ModelManager
from star_wars_rag.database import DatabaseManager, setup_database
from star_wars_rag.api import app as fastapi_app

__all__ = [
    "DialogueProcessor",
    "StarWarsEmbedder", 
    "DialogueRetriever",
    "StarWarsRAGApp",
    "StarWarsChatApp",
    "LocalLLM",
    "create_llm",
    "StarWarsPromptBuilder",
    "ModelManager",
    "DatabaseManager",
    "setup_database",
    "fastapi_app"
]
