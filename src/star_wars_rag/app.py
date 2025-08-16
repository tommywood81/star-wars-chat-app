"""
Main application module for Star Wars RAG system.

This module provides a high-level interface for the complete RAG pipeline.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np

from .data_processor import DialogueProcessor
from .embeddings import StarWarsEmbedder
from .retrieval import DialogueRetriever

logger = logging.getLogger(__name__)


class StarWarsRAGApp:
    """High-level interface for the Star Wars RAG chat application."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the RAG application.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.processor = DialogueProcessor()
        self.embedder = StarWarsEmbedder(model_name)
        self.retriever = DialogueRetriever(self.embedder)
        self.is_loaded = False
        
        logger.info("Initialized Star Wars RAG application")
    
    def load_from_scripts(self, script_directory: Union[str, Path], 
                         pattern: str = "*.txt",
                         save_processed_data: Optional[str] = None,
                         save_embeddings: Optional[str] = None) -> None:
        """Load and process script files to build the RAG system.
        
        Args:
            script_directory: Directory containing script files
            pattern: File pattern to match (e.g., "*.txt")
            save_processed_data: Path to save processed dialogue data (CSV)
            save_embeddings: Path to save embeddings (NPY)
            
        Raises:
            FileNotFoundError: If script directory doesn't exist
            ValueError: If no valid scripts found
        """
        logger.info(f"Loading scripts from {script_directory} with pattern {pattern}")
        
        # Process scripts
        dialogue_df = self.processor.process_multiple_scripts(script_directory, pattern)
        
        if dialogue_df.empty:
            raise ValueError("No dialogue data extracted from scripts")
        
        # Save processed data if requested
        if save_processed_data:
            dialogue_df.to_csv(save_processed_data, index=False)
            logger.info(f"Saved processed data to {save_processed_data}")
        
        # Generate embeddings and load into retriever
        self.retriever.load_dialogue_data(dialogue_df)
        
        # Save embeddings if requested
        if save_embeddings:
            self.embedder.save_embeddings(self.retriever.embeddings, save_embeddings)
            logger.info(f"Saved embeddings to {save_embeddings}")
        
        self.is_loaded = True
        
        # Log statistics
        stats = self.get_system_stats()
        logger.info(f"System loaded: {stats}")
    
    def load_from_processed_data(self, dialogue_csv: Union[str, Path],
                                embeddings_npy: Optional[Union[str, Path]] = None) -> None:
        """Load system from pre-processed dialogue data and embeddings.
        
        Args:
            dialogue_csv: Path to processed dialogue CSV file
            embeddings_npy: Path to embeddings NPY file (computed if None)
            
        Raises:
            FileNotFoundError: If required files don't exist
        """
        logger.info(f"Loading from processed data: {dialogue_csv}")
        
        # Load dialogue data
        dialogue_df = pd.read_csv(dialogue_csv)
        
        # Load or compute embeddings
        if embeddings_npy and Path(embeddings_npy).exists():
            embeddings = self.embedder.load_embeddings(str(embeddings_npy))
            self.retriever.load_dialogue_data(dialogue_df, embeddings)
            logger.info(f"Loaded pre-computed embeddings from {embeddings_npy}")
        else:
            self.retriever.load_dialogue_data(dialogue_df)
            logger.info("Computed embeddings from dialogue data")
        
        self.is_loaded = True
        
        # Log statistics
        stats = self.get_system_stats()
        logger.info(f"System loaded: {stats}")
    
    def chat_with_character(self, query: str, character: str, 
                           context_size: int = 3) -> Dict:
        """Get character-specific response based on query.
        
        Args:
            query: User's query/message
            character: Character to chat with
            context_size: Number of relevant dialogue lines to retrieve
            
        Returns:
            Dictionary with character response and metadata
            
        Raises:
            RuntimeError: If system is not loaded
            ValueError: If character not found
        """
        if not self.is_loaded:
            raise RuntimeError("System not loaded. Call load_from_scripts() or load_from_processed_data() first.")
        
        # Check if character exists
        available_characters = self.retriever.get_available_characters()
        if character not in available_characters:
            raise ValueError(f"Character '{character}' not found. Available: {available_characters}")
        
        # Retrieve relevant dialogue
        relevant_dialogue = self.retriever.retrieve_similar_dialogue(
            query,
            character_filter=character,
            top_k=context_size
        )
        
        if not relevant_dialogue:
            return {
                'character': character,
                'response': f"I don't have anything to say about that.",
                'context': [],
                'query': query
            }
        
        # For now, return the most relevant dialogue as response
        # In a full implementation, this would be fed to an LLM
        best_response = relevant_dialogue[0]
        
        return {
            'character': character,
            'response': best_response['dialogue'],
            'similarity': best_response['similarity'],
            'context': relevant_dialogue,
            'query': query,
            'movie': best_response.get('movie', 'Unknown'),
            'scene': best_response.get('scene', 'Unknown')
        }
    
    def search_dialogue(self, query: str, top_k: int = 5,
                       character_filter: Optional[str] = None,
                       movie_filter: Optional[str] = None,
                       min_similarity: float = 0.0) -> List[Dict]:
        """Search for relevant dialogue across all characters.
        
        Args:
            query: Search query
            top_k: Number of results to return
            character_filter: Filter by specific character
            movie_filter: Filter by specific movie
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of relevant dialogue results
            
        Raises:
            RuntimeError: If system is not loaded
        """
        if not self.is_loaded:
            raise RuntimeError("System not loaded. Call load_from_scripts() or load_from_processed_data() first.")
        
        return self.retriever.retrieve_similar_dialogue(
            query=query,
            top_k=top_k,
            character_filter=character_filter,
            movie_filter=movie_filter,
            min_similarity=min_similarity
        )
    
    def get_character_dialogue_sample(self, character: str, 
                                    sample_size: int = 5) -> List[Dict]:
        """Get sample dialogue for a character.
        
        Args:
            character: Character name
            sample_size: Number of dialogue samples
            
        Returns:
            List of dialogue samples
            
        Raises:
            RuntimeError: If system is not loaded
        """
        if not self.is_loaded:
            raise RuntimeError("System not loaded. Call load_from_scripts() or load_from_processed_data() first.")
        
        return self.retriever.get_character_dialogue(character, sample_size)
    
    def get_system_stats(self) -> Dict:
        """Get statistics about the loaded system.
        
        Returns:
            Dictionary with system statistics
            
        Raises:
            RuntimeError: If system is not loaded
        """
        if not self.is_loaded:
            raise RuntimeError("System not loaded.")
        
        characters = self.retriever.get_available_characters()
        movies = self.retriever.get_available_movies()
        char_stats = self.retriever.get_character_stats()
        
        total_dialogue = sum(char_stats.values())
        
        return {
            'total_dialogue_lines': total_dialogue,
            'num_characters': len(characters),
            'num_movies': len(movies),
            'characters': characters,
            'movies': movies,
            'top_characters': dict(list(char_stats.items())[:10]),
            'embedding_model': self.embedder.model_name,
            'embedding_dimension': self.embedder.embedding_dim
        }
    
    def test_retrieval_quality(self, test_queries: Optional[List[str]] = None) -> Dict:
        """Test retrieval quality with sample queries.
        
        Args:
            test_queries: List of test queries (uses defaults if None)
            
        Returns:
            Dictionary with test results
            
        Raises:
            RuntimeError: If system is not loaded
        """
        if not self.is_loaded:
            raise RuntimeError("System not loaded.")
        
        if test_queries is None:
            test_queries = [
                "Tell me about the Force",
                "I need help with something",
                "The situation is dangerous",
                "Calculate the odds",
                "I have a bad feeling about this"
            ]
        
        results = {}
        total_found = 0
        
        for query in test_queries:
            dialogue_results = self.search_dialogue(query, top_k=3)
            results[query] = {
                'num_results': len(dialogue_results),
                'top_similarity': dialogue_results[0]['similarity'] if dialogue_results else 0,
                'results': dialogue_results
            }
            total_found += len(dialogue_results)
        
        return {
            'test_queries': test_queries,
            'total_results_found': total_found,
            'average_results_per_query': total_found / len(test_queries),
            'query_results': results
        }
