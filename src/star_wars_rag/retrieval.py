"""
Retrieval module for Star Wars dialogue RAG system.

This module handles dialogue retrieval based on semantic similarity.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
import logging
from star_wars_rag.embeddings import StarWarsEmbedder

logger = logging.getLogger(__name__)


class DialogueRetriever:
    """Retrieve relevant Star Wars dialogue based on queries."""
    
    def __init__(self, embedder: Optional[StarWarsEmbedder] = None):
        """Initialize the dialogue retriever.
        
        Args:
            embedder: Embedder instance (creates new one if None)
        """
        self.embedder = embedder or StarWarsEmbedder()
        self.dialogue_df: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        self._is_ready = False
    
    def load_dialogue_data(self, dialogue_df: pd.DataFrame, 
                          embeddings: Optional[np.ndarray] = None) -> None:
        """Load dialogue data and embeddings.
        
        Args:
            dialogue_df: DataFrame with dialogue data
            embeddings: Pre-computed embeddings (computed if None)
            
        Raises:
            ValueError: If dialogue_df is empty or missing required columns
        """
        required_columns = ['dialogue_clean', 'character_normalized']
        missing_columns = [col for col in required_columns if col not in dialogue_df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if dialogue_df.empty:
            raise ValueError("Dialogue DataFrame is empty")
        
        self.dialogue_df = dialogue_df.copy()
        
        # Compute embeddings if not provided
        if embeddings is None:
            logger.info("Computing embeddings for dialogue data...")
            texts = self.dialogue_df['dialogue_clean'].tolist()
            self.embeddings = self.embedder.embed_batch(texts)
        else:
            if len(embeddings) != len(dialogue_df):
                raise ValueError(
                    f"Embeddings length ({len(embeddings)}) doesn't match "
                    f"dialogue data length ({len(dialogue_df)})"
                )
            self.embeddings = embeddings
        
        self._is_ready = True
        logger.info(f"Loaded {len(self.dialogue_df)} dialogue entries with embeddings")
    
    def retrieve_similar_dialogue(self, query: str, top_k: int = 5,
                                 character_filter: Optional[str] = None,
                                 movie_filter: Optional[str] = None,
                                 min_similarity: float = 0.0) -> List[Dict]:
        """Retrieve dialogue similar to the query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            character_filter: Filter by specific character
            movie_filter: Filter by specific movie
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of dictionaries with dialogue and metadata
            
        Raises:
            RuntimeError: If retriever is not ready
        """
        if not self._is_ready:
            raise RuntimeError("Retriever not ready. Call load_dialogue_data() first.")
        
        # Apply filters
        filtered_df, filtered_embeddings = self._apply_filters(
            character_filter, movie_filter
        )
        
        if filtered_df.empty:
            logger.warning("No dialogue found matching filters")
            return []
        
        # Embed query and compute similarities
        query_embedding = self.embedder.embed_text(query)
        similarities = self.embedder.compute_similarity(
            query_embedding, filtered_embeddings
        )
        
        # Filter by minimum similarity
        valid_indices = similarities >= min_similarity
        if not valid_indices.any():
            logger.warning(f"No results above similarity threshold {min_similarity}")
            return []
        
        # Get top-k results
        valid_similarities = similarities[valid_indices]
        valid_df = filtered_df[valid_indices].reset_index(drop=True)
        
        top_indices = np.argsort(valid_similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            row = valid_df.iloc[idx]
            results.append({
                'similarity': float(valid_similarities[idx]),
                'character': row['character_normalized'],
                'dialogue': row['dialogue_clean'],
                'movie': row.get('movie', 'Unknown'),
                'scene': row.get('scene', 'Unknown'),
                'original_index': row.get('line_number', idx)
            })
        
        logger.debug(f"Retrieved {len(results)} results for query: '{query[:50]}...'")
        return results
    
    def _apply_filters(self, character_filter: Optional[str] = None,
                      movie_filter: Optional[str] = None) -> tuple:
        """Apply character and movie filters to data.
        
        Args:
            character_filter: Character name to filter by
            movie_filter: Movie name to filter by
            
        Returns:
            Tuple of (filtered_df, filtered_embeddings)
        """
        mask = pd.Series([True] * len(self.dialogue_df))
        
        if character_filter:
            char_mask = self.dialogue_df['character_normalized'].str.contains(
                character_filter, case=False, na=False
            )
            mask &= char_mask
            
        if movie_filter:
            movie_mask = self.dialogue_df['movie'].str.contains(
                movie_filter, case=False, na=False
            )
            mask &= movie_mask
        
        filtered_df = self.dialogue_df[mask].reset_index(drop=True)
        filtered_embeddings = self.embeddings[mask]
        
        return filtered_df, filtered_embeddings
    
    def get_character_dialogue(self, character: str, 
                              sample_size: Optional[int] = None) -> List[Dict]:
        """Get dialogue for a specific character.
        
        Args:
            character: Character name
            sample_size: Maximum number of samples (all if None)
            
        Returns:
            List of dialogue dictionaries for the character
        """
        if not self._is_ready:
            raise RuntimeError("Retriever not ready. Call load_dialogue_data() first.")
        
        char_mask = self.dialogue_df['character_normalized'].str.contains(
            character, case=False, na=False
        )
        char_df = self.dialogue_df[char_mask]
        
        if char_df.empty:
            return []
        
        if sample_size and len(char_df) > sample_size:
            char_df = char_df.sample(n=sample_size, random_state=42)
        
        results = []
        for _, row in char_df.iterrows():
            results.append({
                'character': row['character_normalized'],
                'dialogue': row['dialogue_clean'],
                'movie': row.get('movie', 'Unknown'),
                'scene': row.get('scene', 'Unknown')
            })
        
        return results
    
    def get_available_characters(self) -> List[str]:
        """Get list of available characters.
        
        Returns:
            Sorted list of character names
        """
        if not self._is_ready:
            return []
        
        return sorted(self.dialogue_df['character_normalized'].unique())
    
    def get_available_movies(self) -> List[str]:
        """Get list of available movies.
        
        Returns:
            Sorted list of movie names
        """
        if not self._is_ready:
            return []
        
        return sorted(self.dialogue_df['movie'].unique())
    
    def get_character_stats(self) -> Dict[str, int]:
        """Get dialogue count statistics by character.
        
        Returns:
            Dictionary mapping character names to dialogue counts
        """
        if not self._is_ready:
            return {}
        
        return self.dialogue_df['character_normalized'].value_counts().to_dict()
    
    def search_dialogue_by_content(self, search_term: str,
                                  character_filter: Optional[str] = None) -> List[Dict]:
        """Search dialogue by text content (not semantic similarity).
        
        Args:
            search_term: Text to search for
            character_filter: Optional character filter
            
        Returns:
            List of matching dialogue entries
        """
        if not self._is_ready:
            raise RuntimeError("Retriever not ready. Call load_dialogue_data() first.")
        
        # Text search mask
        text_mask = self.dialogue_df['dialogue_clean'].str.contains(
            search_term, case=False, na=False
        )
        
        # Character filter if specified
        if character_filter:
            char_mask = self.dialogue_df['character_normalized'].str.contains(
                character_filter, case=False, na=False
            )
            text_mask &= char_mask
        
        results_df = self.dialogue_df[text_mask]
        
        results = []
        for _, row in results_df.iterrows():
            results.append({
                'character': row['character_normalized'],
                'dialogue': row['dialogue_clean'],
                'movie': row.get('movie', 'Unknown'),
                'scene': row.get('scene', 'Unknown')
            })
        
        return results
    
    def is_ready(self) -> bool:
        """Check if retriever is ready for use.
        
        Returns:
            True if ready for retrieval operations
        """
        return self._is_ready
