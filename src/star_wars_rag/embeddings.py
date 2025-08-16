"""
Embedding module for Star Wars dialogue using sentence transformers.

This module handles text embedding generation for the RAG system.
"""

import numpy as np
from typing import List, Union, Optional
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class StarWarsEmbedder:
    """Embedding utility class for Star Wars dialogue."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedder.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
    def load_model(self) -> SentenceTransformer:
        """Load the embedding model.
        
        Returns:
            Loaded SentenceTransformer model
            
        Raises:
            RuntimeError: If model loading fails
        """
        if self.model is None:
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Model loaded successfully on device: {self.model.device}")
            except Exception as e:
                logger.error(f"Failed to load model {self.model_name}: {e}")
                raise RuntimeError(f"Model loading failed: {e}")
        
        return self.model
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
            
        Raises:
            ValueError: If text is empty
        """
        if not text.strip():
            raise ValueError("Cannot embed empty text")
            
        if self.model is None:
            self.load_model()
        
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        return embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 32, 
                   show_progress: bool = True) -> np.ndarray:
        """Embed a batch of texts efficiently.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings with shape (n_texts, embedding_dim)
            
        Raises:
            ValueError: If texts list is empty
        """
        if not texts:
            raise ValueError("Cannot embed empty text list")
        
        # Filter out empty texts and track indices
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text.strip())
                valid_indices.append(i)
        
        if not valid_texts:
            raise ValueError("No valid texts to embed")
        
        if self.model is None:
            self.load_model()
        
        logger.info(f"Embedding {len(valid_texts)} texts in batches of {batch_size}")
        
        # Process in batches to manage memory
        embeddings = []
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch, 
                convert_to_numpy=True,
                show_progress_bar=show_progress and i == 0
            )
            embeddings.extend(batch_embeddings)
        
        result = np.array(embeddings)
        logger.info(f"Generated embeddings with shape: {result.shape}")
        return result
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                          stored_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and stored embeddings.
        
        Args:
            query_embedding: Single query embedding
            stored_embeddings: Array of stored embeddings
            
        Returns:
            Array of similarity scores
            
        Raises:
            ValueError: If embeddings have wrong shape or dimension mismatch
        """
        # Reshape if needed
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if len(stored_embeddings.shape) == 1:
            stored_embeddings = stored_embeddings.reshape(1, -1)
        
        # Validate dimensions
        if query_embedding.shape[1] != stored_embeddings.shape[1]:
            raise ValueError(
                f"Dimension mismatch: query {query_embedding.shape[1]} "
                f"vs stored {stored_embeddings.shape[1]}"
            )
        
        similarities = cosine_similarity(query_embedding, stored_embeddings)[0]
        return similarities
    
    def find_most_similar(self, query: str, texts: List[str], 
                         embeddings: Optional[np.ndarray] = None,
                         top_k: int = 5) -> List[dict]:
        """Find most similar texts to a query.
        
        Args:
            query: Query text
            texts: List of candidate texts
            embeddings: Pre-computed embeddings (computed if None)
            top_k: Number of results to return
            
        Returns:
            List of dictionaries with similarity scores and texts
        """
        if not texts:
            return []
        
        # Compute embeddings if not provided
        if embeddings is None:
            embeddings = self.embed_batch(texts, show_progress=False)
        
        # Embed query
        query_embedding = self.embed_text(query)
        
        # Compute similarities
        similarities = self.compute_similarity(query_embedding, embeddings)
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'index': int(idx),
                'text': texts[idx],
                'similarity': float(similarities[idx])
            })
        
        return results
    
    def validate_embeddings(self, embeddings: np.ndarray) -> dict:
        """Validate embedding array and return statistics.
        
        Args:
            embeddings: Array of embeddings to validate
            
        Returns:
            Dictionary with validation statistics
        """
        stats = {
            'shape': embeddings.shape,
            'dtype': str(embeddings.dtype),
            'memory_mb': embeddings.nbytes / (1024 * 1024),
            'has_nan': bool(np.isnan(embeddings).any()),
            'has_inf': bool(np.isinf(embeddings).any()),
            'mean_norm': float(np.mean(np.linalg.norm(embeddings, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(embeddings, axis=1)))
        }
        
        if stats['has_nan'] or stats['has_inf']:
            logger.warning("Embeddings contain NaN or Inf values")
        
        logger.info(f"Embedding validation: {stats}")
        return stats
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str) -> None:
        """Save embeddings to file.
        
        Args:
            embeddings: Embeddings array to save
            filepath: Path to save file
        """
        try:
            np.save(filepath, embeddings)
            logger.info(f"Saved embeddings to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            raise
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Load embeddings from file.
        
        Args:
            filepath: Path to embeddings file
            
        Returns:
            Loaded embeddings array
        """
        try:
            embeddings = np.load(filepath)
            logger.info(f"Loaded embeddings from {filepath}: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise
