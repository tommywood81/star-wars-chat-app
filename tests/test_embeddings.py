"""
Tests for the StarWarsEmbedder module using real dialogue data.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from star_wars_rag.embeddings import StarWarsEmbedder


class TestStarWarsEmbedder:
    """Test cases for StarWarsEmbedder class."""
    
    def test_init(self):
        """Test StarWarsEmbedder initialization."""
        embedder = StarWarsEmbedder()
        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder.model is None  # Not loaded yet
        assert embedder.embedding_dim == 384
        
        # Test custom model name
        custom_embedder = StarWarsEmbedder("different-model")
        assert custom_embedder.model_name == "different-model"
    
    @pytest.mark.slow
    def test_load_model(self):
        """Test model loading."""
        embedder = StarWarsEmbedder()
        
        model = embedder.load_model()
        
        assert model is not None
        assert embedder.model is not None
        assert embedder.model == model
        
        # Test that loading again returns the same model
        model2 = embedder.load_model()
        assert model is model2
    
    @pytest.mark.slow
    def test_embed_text(self):
        """Test single text embedding."""
        embedder = StarWarsEmbedder()
        
        # Test with Star Wars quote
        text = "May the Force be with you."
        embedding = embedder.embed_text(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)  # Expected dimension
        assert embedding.dtype == np.float32
        
        # Test that embeddings are normalized (approximately)
        norm = np.linalg.norm(embedding)
        assert 0.8 < norm < 1.2  # Should be approximately unit norm
        
        # Test different text produces different embedding
        different_text = "I find your lack of faith disturbing."
        different_embedding = embedder.embed_text(different_text)
        
        # Embeddings should be different
        similarity = np.dot(embedding, different_embedding)
        assert similarity < 0.9  # Not too similar
    
    def test_embed_text_empty_string(self):
        """Test embedding empty string raises error."""
        embedder = StarWarsEmbedder()
        
        with pytest.raises(ValueError, match="Cannot embed empty text"):
            embedder.embed_text("")
        
        with pytest.raises(ValueError, match="Cannot embed empty text"):
            embedder.embed_text("   ")  # Just whitespace
    
    @pytest.mark.slow
    def test_embed_batch(self):
        """Test batch embedding."""
        embedder = StarWarsEmbedder()
        
        # Star Wars quotes for testing
        texts = [
            "May the Force be with you.",
            "I am your father.",
            "Help me, Obi-Wan Kenobi.",
            "The Force will be with you, always.",
            "I find your lack of faith disturbing."
        ]
        
        embeddings = embedder.embed_batch(texts, show_progress=False)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (len(texts), 384)
        assert embeddings.dtype == np.float32
        
        # Test that different texts have different embeddings
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = np.dot(embeddings[i], embeddings[j])
                assert similarity < 0.95  # Should not be too similar
    
    @pytest.mark.slow
    def test_embed_batch_with_empty_texts(self):
        """Test batch embedding with some empty texts."""
        embedder = StarWarsEmbedder()
        
        texts = [
            "Valid text here",
            "",  # Empty
            "Another valid text",
            "   ",  # Just whitespace
            "Third valid text"
        ]
        
        # Should filter out empty texts and still work
        embeddings = embedder.embed_batch(texts, show_progress=False)
        
        # Should have embeddings for 3 valid texts
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == 384
    
    def test_embed_batch_empty_list(self):
        """Test batch embedding with empty list."""
        embedder = StarWarsEmbedder()
        
        with pytest.raises(ValueError, match="Cannot embed empty text list"):
            embedder.embed_batch([])
        
        with pytest.raises(ValueError, match="No valid texts to embed"):
            embedder.embed_batch(["", "   ", ""])
    
    @pytest.mark.slow
    def test_compute_similarity(self):
        """Test similarity computation."""
        embedder = StarWarsEmbedder()
        
        # Create test embeddings
        text1 = "The Force is strong with this one."
        text2 = "The Force is powerful in you."
        text3 = "I hate sand. It's coarse and rough."
        
        emb1 = embedder.embed_text(text1)
        emb2 = embedder.embed_text(text2)
        emb3 = embedder.embed_text(text3)
        
        stored_embeddings = np.array([emb2, emb3])
        
        similarities = embedder.compute_similarity(emb1, stored_embeddings)
        
        assert len(similarities) == 2
        assert isinstance(similarities, np.ndarray)
        
        # Force-related texts should be more similar
        assert similarities[0] > similarities[1]
        assert similarities[0] > 0.5  # Should be reasonably similar
        
    def test_compute_similarity_dimension_mismatch(self):
        """Test similarity computation with mismatched dimensions."""
        embedder = StarWarsEmbedder()
        
        query_emb = np.random.random((5,))  # Wrong dimension
        stored_embs = np.random.random((3, 384))  # Correct dimension
        
        with pytest.raises(ValueError, match="Dimension mismatch"):
            embedder.compute_similarity(query_emb, stored_embs)
    
    @pytest.mark.slow
    def test_find_most_similar(self):
        """Test finding most similar texts."""
        embedder = StarWarsEmbedder()
        
        texts = [
            "Luke, I am your father.",
            "The Force will be with you.",
            "I find your lack of faith disturbing.",
            "Help me, Obi-Wan Kenobi.",
            "Never tell me the odds!"
        ]
        
        query = "May the Force guide you."
        
        results = embedder.find_most_similar(query, texts, top_k=3)
        
        assert len(results) == 3
        
        for result in results:
            assert 'index' in result
            assert 'text' in result
            assert 'similarity' in result
            assert isinstance(result['similarity'], float)
            assert 0 <= result['similarity'] <= 1
        
        # Results should be sorted by similarity (highest first)
        similarities = [r['similarity'] for r in results]
        assert similarities == sorted(similarities, reverse=True)
        
        # Force-related text should be most similar
        most_similar_text = results[0]['text']
        assert "Force" in most_similar_text
    
    def test_find_most_similar_empty_texts(self):
        """Test finding similar texts with empty list."""
        embedder = StarWarsEmbedder()
        
        results = embedder.find_most_similar("test query", [])
        assert results == []
    
    @pytest.mark.slow
    def test_validate_embeddings(self):
        """Test embedding validation."""
        embedder = StarWarsEmbedder()
        
        # Create test embeddings
        texts = ["Test text 1", "Test text 2", "Test text 3"]
        embeddings = embedder.embed_batch(texts, show_progress=False)
        
        stats = embedder.validate_embeddings(embeddings)
        
        assert 'shape' in stats
        assert 'dtype' in stats
        assert 'memory_mb' in stats
        assert 'has_nan' in stats
        assert 'has_inf' in stats
        assert 'mean_norm' in stats
        assert 'std_norm' in stats
        
        assert stats['shape'] == (3, 384)
        assert not stats['has_nan']
        assert not stats['has_inf']
        assert stats['mean_norm'] > 0
    
    def test_validate_embeddings_with_invalid_data(self):
        """Test validation with invalid embedding data."""
        embedder = StarWarsEmbedder()
        
        # Create embeddings with NaN values
        bad_embeddings = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]])
        
        stats = embedder.validate_embeddings(bad_embeddings)
        
        assert stats['has_nan']
        assert not stats['has_inf']
        
        # Create embeddings with Inf values
        bad_embeddings = np.array([[1.0, 2.0, np.inf], [4.0, 5.0, 6.0]])
        
        stats = embedder.validate_embeddings(bad_embeddings)
        
        assert not stats['has_nan']
        assert stats['has_inf']
    
    def test_save_and_load_embeddings(self, tmp_path):
        """Test saving and loading embeddings."""
        embedder = StarWarsEmbedder()
        
        # Create test embeddings
        test_embeddings = np.random.random((5, 384)).astype(np.float32)
        
        # Save embeddings
        save_path = tmp_path / "test_embeddings.npy"
        embedder.save_embeddings(test_embeddings, str(save_path))
        
        assert save_path.exists()
        
        # Load embeddings
        loaded_embeddings = embedder.load_embeddings(str(save_path))
        
        assert isinstance(loaded_embeddings, np.ndarray)
        assert loaded_embeddings.shape == test_embeddings.shape
        np.testing.assert_array_equal(loaded_embeddings, test_embeddings)
    
    def test_load_embeddings_nonexistent_file(self):
        """Test loading from non-existent file."""
        embedder = StarWarsEmbedder()
        
        with pytest.raises(Exception):  # Could be FileNotFoundError or others
            embedder.load_embeddings("nonexistent_file.npy")


class TestStarWarsEmbedderIntegration:
    """Integration tests for StarWarsEmbedder using real dialogue."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.real_data
    def test_embed_real_dialogue(self, sample_dialogue_data):
        """Test embedding real Star Wars dialogue."""
        if sample_dialogue_data.empty:
            pytest.skip("No real dialogue data available")
        
        embedder = StarWarsEmbedder()
        
        # Take a sample of dialogue for speed
        sample_texts = sample_dialogue_data['dialogue_clean'].head(20).tolist()
        
        embeddings = embedder.embed_batch(sample_texts, show_progress=False)
        
        # Validate results
        assert embeddings.shape == (len(sample_texts), 384)
        
        # Check embedding quality
        stats = embedder.validate_embeddings(embeddings)
        assert not stats['has_nan']
        assert not stats['has_inf']
        assert stats['mean_norm'] > 0.8  # Should be reasonably normalized
        
        print(f"Embedded {len(sample_texts)} real dialogue lines")
        print(f"Embedding stats: {stats}")
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_semantic_similarity_with_star_wars_content(self):
        """Test that semantically similar Star Wars content has high similarity."""
        embedder = StarWarsEmbedder()
        
        # Test similar concepts
        force_texts = [
            "The Force is strong with you.",
            "May the Force be with you.",
            "Use the Force, Luke."
        ]
        
        different_text = "The Death Star plans are not in the main computer."
        
        force_embeddings = embedder.embed_batch(force_texts, show_progress=False)
        different_embedding = embedder.embed_text(different_text)
        
        # Force-related texts should be similar to each other
        for i in range(len(force_texts)):
            for j in range(i + 1, len(force_texts)):
                similarity = np.dot(force_embeddings[i], force_embeddings[j])
                assert similarity > 0.7, f"Force texts should be similar: {similarity}"
        
        # Force texts should be less similar to different content
        for force_emb in force_embeddings:
            similarity = np.dot(force_emb, different_embedding)
            assert similarity < 0.6, f"Different topics should be less similar: {similarity}"
    
    @pytest.mark.integration
    @pytest.mark.slow 
    def test_character_specific_similarity(self):
        """Test that character-specific dialogue shows patterns."""
        embedder = StarWarsEmbedder()
        
        # Vader's characteristic phrases
        vader_texts = [
            "I find your lack of faith disturbing.",
            "You have failed me for the last time.",
            "The Force is strong with this one."
        ]
        
        # C-3PO's characteristic phrases  
        threepio_texts = [
            "The odds of successfully navigating an asteroid field are approximately 3,720 to 1.",
            "I am C-3PO, human-cyborg relations.",
            "We're doomed!"
        ]
        
        vader_embeddings = embedder.embed_batch(vader_texts, show_progress=False)
        threepio_embeddings = embedder.embed_batch(threepio_texts, show_progress=False)
        
        # Calculate average embeddings for each character
        vader_avg = np.mean(vader_embeddings, axis=0)
        threepio_avg = np.mean(threepio_embeddings, axis=0)
        
        # Characters should have different styles
        character_similarity = np.dot(vader_avg, threepio_avg)
        assert character_similarity < 0.8, "Different characters should have different styles"
        
        # Test a new Vader-like phrase
        new_vader_text = "Your lack of vision disappoints me."
        new_vader_emb = embedder.embed_text(new_vader_text)
        
        vader_sim = np.dot(new_vader_emb, vader_avg)
        threepio_sim = np.dot(new_vader_emb, threepio_avg)
        
        # Should be more similar to Vader than C-3PO
        assert vader_sim > threepio_sim, "Vader-like text should be more similar to Vader"
