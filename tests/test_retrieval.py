"""
Tests for the DialogueRetriever module using real Star Wars data.
"""

import pytest
import pandas as pd
import numpy as np

from star_wars_rag.retrieval import DialogueRetriever
from star_wars_rag.embeddings import StarWarsEmbedder


class TestDialogueRetriever:
    """Test cases for DialogueRetriever class."""
    
    def test_init(self):
        """Test DialogueRetriever initialization."""
        retriever = DialogueRetriever()
        assert retriever.embedder is not None
        assert isinstance(retriever.embedder, StarWarsEmbedder)
        assert not retriever.is_ready()
        
        # Test with custom embedder
        custom_embedder = StarWarsEmbedder()
        custom_retriever = DialogueRetriever(custom_embedder)
        assert custom_retriever.embedder is custom_embedder
    
    def test_load_dialogue_data_with_mock_data(self, mock_dialogue_data, mock_embeddings):
        """Test loading dialogue data."""
        retriever = DialogueRetriever()
        
        retriever.load_dialogue_data(mock_dialogue_data, mock_embeddings)
        
        assert retriever.is_ready()
        assert retriever.dialogue_df is not None
        assert len(retriever.dialogue_df) == len(mock_dialogue_data)
        assert retriever.embeddings is not None
        assert retriever.embeddings.shape == mock_embeddings.shape
    
    def test_load_dialogue_data_missing_columns(self):
        """Test loading data with missing required columns."""
        retriever = DialogueRetriever()
        
        # Missing required columns
        bad_data = pd.DataFrame([{'some_column': 'value'}])
        
        with pytest.raises(ValueError, match="Missing required columns"):
            retriever.load_dialogue_data(bad_data)
    
    def test_load_dialogue_data_empty_dataframe(self):
        """Test loading empty DataFrame."""
        retriever = DialogueRetriever()
        
        empty_df = pd.DataFrame(columns=['dialogue_clean', 'character_normalized'])
        
        with pytest.raises(ValueError, match="Dialogue DataFrame is empty"):
            retriever.load_dialogue_data(empty_df)
    
    def test_load_dialogue_data_embeddings_length_mismatch(self, mock_dialogue_data):
        """Test loading data with mismatched embeddings length."""
        retriever = DialogueRetriever()
        
        wrong_size_embeddings = np.random.random((2, 384))  # Wrong size
        
        with pytest.raises(ValueError, match="Embeddings length"):
            retriever.load_dialogue_data(mock_dialogue_data, wrong_size_embeddings)
    
    @pytest.mark.slow
    def test_load_dialogue_data_compute_embeddings(self, mock_dialogue_data):
        """Test loading data and computing embeddings automatically."""
        retriever = DialogueRetriever()
        
        # Load without pre-computed embeddings
        retriever.load_dialogue_data(mock_dialogue_data)
        
        assert retriever.is_ready()
        assert retriever.embeddings is not None
        assert retriever.embeddings.shape[0] == len(mock_dialogue_data)
        assert retriever.embeddings.shape[1] == 384  # Expected dimension
    
    def test_retrieve_similar_dialogue_not_ready(self):
        """Test retrieval when not ready."""
        retriever = DialogueRetriever()
        
        with pytest.raises(RuntimeError, match="Retriever not ready"):
            retriever.retrieve_similar_dialogue("test query")
    
    def test_retrieve_similar_dialogue_basic(self, mock_dialogue_data, mock_embeddings):
        """Test basic dialogue retrieval."""
        retriever = DialogueRetriever()
        retriever.load_dialogue_data(mock_dialogue_data, mock_embeddings)
        
        results = retriever.retrieve_similar_dialogue("Force", top_k=2)
        
        assert isinstance(results, list)
        assert len(results) <= 2
        
        for result in results:
            assert 'similarity' in result
            assert 'character' in result
            assert 'dialogue' in result
            assert 'movie' in result
            assert 'scene' in result
            assert isinstance(result['similarity'], float)
    
    def test_retrieve_similar_dialogue_character_filter(self, mock_dialogue_data, mock_embeddings):
        """Test retrieval with character filter."""
        retriever = DialogueRetriever()
        retriever.load_dialogue_data(mock_dialogue_data, mock_embeddings)
        
        results = retriever.retrieve_similar_dialogue(
            "test query", 
            character_filter="Luke",
            top_k=5
        )
        
        # Should only return Luke's dialogue
        for result in results:
            assert "Luke" in result['character']
    
    def test_retrieve_similar_dialogue_movie_filter(self, mock_dialogue_data, mock_embeddings):
        """Test retrieval with movie filter."""
        retriever = DialogueRetriever()
        retriever.load_dialogue_data(mock_dialogue_data, mock_embeddings)
        
        results = retriever.retrieve_similar_dialogue(
            "test query",
            movie_filter="New Hope",
            top_k=5
        )
        
        # Should only return dialogue from A New Hope
        for result in results:
            assert "New Hope" in result['movie']
    
    def test_retrieve_similar_dialogue_min_similarity(self, mock_dialogue_data, mock_embeddings):
        """Test retrieval with minimum similarity threshold."""
        retriever = DialogueRetriever()
        retriever.load_dialogue_data(mock_dialogue_data, mock_embeddings)
        
        # Set high threshold - should return fewer or no results
        results = retriever.retrieve_similar_dialogue(
            "completely unrelated query about space dragons",
            min_similarity=0.9,
            top_k=5
        )
        
        # Should filter out low similarity results
        for result in results:
            assert result['similarity'] >= 0.9
    
    def test_retrieve_similar_dialogue_no_matches(self, mock_dialogue_data, mock_embeddings):
        """Test retrieval when no matches meet criteria."""
        retriever = DialogueRetriever()
        retriever.load_dialogue_data(mock_dialogue_data, mock_embeddings)
        
        # Filter that matches nothing
        results = retriever.retrieve_similar_dialogue(
            "test query",
            character_filter="Nonexistent Character"
        )
        
        assert results == []
    
    def test_get_character_dialogue(self, mock_dialogue_data, mock_embeddings):
        """Test getting character-specific dialogue."""
        retriever = DialogueRetriever()
        retriever.load_dialogue_data(mock_dialogue_data, mock_embeddings)
        
        luke_dialogue = retriever.get_character_dialogue("Luke")
        
        assert isinstance(luke_dialogue, list)
        assert len(luke_dialogue) > 0
        
        for item in luke_dialogue:
            assert 'character' in item
            assert 'dialogue' in item
            assert "Luke" in item['character']
    
    def test_get_character_dialogue_nonexistent(self, mock_dialogue_data, mock_embeddings):
        """Test getting dialogue for non-existent character."""
        retriever = DialogueRetriever()
        retriever.load_dialogue_data(mock_dialogue_data, mock_embeddings)
        
        results = retriever.get_character_dialogue("Jar Jar Binks")
        assert results == []
    
    def test_get_character_dialogue_sample_size(self, mock_dialogue_data, mock_embeddings):
        """Test getting character dialogue with sample size limit."""
        retriever = DialogueRetriever()
        retriever.load_dialogue_data(mock_dialogue_data, mock_embeddings)
        
        results = retriever.get_character_dialogue("Luke", sample_size=1)
        assert len(results) <= 1
    
    def test_get_available_characters(self, mock_dialogue_data, mock_embeddings):
        """Test getting available characters."""
        retriever = DialogueRetriever()
        retriever.load_dialogue_data(mock_dialogue_data, mock_embeddings)
        
        characters = retriever.get_available_characters()
        
        assert isinstance(characters, list)
        assert len(characters) > 0
        assert "Luke Skywalker" in characters
        assert "C-3PO" in characters
        assert "Darth Vader" in characters
    
    def test_get_available_characters_not_ready(self):
        """Test getting characters when not ready."""
        retriever = DialogueRetriever()
        
        characters = retriever.get_available_characters()
        assert characters == []
    
    def test_get_available_movies(self, mock_dialogue_data, mock_embeddings):
        """Test getting available movies."""
        retriever = DialogueRetriever()
        retriever.load_dialogue_data(mock_dialogue_data, mock_embeddings)
        
        movies = retriever.get_available_movies()
        
        assert isinstance(movies, list)
        assert len(movies) > 0
        assert "A New Hope" in movies
    
    def test_get_character_stats(self, mock_dialogue_data, mock_embeddings):
        """Test getting character statistics."""
        retriever = DialogueRetriever()
        retriever.load_dialogue_data(mock_dialogue_data, mock_embeddings)
        
        stats = retriever.get_character_stats()
        
        assert isinstance(stats, dict)
        assert len(stats) > 0
        
        for character, count in stats.items():
            assert isinstance(character, str)
            assert isinstance(count, (int, np.integer))
            assert count > 0
    
    def test_search_dialogue_by_content(self, mock_dialogue_data, mock_embeddings):
        """Test text-based dialogue search."""
        retriever = DialogueRetriever()
        retriever.load_dialogue_data(mock_dialogue_data, mock_embeddings)
        
        # Search for "Force" in dialogue
        results = retriever.search_dialogue_by_content("Force")
        
        assert isinstance(results, list)
        
        for result in results:
            assert 'character' in result
            assert 'dialogue' in result
            assert "Force" in result['dialogue']
    
    def test_search_dialogue_by_content_with_character_filter(self, mock_dialogue_data, mock_embeddings):
        """Test content search with character filter."""
        retriever = DialogueRetriever()
        retriever.load_dialogue_data(mock_dialogue_data, mock_embeddings)
        
        results = retriever.search_dialogue_by_content("want", character_filter="Luke")
        
        for result in results:
            assert "want" in result['dialogue'].lower()
            assert "Luke" in result['character']
    
    def test_search_dialogue_by_content_not_ready(self):
        """Test content search when not ready."""
        retriever = DialogueRetriever()
        
        with pytest.raises(RuntimeError, match="Retriever not ready"):
            retriever.search_dialogue_by_content("test")


class TestDialogueRetrieverIntegration:
    """Integration tests for DialogueRetriever using real data."""
    
    @pytest.mark.integration
    @pytest.mark.real_data
    @pytest.mark.slow
    def test_retrieval_with_real_data(self, retriever_with_data):
        """Test retrieval with real Star Wars dialogue data."""
        if not retriever_with_data.is_ready():
            pytest.skip("No real data available for testing")
        
        # Test Force-related query
        results = retriever_with_data.retrieve_similar_dialogue(
            "Tell me about the Force",
            top_k=3
        )
        
        assert len(results) > 0
        assert len(results) <= 3
        
        # Check result structure
        for result in results:
            assert 'similarity' in result
            assert 'character' in result
            assert 'dialogue' in result
            assert result['similarity'] > 0
        
        # Results should be sorted by similarity
        similarities = [r['similarity'] for r in results]
        assert similarities == sorted(similarities, reverse=True)
        
        print(f"Found {len(results)} Force-related results:")
        for i, result in enumerate(results):
            print(f"{i+1}. [{result['similarity']:.3f}] {result['character']}: {result['dialogue'][:60]}...")
    
    @pytest.mark.integration
    @pytest.mark.real_data
    def test_character_specific_retrieval_real_data(self, retriever_with_data):
        """Test character-specific retrieval with real data."""
        if not retriever_with_data.is_ready():
            pytest.skip("No real data available for testing")
        
        # Get available characters
        characters = retriever_with_data.get_available_characters()
        if not characters:
            pytest.skip("No characters available in test data")
        
        # Test with first available character
        test_character = characters[0]
        
        # Get character's dialogue
        char_dialogue = retriever_with_data.get_character_dialogue(test_character, sample_size=3)
        
        assert len(char_dialogue) > 0
        
        for item in char_dialogue:
            assert item['character'] == test_character
            assert len(item['dialogue']) > 0
        
        print(f"Found {len(char_dialogue)} dialogue lines for {test_character}")
        
        # Test retrieval filtered by character
        results = retriever_with_data.retrieve_similar_dialogue(
            "What do you think?",
            character_filter=test_character,
            top_k=2
        )
        
        # Should only return this character's dialogue
        for result in results:
            assert result['character'] == test_character
    
    @pytest.mark.integration
    @pytest.mark.real_data 
    def test_semantic_retrieval_quality_real_data(self, retriever_with_data):
        """Test semantic retrieval quality with real data."""
        if not retriever_with_data.is_ready():
            pytest.skip("No real data available for testing")
        
        # Test queries with expected semantic matches
        test_queries = [
            ("learning about power", ["Force", "power", "strong"]),
            ("seeking help", ["help", "please", "need"]),
            ("expressing fear", ["afraid", "scared", "worry", "danger"])
        ]
        
        for query, expected_keywords in test_queries:
            results = retriever_with_data.retrieve_similar_dialogue(query, top_k=3)
            
            if results:  # Only test if we got results
                # Check if any results contain expected keywords
                found_relevant = False
                for result in results:
                    dialogue_lower = result['dialogue'].lower()
                    if any(keyword in dialogue_lower for keyword in expected_keywords):
                        found_relevant = True
                        break
                
                print(f"Query: '{query}' -> Found relevant: {found_relevant}")
                if results:
                    print(f"  Top result: {results[0]['dialogue'][:60]}...")
    
    @pytest.mark.integration
    @pytest.mark.real_data
    def test_retrieval_statistics_real_data(self, retriever_with_data):
        """Test retrieval statistics with real data."""
        if not retriever_with_data.is_ready():
            pytest.skip("No real data available for testing")
        
        # Get system statistics
        characters = retriever_with_data.get_available_characters()
        movies = retriever_with_data.get_available_movies()
        char_stats = retriever_with_data.get_character_stats()
        
        assert len(characters) > 0
        assert len(movies) > 0
        assert len(char_stats) > 0
        
        # Check statistics consistency
        assert len(characters) == len(char_stats)
        
        # Character stats should be positive
        for character, count in char_stats.items():
            assert count > 0
            assert character in characters
        
        print(f"System loaded with:")
        print(f"  Characters: {len(characters)}")
        print(f"  Movies: {movies}")
        print(f"  Top characters: {dict(list(char_stats.items())[:5])}")
    
    @pytest.mark.integration
    @pytest.mark.real_data
    @pytest.mark.slow
    def test_end_to_end_retrieval_pipeline(self, sample_dialogue_data):
        """Test complete end-to-end retrieval pipeline."""
        if sample_dialogue_data.empty:
            pytest.skip("No real dialogue data available")
        
        # Create fresh retriever and load data
        retriever = DialogueRetriever()
        
        # Use subset for faster testing
        test_data = sample_dialogue_data.head(30)
        retriever.load_dialogue_data(test_data)
        
        assert retriever.is_ready()
        
        # Test various retrieval scenarios
        scenarios = [
            "Tell me about hope",
            "I need help with something",
            "The situation is dangerous"
        ]
        
        for query in scenarios:
            # Basic retrieval
            results = retriever.retrieve_similar_dialogue(query, top_k=3)
            
            assert isinstance(results, list)
            assert len(results) <= 3
            
            # Test with filters if characters available
            characters = retriever.get_available_characters()
            if characters:
                char_results = retriever.retrieve_similar_dialogue(
                    query,
                    character_filter=characters[0],
                    top_k=2
                )
                
                for result in char_results:
                    assert result['character'] == characters[0]
            
            # Content search
            content_results = retriever.search_dialogue_by_content(query.split()[-1])  # Last word
            
            assert isinstance(content_results, list)
            
            print(f"Query '{query}': {len(results)} semantic, {len(content_results)} content matches")
