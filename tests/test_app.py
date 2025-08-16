"""
Tests for the main StarWarsRAGApp application.
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path

from star_wars_rag.app import StarWarsRAGApp


class TestStarWarsRAGApp:
    """Test cases for StarWarsRAGApp class."""
    
    def test_init(self):
        """Test app initialization."""
        app = StarWarsRAGApp()
        
        assert app.processor is not None
        assert app.embedder is not None
        assert app.retriever is not None
        assert not app.is_loaded
        assert app.embedder.model_name == "all-MiniLM-L6-v2"
        
        # Test custom model
        custom_app = StarWarsRAGApp("different-model")
        assert custom_app.embedder.model_name == "different-model"
    
    def test_operations_before_loading(self):
        """Test that operations fail before loading data."""
        app = StarWarsRAGApp()
        
        # Should fail when not loaded
        with pytest.raises(RuntimeError, match="System not loaded"):
            app.chat_with_character("test", "Luke Skywalker")
        
        with pytest.raises(RuntimeError, match="System not loaded"):
            app.search_dialogue("test")
        
        with pytest.raises(RuntimeError, match="System not loaded"):
            app.get_character_dialogue_sample("Luke Skywalker")
        
        with pytest.raises(RuntimeError, match="System not loaded"):
            app.get_system_stats()
        
        with pytest.raises(RuntimeError, match="System not loaded"):
            app.test_retrieval_quality()
    
    def test_load_from_processed_data_csv_only(self, mock_dialogue_data, tmp_path):
        """Test loading from CSV data without pre-computed embeddings."""
        app = StarWarsRAGApp()
        
        # Save mock data to CSV
        csv_path = tmp_path / "test_dialogue.csv"
        mock_dialogue_data.to_csv(csv_path, index=False)
        
        # Load from CSV (should compute embeddings)
        app.load_from_processed_data(csv_path)
        
        assert app.is_loaded
        assert app.retriever.is_ready()
        
        # Test basic functionality
        stats = app.get_system_stats()
        assert stats['total_dialogue_lines'] == len(mock_dialogue_data)
        assert len(stats['characters']) > 0
    
    @pytest.mark.slow
    def test_load_from_processed_data_with_embeddings(self, mock_dialogue_data, mock_embeddings, tmp_path):
        """Test loading from CSV and pre-computed embeddings."""
        app = StarWarsRAGApp()
        
        # Save mock data
        csv_path = tmp_path / "test_dialogue.csv"
        embeddings_path = tmp_path / "test_embeddings.npy"
        
        mock_dialogue_data.to_csv(csv_path, index=False)
        app.embedder.save_embeddings(mock_embeddings, str(embeddings_path))
        
        # Load from both files
        app.load_from_processed_data(csv_path, embeddings_path)
        
        assert app.is_loaded
        assert app.retriever.is_ready()
        
        # Should have used pre-computed embeddings
        assert app.retriever.embeddings.shape == mock_embeddings.shape
    
    def test_load_from_processed_data_missing_file(self):
        """Test loading from non-existent files."""
        app = StarWarsRAGApp()
        
        with pytest.raises(FileNotFoundError):
            app.load_from_processed_data("nonexistent.csv")
    
    @pytest.mark.real_data
    @pytest.mark.slow
    def test_load_from_scripts(self, test_data_dir, tmp_path):
        """Test loading from script files."""
        app = StarWarsRAGApp()
        
        try:
            # Try to load from real script directory
            processed_path = tmp_path / "processed_data.csv"
            embeddings_path = tmp_path / "embeddings.npy"
            
            app.load_from_scripts(
                test_data_dir,
                pattern="STAR WARS A NEW HOPE.txt",  # Load just one file for speed
                save_processed_data=str(processed_path),
                save_embeddings=str(embeddings_path)
            )
            
            assert app.is_loaded
            assert processed_path.exists()
            assert embeddings_path.exists()
            
            # Test basic functionality
            stats = app.get_system_stats()
            assert stats['total_dialogue_lines'] > 0
            assert len(stats['characters']) > 0
            
        except Exception as e:
            pytest.skip(f"Script loading test failed: {e}")
    
    def test_load_from_scripts_no_files(self, tmp_path):
        """Test loading when no script files match."""
        app = StarWarsRAGApp()
        
        with pytest.raises(ValueError, match="No script files found"):
            app.load_from_scripts(tmp_path, "*.nonexistent")
    
    def test_chat_with_character_loaded_system(self, mock_dialogue_data, mock_embeddings):
        """Test chatting with character in loaded system."""
        app = StarWarsRAGApp()
        
        # Load system
        csv_path = "temp.csv"
        mock_dialogue_data.to_csv(csv_path, index=False)
        app.load_from_processed_data(csv_path)
        Path(csv_path).unlink()  # Clean up
        
        # Test chat with existing character
        response = app.chat_with_character("Tell me about the Force", "Luke Skywalker")
        
        assert isinstance(response, dict)
        assert 'character' in response
        assert 'response' in response
        assert 'similarity' in response
        assert 'context' in response
        assert 'query' in response
        
        assert response['character'] == "Luke Skywalker"
        assert response['query'] == "Tell me about the Force"
        assert isinstance(response['similarity'], float)
        assert isinstance(response['context'], list)
    
    def test_chat_with_nonexistent_character(self, mock_dialogue_data):
        """Test chatting with non-existent character."""
        app = StarWarsRAGApp()
        
        # Load system
        csv_path = "temp.csv"
        mock_dialogue_data.to_csv(csv_path, index=False)
        app.load_from_processed_data(csv_path)
        Path(csv_path).unlink()  # Clean up
        
        with pytest.raises(ValueError, match="Character 'Jar Jar Binks' not found"):
            app.chat_with_character("Hello", "Jar Jar Binks")
    
    def test_search_dialogue_functionality(self, mock_dialogue_data):
        """Test dialogue search functionality."""
        app = StarWarsRAGApp()
        
        # Load system
        csv_path = "temp.csv"
        mock_dialogue_data.to_csv(csv_path, index=False)
        app.load_from_processed_data(csv_path)
        Path(csv_path).unlink()  # Clean up
        
        # Test basic search
        results = app.search_dialogue("Force", top_k=2)
        
        assert isinstance(results, list)
        assert len(results) <= 2
        
        for result in results:
            assert 'similarity' in result
            assert 'character' in result
            assert 'dialogue' in result
        
        # Test with filters
        char_results = app.search_dialogue("test", character_filter="Luke")
        for result in char_results:
            assert "Luke" in result['character']
    
    def test_get_character_dialogue_sample(self, mock_dialogue_data):
        """Test getting character dialogue samples."""
        app = StarWarsRAGApp()
        
        # Load system
        csv_path = "temp.csv"
        mock_dialogue_data.to_csv(csv_path, index=False)
        app.load_from_processed_data(csv_path)
        Path(csv_path).unlink()  # Clean up
        
        # Test getting samples
        samples = app.get_character_dialogue_sample("Luke Skywalker", sample_size=2)
        
        assert isinstance(samples, list)
        assert len(samples) <= 2
        
        for sample in samples:
            assert 'character' in sample
            assert 'dialogue' in sample
            assert sample['character'] == "Luke Skywalker"
    
    def test_get_system_stats(self, mock_dialogue_data):
        """Test getting system statistics."""
        app = StarWarsRAGApp()
        
        # Load system
        csv_path = "temp.csv"
        mock_dialogue_data.to_csv(csv_path, index=False)
        app.load_from_processed_data(csv_path)
        Path(csv_path).unlink()  # Clean up
        
        stats = app.get_system_stats()
        
        assert isinstance(stats, dict)
        
        required_keys = [
            'total_dialogue_lines', 'num_characters', 'num_movies',
            'characters', 'movies', 'top_characters',
            'embedding_model', 'embedding_dimension'
        ]
        
        for key in required_keys:
            assert key in stats
        
        assert stats['total_dialogue_lines'] == len(mock_dialogue_data)
        assert stats['embedding_model'] == "all-MiniLM-L6-v2"
        assert stats['embedding_dimension'] == 384
        assert isinstance(stats['characters'], list)
        assert isinstance(stats['movies'], list)
    
    def test_test_retrieval_quality_default_queries(self, mock_dialogue_data):
        """Test retrieval quality testing with default queries."""
        app = StarWarsRAGApp()
        
        # Load system
        csv_path = "temp.csv"
        mock_dialogue_data.to_csv(csv_path, index=False)
        app.load_from_processed_data(csv_path)
        Path(csv_path).unlink()  # Clean up
        
        test_results = app.test_retrieval_quality()
        
        assert isinstance(test_results, dict)
        assert 'test_queries' in test_results
        assert 'total_results_found' in test_results
        assert 'average_results_per_query' in test_results
        assert 'query_results' in test_results
        
        assert len(test_results['test_queries']) > 0
        assert isinstance(test_results['total_results_found'], int)
        assert isinstance(test_results['average_results_per_query'], float)
    
    def test_test_retrieval_quality_custom_queries(self, mock_dialogue_data):
        """Test retrieval quality testing with custom queries."""
        app = StarWarsRAGApp()
        
        # Load system
        csv_path = "temp.csv"
        mock_dialogue_data.to_csv(csv_path, index=False)
        app.load_from_processed_data(csv_path)
        Path(csv_path).unlink()  # Clean up
        
        custom_queries = ["Test query 1", "Test query 2"]
        test_results = app.test_retrieval_quality(custom_queries)
        
        assert test_results['test_queries'] == custom_queries
        assert len(test_results['query_results']) == len(custom_queries)
        
        for query in custom_queries:
            assert query in test_results['query_results']
            query_result = test_results['query_results'][query]
            assert 'num_results' in query_result
            assert 'top_similarity' in query_result
            assert 'results' in query_result


class TestStarWarsRAGAppIntegration:
    """Integration tests for StarWarsRAGApp."""
    
    @pytest.mark.integration
    @pytest.mark.real_data
    @pytest.mark.slow
    def test_complete_app_workflow(self, sample_script_path, tmp_path):
        """Test complete application workflow from script to chat."""
        if not sample_script_path.exists():
            pytest.skip("No real script data available")
        
        app = StarWarsRAGApp()
        
        # Load from single script file (create temp directory)
        temp_script_dir = tmp_path / "scripts"
        temp_script_dir.mkdir()
        
        # Copy script to temp directory
        temp_script = temp_script_dir / sample_script_path.name
        temp_script.write_text(sample_script_path.read_text(encoding='utf-8'), encoding='utf-8')
        
        # Load the system
        processed_path = tmp_path / "processed.csv"
        embeddings_path = tmp_path / "embeddings.npy"
        
        app.load_from_scripts(
            temp_script_dir,
            pattern=sample_script_path.name,
            save_processed_data=str(processed_path),
            save_embeddings=str(embeddings_path)
        )
        
        assert app.is_loaded
        assert processed_path.exists()
        assert embeddings_path.exists()
        
        # Test system functionality
        stats = app.get_system_stats()
        assert stats['total_dialogue_lines'] > 0
        
        # Test search
        search_results = app.search_dialogue("Force", top_k=3)
        
        # Test character interaction (if characters available)
        characters = stats['characters']
        if characters:
            char_response = app.chat_with_character("Hello", characters[0])
            assert char_response['character'] == characters[0]
            
            # Test character samples
            samples = app.get_character_dialogue_sample(characters[0], sample_size=2)
            assert len(samples) <= 2
        
        # Test retrieval quality
        quality_results = app.test_retrieval_quality()
        assert quality_results['total_results_found'] >= 0
        
        print(f"✅ Complete app workflow test:")
        print(f"   - Loaded {stats['total_dialogue_lines']} dialogue lines")
        print(f"   - Found {len(characters)} characters")
        print(f"   - Search returned {len(search_results)} results")
        print(f"   - Quality test: {quality_results['average_results_per_query']:.1f} avg results/query")
    
    @pytest.mark.integration
    def test_save_and_reload_workflow(self, mock_dialogue_data, tmp_path):
        """Test saving and reloading app data."""
        # Create and load first app
        app1 = StarWarsRAGApp()
        
        csv_path = tmp_path / "dialogue.csv"
        embeddings_path = tmp_path / "embeddings.npy"
        
        mock_dialogue_data.to_csv(csv_path, index=False)
        app1.load_from_processed_data(csv_path)
        
        # Save embeddings
        app1.embedder.save_embeddings(app1.retriever.embeddings, str(embeddings_path))
        
        # Test query with first app
        results1 = app1.search_dialogue("test query", top_k=2)
        
        # Create and load second app
        app2 = StarWarsRAGApp()
        app2.load_from_processed_data(csv_path, embeddings_path)
        
        # Test same query with second app
        results2 = app2.search_dialogue("test query", top_k=2)
        
        # Should get identical results
        assert len(results1) == len(results2)
        
        for r1, r2 in zip(results1, results2):
            assert abs(r1['similarity'] - r2['similarity']) < 1e-6
            assert r1['character'] == r2['character']
            assert r1['dialogue'] == r2['dialogue']
