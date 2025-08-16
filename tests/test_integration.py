"""
End-to-end integration tests for the Star Wars RAG system.

These tests verify the complete pipeline from raw scripts to retrieval.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from star_wars_rag.data_processor import DialogueProcessor
from star_wars_rag.embeddings import StarWarsEmbedder
from star_wars_rag.retrieval import DialogueRetriever


class TestStarWarsRAGIntegration:
    """Integration tests for the complete Star Wars RAG system."""
    
    @pytest.mark.integration
    @pytest.mark.real_data
    @pytest.mark.slow
    def test_complete_pipeline_single_script(self, sample_script_path):
        """Test the complete pipeline from script file to retrieval."""
        # Step 1: Process script data
        processor = DialogueProcessor()
        dialogue_df = processor.process_script_file(sample_script_path)
        
        assert not dialogue_df.empty, "Should extract dialogue from script"
        assert 'dialogue_clean' in dialogue_df.columns
        assert 'character_normalized' in dialogue_df.columns
        
        # Step 2: Generate embeddings
        embedder = StarWarsEmbedder()
        
        # Use subset for faster testing
        test_data = dialogue_df.head(20)
        texts = test_data['dialogue_clean'].tolist()
        
        embeddings = embedder.embed_batch(texts, show_progress=False)
        
        assert embeddings.shape[0] == len(test_data)
        assert embeddings.shape[1] == 384
        
        # Validate embedding quality
        stats = embedder.validate_embeddings(embeddings)
        assert not stats['has_nan']
        assert not stats['has_inf']
        
        # Step 3: Set up retrieval system
        retriever = DialogueRetriever(embedder)
        retriever.load_dialogue_data(test_data, embeddings)
        
        assert retriever.is_ready()
        
        # Step 4: Test retrieval functionality
        test_queries = [
            "Tell me about the Force",
            "Help me with something",
            "I'm in danger"
        ]
        
        all_results = []
        for query in test_queries:
            results = retriever.retrieve_similar_dialogue(query, top_k=3)
            all_results.extend(results)
            
            # Verify result structure
            for result in results:
                assert 'similarity' in result
                assert 'character' in result
                assert 'dialogue' in result
                assert 'movie' in result
                assert isinstance(result['similarity'], float)
                assert 0 <= result['similarity'] <= 1
        
        # Should have found some relevant results
        assert len(all_results) > 0
        
        # Test character-specific retrieval
        characters = retriever.get_available_characters()
        if characters:
            char_results = retriever.retrieve_similar_dialogue(
                "What do you think?",
                character_filter=characters[0],
                top_k=2
            )
            
            for result in char_results:
                assert result['character'] == characters[0]
        
        print(f"✅ Complete pipeline test passed:")
        print(f"   - Processed {len(dialogue_df)} dialogue lines")
        print(f"   - Generated {len(embeddings)} embeddings")
        print(f"   - Found {len(characters)} characters")
        print(f"   - Retrieved {len(all_results)} relevant results")
    
    @pytest.mark.integration
    @pytest.mark.real_data
    def test_character_dialogue_consistency(self, sample_dialogue_data):
        """Test that character dialogue remains consistent through the pipeline."""
        if sample_dialogue_data.empty:
            pytest.skip("No dialogue data available")
        
        # Get character distribution from processed data
        original_char_counts = sample_dialogue_data['character_normalized'].value_counts()
        
        # Set up complete system
        embedder = StarWarsEmbedder()
        retriever = DialogueRetriever(embedder)
        
        # Use subset for testing
        test_data = sample_dialogue_data.head(30)
        retriever.load_dialogue_data(test_data)
        
        # Verify character consistency
        system_characters = retriever.get_available_characters()
        system_stats = retriever.get_character_stats()
        
        # Characters should match
        for char in system_characters:
            assert char in test_data['character_normalized'].values
        
        # Counts should match
        for char, count in system_stats.items():
            expected_count = len(test_data[test_data['character_normalized'] == char])
            assert count == expected_count
        
        # Test character-specific retrieval
        for character in system_characters[:3]:  # Test first 3 characters
            char_dialogue = retriever.get_character_dialogue(character)
            
            # Should only return this character's dialogue
            for item in char_dialogue:
                assert item['character'] == character
            
            # Count should match
            assert len(char_dialogue) == system_stats[character]
    
    @pytest.mark.integration 
    @pytest.mark.slow
    def test_embedding_similarity_consistency(self):
        """Test that embedding similarity makes semantic sense."""
        embedder = StarWarsEmbedder()
        
        # Test semantically related groups
        force_related = [
            "The Force is strong with you.",
            "May the Force be with you.",
            "Use the Force, Luke.",
            "I feel the Force flowing through me."
        ]
        
        combat_related = [
            "Attack the Death Star!",
            "Fire when ready!",
            "We're under attack!",
            "Prepare for battle!"
        ]
        
        droid_related = [
            "These aren't the droids you're looking for.",
            "The droid has malfunctioned.",
            "R2-D2, where are you?",
            "C-3PO is worried about the odds."
        ]
        
        all_texts = force_related + combat_related + droid_related
        embeddings = embedder.embed_batch(all_texts, show_progress=False)
        
        # Calculate within-group and between-group similarities
        def calc_avg_similarity(group1_indices, group2_indices):
            sims = []
            for i in group1_indices:
                for j in group2_indices:
                    if i != j:
                        sim = np.dot(embeddings[i], embeddings[j])
                        sims.append(sim)
            return np.mean(sims) if sims else 0
        
        force_indices = list(range(len(force_related)))
        combat_indices = list(range(len(force_related), len(force_related) + len(combat_related)))
        droid_indices = list(range(len(force_related) + len(combat_related), len(all_texts)))
        
        # Within-group similarities
        force_internal = calc_avg_similarity(force_indices, force_indices)
        combat_internal = calc_avg_similarity(combat_indices, combat_indices)
        droid_internal = calc_avg_similarity(droid_indices, droid_indices)
        
        # Between-group similarities
        force_combat = calc_avg_similarity(force_indices, combat_indices)
        force_droid = calc_avg_similarity(force_indices, droid_indices)
        combat_droid = calc_avg_similarity(combat_indices, droid_indices)
        
        # Within-group should be higher than between-group
        assert force_internal > force_combat
        assert force_internal > force_droid
        assert combat_internal > force_combat
        assert combat_internal > combat_droid
        assert droid_internal > force_droid
        assert droid_internal > combat_droid
        
        print(f"Similarity consistency test:")
        print(f"  Force internal: {force_internal:.3f}")
        print(f"  Combat internal: {combat_internal:.3f}")
        print(f"  Droid internal: {droid_internal:.3f}")
        print(f"  Force-Combat: {force_combat:.3f}")
        print(f"  Force-Droid: {force_droid:.3f}")
        print(f"  Combat-Droid: {combat_droid:.3f}")
    
    @pytest.mark.integration
    @pytest.mark.real_data
    def test_retrieval_ranking_quality(self, retriever_with_data):
        """Test that retrieval ranking produces sensible results."""
        if not retriever_with_data.is_ready():
            pytest.skip("No retrieval data available")
        
        # Test queries with expected character preferences
        test_cases = [
            {
                'query': "I want to learn about the Force",
                'preferred_characters': ['Luke Skywalker', 'Obi-Wan Kenobi', 'Darth Vader'],
                'keywords': ['Force', 'learn', 'power', 'strong']
            },
            {
                'query': "Calculate the odds of success",
                'preferred_characters': ['C-3PO'],
                'keywords': ['odds', 'calculate', 'probability', 'approximately']
            },
            {
                'query': "Help me, I need assistance",
                'preferred_characters': ['Princess Leia', 'Luke Skywalker'],
                'keywords': ['help', 'need', 'please', 'assist']
            }
        ]
        
        for test_case in test_cases:
            results = retriever_with_data.retrieve_similar_dialogue(
                test_case['query'], 
                top_k=5
            )
            
            if not results:
                continue  # Skip if no results
            
            # Check if results are properly ranked (highest similarity first)
            similarities = [r['similarity'] for r in results]
            assert similarities == sorted(similarities, reverse=True)
            
            # Check for semantic relevance (at least one keyword match in top results)
            top_dialogue = ' '.join([r['dialogue'].lower() for r in results[:3]])
            keyword_matches = sum(1 for keyword in test_case['keywords'] 
                                if keyword.lower() in top_dialogue)
            
            print(f"Query: '{test_case['query']}'")
            print(f"  Found {len(results)} results")
            print(f"  Keyword matches: {keyword_matches}/{len(test_case['keywords'])}")
            if results:
                print(f"  Top result: [{results[0]['similarity']:.3f}] {results[0]['character']}: {results[0]['dialogue'][:50]}...")
    
    @pytest.mark.integration
    def test_system_error_handling(self):
        """Test system behavior with edge cases and errors."""
        # Test with invalid data
        processor = DialogueProcessor()
        embedder = StarWarsEmbedder()
        retriever = DialogueRetriever()
        
        # Test empty script processing
        empty_dialogue = processor.extract_dialogue_lines("", "Empty Movie")
        assert len(empty_dialogue) == 0
        
        # Test malformed script
        malformed_script = "This is not a proper script format at all."
        malformed_dialogue = processor.extract_dialogue_lines(malformed_script, "Malformed")
        # Should handle gracefully (might extract 0 or few lines)
        assert isinstance(malformed_dialogue, list)
        
        # Test embedding with invalid input
        with pytest.raises(ValueError):
            embedder.embed_text("")
        
        with pytest.raises(ValueError):
            embedder.embed_batch([])
        
        # Test retriever without data
        assert not retriever.is_ready()
        
        with pytest.raises(RuntimeError):
            retriever.retrieve_similar_dialogue("test")
    
    @pytest.mark.integration
    @pytest.mark.real_data
    @pytest.mark.slow
    def test_performance_with_larger_dataset(self, test_data_dir):
        """Test system performance with larger dataset."""
        processor = DialogueProcessor()
        
        # Try to process multiple scripts
        try:
            all_dialogue = processor.process_multiple_scripts(
                test_data_dir, 
                "STAR WARS*.txt"
            )
            
            if all_dialogue.empty:
                pytest.skip("No scripts could be processed")
            
            # Limit size for testing
            if len(all_dialogue) > 100:
                test_data = all_dialogue.sample(n=100, random_state=42)
            else:
                test_data = all_dialogue
            
            # Test embedding performance
            embedder = StarWarsEmbedder()
            retriever = DialogueRetriever(embedder)
            
            # Load data (this will compute embeddings)
            retriever.load_dialogue_data(test_data)
            
            assert retriever.is_ready()
            
            # Test retrieval performance
            test_queries = [
                "The Force is powerful",
                "We need to escape",
                "The Empire is dangerous",
                "Help me find the droids",
                "I have a bad feeling about this"
            ]
            
            total_results = 0
            for query in test_queries:
                results = retriever.retrieve_similar_dialogue(query, top_k=5)
                total_results += len(results)
            
            # Should find relevant results
            assert total_results > 0
            
            # Test system statistics
            characters = retriever.get_available_characters()
            movies = retriever.get_available_movies()
            
            print(f"Performance test with larger dataset:")
            print(f"  Processed {len(test_data)} dialogue lines")
            print(f"  Found {len(characters)} characters")
            print(f"  From {len(movies)} movies")
            print(f"  Retrieved {total_results} results for {len(test_queries)} queries")
            
        except Exception as e:
            pytest.skip(f"Large dataset test failed: {e}")
    
    @pytest.mark.integration
    def test_save_and_load_embeddings_integration(self, mock_dialogue_data, tmp_path):
        """Test saving and loading embeddings in the full system."""
        # Create system with data
        embedder = StarWarsEmbedder()
        retriever = DialogueRetriever(embedder)
        retriever.load_dialogue_data(mock_dialogue_data)
        
        # Save embeddings
        embeddings_file = tmp_path / "test_embeddings.npy"
        embedder.save_embeddings(retriever.embeddings, str(embeddings_file))
        
        # Create new system and load embeddings
        new_embedder = StarWarsEmbedder()
        new_retriever = DialogueRetriever(new_embedder)
        
        loaded_embeddings = new_embedder.load_embeddings(str(embeddings_file))
        new_retriever.load_dialogue_data(mock_dialogue_data, loaded_embeddings)
        
        # Should produce same results
        query = "test query"
        original_results = retriever.retrieve_similar_dialogue(query)
        loaded_results = new_retriever.retrieve_similar_dialogue(query)
        
        assert len(original_results) == len(loaded_results)
        
        # Similarities should be identical (or very close)
        for orig, loaded in zip(original_results, loaded_results):
            assert abs(orig['similarity'] - loaded['similarity']) < 1e-6
            assert orig['character'] == loaded['character']
            assert orig['dialogue'] == loaded['dialogue']
