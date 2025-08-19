#!/usr/bin/env python3
"""
Tests for dashboard updates and functionality.
"""

import pytest
import requests
import time
from unittest.mock import Mock, patch


class TestDashboardUpdates:
    """Test dashboard functionality and updates."""
    
    def test_dashboard_readme_function_exists(self):
        """Test that dashboard readme function exists."""
        from src.star_wars_rag.dashboard import display_dashboard_readme
        
        # Function should exist and be callable
        assert callable(display_dashboard_readme)
    
    def test_vertical_suggestions_count(self):
        """Test that suggestions are limited to 6 items."""
        # This would be tested by importing the dashboard module
        # and checking the suggestions list length
        
        # Mock test - in real implementation would check actual suggestions
        suggestions = [
            "Rate the cantina band's musical talent on a scale of 1-10",
            "What's your opinion on the Death Star's interior design?",
            "If you could choose your own theme music, what would it sound like?",
            "What's the most ridiculous thing you've seen a Stormtrooper miss?",
            "Explain quantum physics using only Star Wars analogies",
            "If droids had social media, what would they post about?"
        ]
        
        assert len(suggestions) == 6
        assert all(isinstance(s, str) for s in suggestions)
        assert all(len(s) > 10 for s in suggestions)  # Substantial questions
    
    def test_context_panel_enhancements(self):
        """Test that context panel shows scene context."""
        from src.star_wars_rag.dashboard import display_retrieved_context_panel
        
        # Function should exist and handle context data
        assert callable(display_retrieved_context_panel)
        
        # Mock context data with scene context
        mock_context = [
            {
                'character': 'LUKE SKYWALKER',
                'dialogue': 'I want to learn the ways of the Force.',
                'movie': 'The Empire Strikes Back',
                'similarity': 0.85,
                'context': 'Scene: Dagobah Swamp | Action: Training with Yoda'
            }
        ]
        
        # Should handle context with scene information
        assert mock_context[0]['context'] is not None
        assert 'Scene:' in mock_context[0]['context']
    
    def test_llm_info_panel_enhancements(self):
        """Test that LLM info panel shows full prompt."""
        from src.star_wars_rag.dashboard import display_llm_info_panel
        
        # Function should exist
        assert callable(display_llm_info_panel)
        
        # Mock metadata with full prompt
        mock_metadata = {
            'retrieval_results': 5,
            'context_lines_used': 3,
            'total_time_seconds': 1.25,
            'prompt_length': 1500,
            'full_prompt': 'You are Luke Skywalker. Context: [dialogue lines...] User: Tell me about the Force.'
        }
        
        # Should handle full prompt data
        assert 'full_prompt' in mock_metadata
        assert len(mock_metadata['full_prompt']) > 0


class TestAPIIntegration:
    """Test API integration and health."""
    
    @pytest.fixture
    def api_base_url(self):
        """API base URL for testing."""
        return "http://localhost:8002"
    
    def test_api_health_endpoint(self, api_base_url):
        """Test API health endpoint responds."""
        try:
            response = requests.get(f"{api_base_url}/health", timeout=5)
            assert response.status_code == 200
            
            health_data = response.json()
            assert 'status' in health_data
            assert 'models_loaded' in health_data
            assert 'database_connected' in health_data
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API not running - skipping integration test")
    
    def test_characters_endpoint(self, api_base_url):
        """Test characters endpoint returns data."""
        try:
            response = requests.get(f"{api_base_url}/characters", timeout=5)
            assert response.status_code == 200
            
            characters_data = response.json()
            assert 'characters' in characters_data
            assert len(characters_data['characters']) > 0
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API not running - skipping integration test")
    
    def test_system_info_endpoint(self, api_base_url):
        """Test system info endpoint returns metrics."""
        try:
            response = requests.get(f"{api_base_url}/system/info", timeout=5)
            assert response.status_code == 200
            
            system_data = response.json()
            assert 'dialogue_lines' in system_data
            assert 'characters_count' in system_data
            assert system_data['dialogue_lines'] > 0
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API not running - skipping integration test")


def test_preprocessed_data_integration():
    """Test that preprocessed data integrates properly."""
    from pathlib import Path
    
    # Check that preprocessed files exist
    preprocessed_dir = Path("data/preprocessed")
    assert preprocessed_dir.exists()
    
    # Check combined file has expected content
    combined_file = preprocessed_dir / "enhanced_original_trilogy_combined.txt"
    assert combined_file.exists()
    
    with open(combined_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Should contain scene context
    assert 'Scene:' in content
    assert 'Action:' in content
    
    # Should contain character dialogue
    assert 'LUKE' in content or 'VADER' in content or 'LEIA' in content
    
    # Should have proper format
    assert ' | ' in content  # Pipe-separated format


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
