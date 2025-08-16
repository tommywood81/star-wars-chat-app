"""
Test FastAPI backend endpoints with real Star Wars data.

This test verifies the complete FastAPI backend can handle real chat requests
and return proper responses using the actual Star Wars dialogue dataset.
"""

import pytest
import asyncio
from pathlib import Path
from fastapi.testclient import TestClient
import tempfile
import shutil

# We'll import the FastAPI app once it's created
# from src.star_wars_rag.api import app


class TestFastAPIBackend:
    """Test FastAPI backend with real Star Wars data."""
    
    @pytest.fixture(scope="class")
    def real_data_setup(self):
        """Setup real Star Wars data for testing."""
        data_dir = Path("data/raw")
        if not data_dir.exists():
            pytest.skip("Real data directory not found")
        
        script_files = list(data_dir.glob("*.txt"))
        if not script_files:
            pytest.skip("No script files found in data/raw")
        
        # Use A New Hope for testing
        test_script = None
        for script in script_files:
            if "NEW HOPE" in script.name.upper():
                test_script = script
                break
        
        if test_script is None:
            test_script = script_files[0]  # Use first available
        
        return {
            "script_path": test_script,
            "script_name": test_script.name
        }
    
    @pytest.fixture(scope="class")
    def client(self, real_data_setup):
        """Create FastAPI test client with real data loaded."""
        # Import here to avoid import errors before API is created
        try:
            from src.star_wars_rag.api import app
        except ImportError:
            pytest.skip("FastAPI app not yet implemented")
        
        # Initialize the app with real data
        with TestClient(app) as client:
            # Load real data into the app
            response = client.post("/admin/load-data", json={
                "script_path": str(real_data_setup["script_path"]),
                "force_reload": True
            })
            assert response.status_code == 200
            yield client
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "models_loaded" in data
        assert "database_connected" in data
        assert "data_loaded" in data
        
    def test_chat_endpoint_with_luke(self, client):
        """Test chat endpoint with Luke Skywalker using real data."""
        chat_request = {
            "character": "Luke Skywalker",
            "message": "Tell me about the Force",
            "session_id": "test-session-1"
        }
        
        response = client.post("/chat", json=chat_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "character" in data
        assert data["character"] == "Luke Skywalker"
        assert "response" in data
        assert len(data["response"]) > 0
        assert "context_used" in data
        assert isinstance(data["context_used"], list)
        assert len(data["context_used"]) > 0
        assert "metadata" in data
        
        # Verify the response is character-appropriate
        response_text = data["response"].lower()
        # Luke should mention Force-related concepts or be optimistic
        assert any(word in response_text for word in ["force", "jedi", "hope", "learn", "believe"])
    
    def test_chat_endpoint_with_vader(self, client):
        """Test chat endpoint with Darth Vader using real data."""
        chat_request = {
            "character": "Darth Vader", 
            "message": "What is true power?",
            "session_id": "test-session-2"
        }
        
        response = client.post("/chat", json=chat_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["character"] == "Darth Vader"
        assert len(data["response"]) > 0
        
        # Verify context is relevant to Vader
        context_used = data["context_used"]
        assert len(context_used) > 0
        
        # At least some context should be from Vader
        vader_context = [ctx for ctx in context_used if "vader" in ctx.get("character", "").lower()]
        assert len(vader_context) > 0
    
    def test_stream_chat_endpoint(self, client):
        """Test streaming chat endpoint."""
        chat_request = {
            "character": "Obi-Wan Kenobi",
            "message": "What wisdom can you share?",
            "session_id": "test-session-3"
        }
        
        with client.stream("POST", "/chat/stream", json=chat_request) as response:
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream"
            
            # Collect streaming response
            events = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    import json
                    event_data = json.loads(line[6:])  # Remove "data: " prefix
                    events.append(event_data)
            
            assert len(events) > 0
            
            # Check final event has complete response
            final_event = events[-1]
            assert final_event.get("is_complete") is True
            assert "accumulated_response" in final_event
            assert len(final_event["accumulated_response"]) > 0
    
    def test_characters_endpoint(self, client):
        """Test available characters endpoint."""
        response = client.get("/characters")
        assert response.status_code == 200
        
        data = response.json()
        assert "characters" in data
        assert isinstance(data["characters"], list)
        assert len(data["characters"]) > 0
        
        # Verify main characters are present
        character_names = [char["name"] for char in data["characters"]]
        expected_characters = ["Luke Skywalker", "Darth Vader", "Obi-Wan Kenobi"]
        
        found_characters = [char for char in expected_characters if char in character_names]
        assert len(found_characters) > 0, f"Expected characters not found. Available: {character_names}"
    
    def test_character_stats_endpoint(self, client):
        """Test character statistics endpoint."""
        response = client.get("/characters/Luke Skywalker/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "character" in data
        assert data["character"] == "Luke Skywalker"
        assert "dialogue_count" in data
        assert data["dialogue_count"] > 0
        assert "sample_dialogue" in data
        assert isinstance(data["sample_dialogue"], list)
    
    def test_invalid_character_error(self, client):
        """Test error handling for invalid character."""
        chat_request = {
            "character": "Invalid Character Name",
            "message": "Hello",
            "session_id": "test-session-error"
        }
        
        response = client.post("/chat", json=chat_request)
        assert response.status_code == 400
        
        data = response.json()
        assert "error" in data
        assert "character" in data["error"].lower()
    
    def test_empty_message_error(self, client):
        """Test error handling for empty message."""
        chat_request = {
            "character": "Luke Skywalker",
            "message": "",
            "session_id": "test-session-empty"
        }
        
        response = client.post("/chat", json=chat_request)
        assert response.status_code == 400
        
        data = response.json()
        assert "error" in data
        assert "message" in data["error"].lower()
    
    def test_system_info_endpoint(self, client):
        """Test system information endpoint."""
        response = client.get("/system/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "dialogue_lines" in data
        assert "characters_count" in data
        assert "movies_count" in data
        assert "embedding_model" in data
        assert "llm_model" in data
        
        # Verify we have substantial data loaded
        assert data["dialogue_lines"] > 500  # Should have many lines from A New Hope
        assert data["characters_count"] > 10  # Should have many characters
    
    def test_concurrent_chat_requests(self, client):
        """Test handling multiple concurrent chat requests."""
        import concurrent.futures
        import threading
        
        def make_chat_request(session_id):
            chat_request = {
                "character": "Han Solo",
                "message": f"What's your opinion on session {session_id}?",
                "session_id": f"concurrent-session-{session_id}"
            }
            response = client.post("/chat", json=chat_request)
            return response.status_code, response.json()
        
        # Make 3 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_chat_request, i) for i in range(3)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for status_code, data in results:
            assert status_code == 200
            assert "response" in data
            assert len(data["response"]) > 0


@pytest.mark.real_data
@pytest.mark.integration
class TestFastAPIBackendIntegration:
    """Integration tests for FastAPI backend."""
    
    def test_full_conversation_flow(self):
        """Test a complete conversation flow with context continuity."""
        try:
            from src.star_wars_rag.api import app
        except ImportError:
            pytest.skip("FastAPI app not yet implemented")
        
        with TestClient(app) as client:
            session_id = "integration-test-conversation"
            
            # First message
            response1 = client.post("/chat", json={
                "character": "Luke Skywalker",
                "message": "I want to learn about the Force",
                "session_id": session_id
            })
            assert response1.status_code == 200
            
            # Follow-up message in same session
            response2 = client.post("/chat", json={
                "character": "Luke Skywalker", 
                "message": "Can you tell me more about what you just said?",
                "session_id": session_id
            })
            assert response2.status_code == 200
            
            # Responses should be contextually related
            data1 = response1.json()
            data2 = response2.json()
            
            assert len(data1["response"]) > 0
            assert len(data2["response"]) > 0
            
            # Session should be tracked
            assert data1.get("session_id") == session_id
            assert data2.get("session_id") == session_id
