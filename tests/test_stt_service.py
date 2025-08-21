"""
Tests for the Speech-to-Text (STT) service.

This module contains unit tests and integration tests for the STT service
using OpenAI's Whisper model.
"""

import os
import tempfile
import pytest
import io
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import UploadFile

from src.star_wars_rag.stt_service import app, STTService, TranscriptionResponse


class TestSTTService:
    """Test cases for STTService class."""
    
    def test_stt_service_initialization(self):
        """Test STT service initialization."""
        with patch('src.star_wars_rag.stt_service.whisper.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            service = STTService(model_name="base")
            
            assert service.model_name == "base"
            assert service.model == mock_model
            mock_load.assert_called_once_with("base")
    
    def test_stt_service_initialization_failure(self):
        """Test STT service initialization failure."""
        with patch('src.star_wars_rag.stt_service.whisper.load_model') as mock_load:
            mock_load.side_effect = Exception("Model loading failed")
            
            with pytest.raises(RuntimeError, match="Could not load Whisper model"):
                STTService(model_name="base")
    
    def test_transcribe_audio_success(self):
        """Test successful audio transcription."""
        with patch('src.star_wars_rag.stt_service.whisper.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            # Mock transcription result
            mock_result = {
                "text": "Hello, this is a test transcription.",
                "language": "en",
                "avg_logprob": -0.5,
                "duration": 2.5
            }
            mock_model.transcribe.return_value = mock_result
            
            service = STTService(model_name="base")
            
            # Create a temporary audio file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(b"fake audio data")
                temp_file_path = temp_file.name
            
            try:
                result = service.transcribe_audio(temp_file_path, "en", "transcribe")
                
                assert result == mock_result
                mock_model.transcribe.assert_called_once_with(
                    temp_file_path,
                    language="en",
                    task="transcribe",
                    fp16=False
                )
            finally:
                os.unlink(temp_file_path)
    
    def test_transcribe_audio_file_not_found(self):
        """Test transcription with non-existent file."""
        with patch('src.star_wars_rag.stt_service.whisper.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            service = STTService(model_name="base")
            
            with pytest.raises(RuntimeError, match="Audio file not found"):
                service.transcribe_audio("non_existent_file.wav")
    
    def test_transcribe_audio_transcription_failure(self):
        """Test transcription failure."""
        with patch('src.star_wars_rag.stt_service.whisper.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            mock_model.transcribe.side_effect = Exception("Transcription failed")
            
            service = STTService(model_name="base")
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(b"fake audio data")
                temp_file_path = temp_file.name
            
            try:
                with pytest.raises(RuntimeError, match="Transcription failed"):
                    service.transcribe_audio(temp_file_path)
            finally:
                os.unlink(temp_file_path)


class TestSTTServiceEndpoints:
    """Test cases for STT service FastAPI endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "stt"
        assert "model" in data
    
    def test_list_models(self, client):
        """Test models listing endpoint."""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "available_models" in data
        assert "current_model" in data
        assert isinstance(data["available_models"], list)
        assert len(data["available_models"]) > 0
    
    @patch('src.star_wars_rag.stt_service.stt_service.transcribe_audio')
    def test_transcribe_audio_success(self, mock_transcribe, client):
        """Test successful audio transcription endpoint."""
        # Mock transcription result
        mock_result = {
            "text": "Hello, this is a test transcription.",
            "language": "en",
            "avg_logprob": -0.5,
            "duration": 2.5
        }
        mock_transcribe.return_value = mock_result
        
        # Create mock audio file
        audio_content = b"fake audio data"
        files = {"audio": ("test.wav", io.BytesIO(audio_content), "audio/wav")}
        
        response = client.post("/transcribe", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Hello, this is a test transcription."
        assert data["language"] == "en"
        assert data["confidence"] == -0.5
        assert data["duration"] == 2.5
    
    def test_transcribe_audio_invalid_file_type(self, client):
        """Test transcription with invalid file type."""
        # Create mock non-audio file
        file_content = b"not audio data"
        files = {"audio": ("test.txt", io.BytesIO(file_content), "text/plain")}
        
        response = client.post("/transcribe", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "Invalid file type" in data["detail"]
    
    @patch('src.star_wars_rag.stt_service.stt_service.transcribe_audio')
    def test_transcribe_audio_no_speech_detected(self, mock_transcribe, client):
        """Test transcription with no speech detected."""
        # Mock empty transcription result
        mock_result = {
            "text": "",
            "language": "en",
            "avg_logprob": None,
            "duration": None
        }
        mock_transcribe.return_value = mock_result
        
        # Create mock audio file
        audio_content = b"fake audio data"
        files = {"audio": ("test.wav", io.BytesIO(audio_content), "audio/wav")}
        
        response = client.post("/transcribe", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "No speech detected" in data["detail"]
    
    @patch('src.star_wars_rag.stt_service.stt_service.transcribe_audio')
    def test_transcribe_audio_service_error(self, mock_transcribe, client):
        """Test transcription with service error."""
        mock_transcribe.side_effect = RuntimeError("Service error")
        
        # Create mock audio file
        audio_content = b"fake audio data"
        files = {"audio": ("test.wav", io.BytesIO(audio_content), "audio/wav")}
        
        response = client.post("/transcribe", files=files)
        
        assert response.status_code == 500
        data = response.json()
        assert "Internal server error" in data["detail"]


class TestTranscriptionResponse:
    """Test cases for TranscriptionResponse model."""
    
    def test_transcription_response_creation(self):
        """Test TranscriptionResponse model creation."""
        response = TranscriptionResponse(
            text="Test transcription",
            language="en",
            confidence=0.8,
            duration=1.5
        )
        
        assert response.text == "Test transcription"
        assert response.language == "en"
        assert response.confidence == 0.8
        assert response.duration == 1.5
    
    def test_transcription_response_minimal(self):
        """Test TranscriptionResponse with minimal data."""
        response = TranscriptionResponse(
            text="Test transcription",
            language="en"
        )
        
        assert response.text == "Test transcription"
        assert response.language == "en"
        assert response.confidence is None
        assert response.duration is None


if __name__ == "__main__":
    pytest.main([__file__])
