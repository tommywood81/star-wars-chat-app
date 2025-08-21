"""
Tests for the Text-to-Speech (TTS) service.

This module contains unit tests and integration tests for the TTS service
using Coqui TTS.
"""

import os
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from pathlib import Path

from src.star_wars_rag.tts_service import app, TTSService, TTSRequest, TTSResponse


class TestTTSService:
    """Test cases for TTSService class."""
    
    def test_tts_service_initialization(self):
        """Test TTS service initialization."""
        with patch('src.star_wars_rag.tts_service.TTS') as mock_tts_class:
            mock_tts = Mock()
            mock_tts_class.return_value = mock_tts
            
            service = TTSService(default_voice="ljspeech")
            
            assert service.default_voice == "ljspeech"
            assert service.tts == mock_tts
            mock_tts_class.assert_called_once_with(
                model_name="tts_models/en/ljspeech/tacotron2-DDC",
                progress_bar=False,
                gpu=False
            )
    
    def test_tts_service_initialization_failure(self):
        """Test TTS service initialization failure."""
        with patch('src.star_wars_rag.tts_service.TTS') as mock_tts_class:
            mock_tts_class.side_effect = Exception("TTS loading failed")
            
            with pytest.raises(RuntimeError, match="Could not load TTS model"):
                TTSService(default_voice="ljspeech")
    
    @patch('src.star_wars_rag.tts_service.tempfile.NamedTemporaryFile')
    @patch('src.star_wars_rag.tts_service.os.path.getsize')
    def test_synthesize_speech_success(self, mock_getsize, mock_tempfile):
        """Test successful speech synthesis."""
        with patch('src.star_wars_rag.tts_service.TTS') as mock_tts_class:
            mock_tts = Mock()
            mock_tts_class.return_value = mock_tts
            
            # Mock temporary file
            mock_temp = Mock()
            mock_temp.name = "/tmp/test_audio.wav"
            mock_tempfile.return_value.__enter__.return_value = mock_temp
            
            # Mock file size
            mock_getsize.return_value = 1024
            
            service = TTSService(default_voice="ljspeech")
            
            result = service.synthesize_speech(
                text="Hello, this is a test.",
                voice="ljspeech",
                speed=1.0
            )
            
            assert result["audio_file"] == "/tmp/test_audio.wav"
            assert result["voice"] == "ljspeech"
            assert result["text_length"] == 5
            assert result["file_size"] == 1024
            assert result["duration"] is not None
            
            # Verify TTS was called correctly
            mock_tts.tts_to_file.assert_called_once_with(
                text="Hello, this is a test.",
                file_path="/tmp/test_audio.wav",
                speed=1.0
            )
    
    def test_synthesize_speech_empty_text(self):
        """Test speech synthesis with empty text."""
        with patch('src.star_wars_rag.tts_service.TTS') as mock_tts_class:
            mock_tts = Mock()
            mock_tts_class.return_value = mock_tts
            
            service = TTSService(default_voice="ljspeech")
            
            with pytest.raises(ValueError, match="Text cannot be empty"):
                service.synthesize_speech(text="")
            
            with pytest.raises(ValueError, match="Text cannot be empty"):
                service.synthesize_speech(text="   ")
    
    @patch('src.star_wars_rag.tts_service.tempfile.NamedTemporaryFile')
    def test_synthesize_speech_tts_failure(self, mock_tempfile):
        """Test speech synthesis when TTS fails."""
        with patch('src.star_wars_rag.tts_service.TTS') as mock_tts_class:
            mock_tts = Mock()
            mock_tts_class.return_value = mock_tts
            mock_tts.tts_to_file.side_effect = Exception("TTS synthesis failed")
            
            # Mock temporary file
            mock_temp = Mock()
            mock_temp.name = "/tmp/test_audio.wav"
            mock_tempfile.return_value.__enter__.return_value = mock_temp
            
            service = TTSService(default_voice="ljspeech")
            
            with pytest.raises(RuntimeError, match="Speech synthesis failed"):
                service.synthesize_speech(text="Hello")
    
    def test_get_available_voices(self):
        """Test getting available voices."""
        with patch('src.star_wars_rag.tts_service.TTS') as mock_tts_class:
            mock_tts = Mock()
            mock_tts_class.return_value = mock_tts
            
            service = TTSService(default_voice="ljspeech")
            voices = service.get_available_voices()
            
            expected_voices = ["ljspeech", "vctk", "fastspeech2"]
            assert voices == expected_voices
    
    def test_get_voice_info_valid_voice(self):
        """Test getting voice info for valid voice."""
        with patch('src.star_wars_rag.tts_service.TTS') as mock_tts_class:
            mock_tts = Mock()
            mock_tts_class.return_value = mock_tts
            
            service = TTSService(default_voice="ljspeech")
            info = service.get_voice_info("ljspeech")
            
            assert info["name"] == "ljspeech"
            assert info["model"] == "tts_models/en/ljspeech/tacotron2-DDC"
            assert "speed" in info["supported_features"]
    
    def test_get_voice_info_invalid_voice(self):
        """Test getting voice info for invalid voice."""
        with patch('src.star_wars_rag.tts_service.TTS') as mock_tts_class:
            mock_tts = Mock()
            mock_tts_class.return_value = mock_tts
            
            service = TTSService(default_voice="ljspeech")
            
            with pytest.raises(ValueError, match="Voice 'invalid' not found"):
                service.get_voice_info("invalid")


class TestTTSServiceEndpoints:
    """Test cases for TTS service FastAPI endpoints."""
    
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
        assert data["service"] == "tts"
        assert "default_voice" in data
    
    def test_list_voices(self, client):
        """Test voices listing endpoint."""
        response = client.get("/voices")
        assert response.status_code == 200
        data = response.json()
        assert "available_voices" in data
        assert "default_voice" in data
        assert isinstance(data["available_voices"], list)
        assert len(data["available_voices"]) > 0
    
    def test_get_voice_info_valid(self, client):
        """Test getting voice info for valid voice."""
        response = client.get("/voices/ljspeech")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "ljspeech"
        assert "model" in data
        assert "supported_features" in data
    
    def test_get_voice_info_invalid(self, client):
        """Test getting voice info for invalid voice."""
        response = client.get("/voices/invalid_voice")
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]
    
    @patch('src.star_wars_rag.tts_service.tts_service.synthesize_speech')
    def test_synthesize_speech_success(self, mock_synthesize, client):
        """Test successful speech synthesis endpoint."""
        # Mock synthesis result
        mock_result = {
            "audio_file": "/tmp/test_audio.wav",
            "duration": 2.5,
            "voice": "ljspeech",
            "text_length": 5,
            "file_size": 1024
        }
        mock_synthesize.return_value = mock_result
        
        request_data = {
            "text": "Hello, this is a test.",
            "voice": "ljspeech",
            "speed": 1.0
        }
        
        response = client.post("/synthesize", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["audio_file"] == "/tmp/test_audio.wav"
        assert data["duration"] == 2.5
        assert data["voice"] == "ljspeech"
        assert data["text_length"] == 5
        
        # Verify service was called correctly
        mock_synthesize.assert_called_once_with(
            text="Hello, this is a test.",
            voice="ljspeech",
            speed=1.0,
            emotion=None
        )
    
    @patch('src.star_wars_rag.tts_service.tts_service.synthesize_speech')
    def test_synthesize_speech_validation_error(self, mock_synthesize, client):
        """Test speech synthesis with validation error."""
        mock_synthesize.side_effect = ValueError("Text cannot be empty")
        
        request_data = {
            "text": "",
            "voice": "ljspeech"
        }
        
        response = client.post("/synthesize", json=request_data)
        
        assert response.status_code == 400
        data = response.json()
        assert "Text cannot be empty" in data["detail"]
    
    @patch('src.star_wars_rag.tts_service.tts_service.synthesize_speech')
    def test_synthesize_speech_service_error(self, mock_synthesize, client):
        """Test speech synthesis with service error."""
        mock_synthesize.side_effect = RuntimeError("Service error")
        
        request_data = {
            "text": "Hello",
            "voice": "ljspeech"
        }
        
        response = client.post("/synthesize", json=request_data)
        
        assert response.status_code == 500
        data = response.json()
        assert "Internal server error" in data["detail"]
    
    @patch('src.star_wars_rag.tts_service.tts_service.synthesize_speech')
    def test_synthesize_speech_simple_success(self, mock_synthesize, client):
        """Test simple synthesis endpoint."""
        # Mock synthesis result
        mock_result = {
            "audio_file": "/tmp/test_audio.wav",
            "duration": 2.5,
            "voice": "ljspeech",
            "text_length": 5,
            "file_size": 1024
        }
        mock_synthesize.return_value = mock_result
        
        # Mock file response
        with patch('src.star_wars_rag.tts_service.FileResponse') as mock_file_response:
            mock_file_response.return_value = Mock()
            
            response = client.post("/synthesize-simple?text=Hello&voice=ljspeech")
            
            assert response.status_code == 200
            
            # Verify service was called correctly
            mock_synthesize.assert_called_once_with(
                text="Hello",
                voice="ljspeech"
            )


class TestTTSRequest:
    """Test cases for TTSRequest model."""
    
    def test_tts_request_creation(self):
        """Test TTSRequest model creation."""
        request = TTSRequest(
            text="Hello, world!",
            voice="ljspeech",
            speed=1.2,
            emotion="happy"
        )
        
        assert request.text == "Hello, world!"
        assert request.voice == "ljspeech"
        assert request.speed == 1.2
        assert request.emotion == "happy"
    
    def test_tts_request_defaults(self):
        """Test TTSRequest with default values."""
        request = TTSRequest(text="Hello")
        
        assert request.text == "Hello"
        assert request.voice == "ljspeech"
        assert request.speed == 1.0
        assert request.emotion is None


class TestTTSResponse:
    """Test cases for TTSResponse model."""
    
    def test_tts_response_creation(self):
        """Test TTSResponse model creation."""
        response = TTSResponse(
            audio_file="/tmp/test.wav",
            duration=2.5,
            voice="ljspeech",
            text_length=10
        )
        
        assert response.audio_file == "/tmp/test.wav"
        assert response.duration == 2.5
        assert response.voice == "ljspeech"
        assert response.text_length == 10
    
    def test_tts_response_minimal(self):
        """Test TTSResponse with minimal data."""
        response = TTSResponse(
            audio_file="/tmp/test.wav",
            voice="ljspeech",
            text_length=10
        )
        
        assert response.audio_file == "/tmp/test.wav"
        assert response.voice == "ljspeech"
        assert response.text_length == 10
        assert response.duration is None


if __name__ == "__main__":
    pytest.main([__file__])
