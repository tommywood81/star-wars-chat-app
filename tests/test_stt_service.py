"""
Unit tests for the Speech-to-Text service.

This module contains comprehensive tests for the WhisperSTTService class,
including happy path, error cases, and edge cases.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any

from src.star_wars_rag.services.stt_service import WhisperSTTService
from src.star_wars_rag.core.exceptions import (
    ServiceError, 
    AudioProcessingError, 
    ValidationError,
    ConfigurationError
)


class TestWhisperSTTService:
    """Test cases for WhisperSTTService."""
    
    @pytest.fixture
    def valid_config(self) -> Dict[str, Any]:
        """Provide valid configuration for testing."""
        return {
            "model_name": "base",
            "language": "en",
            "temp_dir": "/tmp"
        }
    
    @pytest.fixture
    def mock_whisper_model(self):
        """Mock Whisper model."""
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            "text": "Hello, this is a test transcription",
            "language": "en",
            "confidence": 0.95
        }
        return mock_model
    
    @pytest.fixture
    def temp_audio_file(self) -> Path:
        """Create a temporary audio file for testing."""
        temp_dir = tempfile.mkdtemp()
        audio_file = Path(temp_dir) / "test_audio.wav"
        
        # Create a dummy audio file
        with open(audio_file, 'wb') as f:
            f.write(b"dummy audio content")
        
        yield audio_file
        
        # Cleanup
        os.unlink(audio_file)
        os.rmdir(temp_dir)
    
    def test_init_with_valid_config(self, valid_config):
        """Test service initialization with valid configuration."""
        service = WhisperSTTService(valid_config)
        
        assert service.model_name == "base"
        assert service.language == "en"
        assert service.temp_dir == Path("/tmp")
        assert service.model is None
    
    def test_init_with_invalid_model_name(self):
        """Test service initialization with invalid model name."""
        config = {"model_name": "invalid_model"}
        
        with pytest.raises(ValidationError) as exc_info:
            WhisperSTTService(config)
        
        assert "Invalid model name" in str(exc_info.value)
        assert exc_info.value.field == "model_name"
    
    def test_init_with_missing_model_name(self):
        """Test service initialization with missing model name."""
        config = {}
        
        with pytest.raises(ValidationError) as exc_info:
            WhisperSTTService(config)
        
        assert "Required field 'model_name' is missing" in str(exc_info.value)
    
    def test_validate_audio_input_valid_file(self, valid_config, temp_audio_file):
        """Test audio input validation with valid file."""
        service = WhisperSTTService(valid_config)
        
        # Should not raise any exception
        service._validate_audio_input(temp_audio_file, "en")
    
    def test_validate_audio_input_nonexistent_file(self, valid_config):
        """Test audio input validation with nonexistent file."""
        service = WhisperSTTService(valid_config)
        nonexistent_file = Path("/nonexistent/audio.wav")
        
        with pytest.raises(ValidationError) as exc_info:
            service._validate_audio_input(nonexistent_file, "en")
        
        assert "Audio file does not exist" in str(exc_info.value)
    
    def test_validate_audio_input_directory(self, valid_config):
        """Test audio input validation with directory path."""
        service = WhisperSTTService(valid_config)
        temp_dir = tempfile.mkdtemp()
        
        try:
            with pytest.raises(ValidationError) as exc_info:
                service._validate_audio_input(Path(temp_dir), "en")
            
            assert "Path is not a file" in str(exc_info.value)
        finally:
            os.rmdir(temp_dir)
    
    def test_validate_audio_input_large_file(self, valid_config):
        """Test audio input validation with file too large."""
        service = WhisperSTTService(valid_config)
        
        # Create a large dummy file
        temp_dir = tempfile.mkdtemp()
        large_file = Path(temp_dir) / "large_audio.wav"
        
        try:
            # Create a file larger than 25MB
            with open(large_file, 'wb') as f:
                f.write(b"0" * (26 * 1024 * 1024))  # 26MB
            
            with pytest.raises(ValidationError) as exc_info:
                service._validate_audio_input(large_file, "en")
            
            assert "File too large" in str(exc_info.value)
        finally:
            os.unlink(large_file)
            os.rmdir(temp_dir)
    
    def test_validate_audio_input_invalid_language(self, valid_config, temp_audio_file):
        """Test audio input validation with invalid language code."""
        service = WhisperSTTService(valid_config)
        
        with pytest.raises(ValidationError) as exc_info:
            service._validate_audio_input(temp_audio_file, "invalid")
        
        assert "Language must be a 2-character language code" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_load_model_success(self, valid_config, mock_whisper_model):
        """Test successful model loading."""
        service = WhisperSTTService(valid_config)
        
        with patch('src.star_wars_rag.services.stt_service.whisper') as mock_whisper:
            mock_whisper.load_model.return_value = mock_whisper_model
            
            await service._load_model()
            
            assert service.model == mock_whisper_model
            mock_whisper.load_model.assert_called_once_with("base")
    
    @pytest.mark.asyncio
    async def test_load_model_import_error(self, valid_config):
        """Test model loading with import error."""
        service = WhisperSTTService(valid_config)
        
        with patch('src.star_wars_rag.services.stt_service.whisper', side_effect=ImportError("No module named 'whisper'")):
            with pytest.raises(ServiceError) as exc_info:
                await service._load_model()
            
            assert "Whisper library not installed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_transcribe_success(self, valid_config, temp_audio_file, mock_whisper_model):
        """Test successful transcription."""
        service = WhisperSTTService(valid_config)
        
        with patch('src.star_wars_rag.services.stt_service.whisper') as mock_whisper:
            mock_whisper.load_model.return_value = mock_whisper_model
            
            result = await service.transcribe(temp_audio_file, "en")
            
            assert result["text"] == "Hello, this is a test transcription"
            assert result["language"] == "en"
            assert result["confidence"] == 0.95
            assert "duration" in result
            assert result["model"] == "base"
    
    @pytest.mark.asyncio
    async def test_transcribe_with_default_language(self, valid_config, temp_audio_file, mock_whisper_model):
        """Test transcription with default language."""
        service = WhisperSTTService(valid_config)
        
        with patch('src.star_wars_rag.services.stt_service.whisper') as mock_whisper:
            mock_whisper.load_model.return_value = mock_whisper_model
            
            result = await service.transcribe(temp_audio_file)
            
            assert result["language"] == "en"  # Should use config default
            mock_whisper_model.transcribe.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_transcribe_invalid_input(self, valid_config):
        """Test transcription with invalid input."""
        service = WhisperSTTService(valid_config)
        nonexistent_file = Path("/nonexistent/audio.wav")
        
        with pytest.raises(AudioProcessingError) as exc_info:
            await service.transcribe(nonexistent_file, "en")
        
        assert "Audio file does not exist" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_transcribe_whisper_error(self, valid_config, temp_audio_file, mock_whisper_model):
        """Test transcription when Whisper fails."""
        service = WhisperSTTService(valid_config)
        
        # Make the model raise an exception
        mock_whisper_model.transcribe.side_effect = Exception("Whisper internal error")
        
        with patch('src.star_wars_rag.services.stt_service.whisper') as mock_whisper:
            mock_whisper.load_model.return_value = mock_whisper_model
            
            with pytest.raises(AudioProcessingError) as exc_info:
                await service.transcribe(temp_audio_file, "en")
            
            assert "Whisper transcription failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, valid_config, mock_whisper_model):
        """Test health check when service is healthy."""
        service = WhisperSTTService(valid_config)
        
        with patch('src.star_wars_rag.services.stt_service.whisper') as mock_whisper:
            mock_whisper.load_model.return_value = mock_whisper_model
            
            health = await service.health_check()
            
            assert health["status"] == "healthy"
            assert health["model_loaded"] is True
            assert health["model_name"] == "base"
            assert health["temp_dir_writable"] is True
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, valid_config):
        """Test health check when service is unhealthy."""
        service = WhisperSTTService(valid_config)
        
        # Mock os.access to return False
        with patch('os.access', return_value=False):
            health = await service.health_check()
            
            assert health["status"] == "unhealthy"
            assert health["temp_dir_writable"] is False
    
    @pytest.mark.asyncio
    async def test_health_check_with_error(self, valid_config):
        """Test health check when an error occurs."""
        service = WhisperSTTService(valid_config)
        
        # Make model loading fail
        with patch('src.star_wars_rag.services.stt_service.whisper', side_effect=Exception("Test error")):
            health = await service.health_check()
            
            assert health["status"] == "unhealthy"
            assert "error" in health
            assert health["model_loaded"] is False
    
    def test_cleanup(self, valid_config):
        """Test service cleanup."""
        service = WhisperSTTService(valid_config)
        service.model = Mock()  # Set a mock model
        
        service.cleanup()
        
        assert service.model is None


class TestSTTServiceFactory:
    """Test cases for STT service factory function."""
    
    def test_create_stt_service(self):
        """Test factory function creates service correctly."""
        config = {"model_name": "base", "language": "en"}
        
        service = create_stt_service(config)
        
        assert isinstance(service, WhisperSTTService)
        assert service.model_name == "base"
        assert service.language == "en"


class TestSTTServiceIntegration:
    """Integration tests for STT service."""
    
    @pytest.mark.asyncio
    async def test_full_transcription_workflow(self):
        """Test complete transcription workflow."""
        config = {"model_name": "base", "language": "en", "temp_dir": "/tmp"}
        
        # Create temporary audio file
        temp_dir = tempfile.mkdtemp()
        audio_file = Path(temp_dir) / "test_audio.wav"
        
        try:
            # Create dummy audio content
            with open(audio_file, 'wb') as f:
                f.write(b"dummy audio content")
            
            service = WhisperSTTService(config)
            
            # Mock Whisper to avoid actual model loading
            mock_model = Mock()
            mock_model.transcribe.return_value = {
                "text": "Test transcription result",
                "language": "en",
                "confidence": 0.9
            }
            
            with patch('src.star_wars_rag.services.stt_service.whisper') as mock_whisper:
                mock_whisper.load_model.return_value = mock_model
                
                # Test transcription
                result = await service.transcribe(audio_file, "en")
                
                assert result["text"] == "Test transcription result"
                assert result["language"] == "en"
                assert result["confidence"] == 0.9
                assert "duration" in result
                
                # Test health check
                health = await service.health_check()
                assert health["status"] == "healthy"
                
                # Test cleanup
                service.cleanup()
                assert service.model is None
                
        finally:
            # Cleanup
            if audio_file.exists():
                os.unlink(audio_file)
            os.rmdir(temp_dir)


# Performance tests
class TestSTTServicePerformance:
    """Performance tests for STT service."""
    
    @pytest.mark.asyncio
    async def test_transcription_performance_logging(self, valid_config, temp_audio_file, mock_whisper_model):
        """Test that performance metrics are logged during transcription."""
        service = WhisperSTTService(valid_config)
        
        with patch('src.star_wars_rag.services.stt_service.whisper') as mock_whisper:
            mock_whisper.load_model.return_value = mock_whisper_model
            
            # Mock the performance logging function
            with patch.object(service, 'log_performance_metric') as mock_log:
                await service.transcribe(temp_audio_file, "en")
                
                # Verify performance metric was logged
                mock_log.assert_called_once()
                call_args = mock_log.call_args
                assert call_args[0][1] == "transcription_duration"
                assert call_args[0][3] == "seconds"
                assert call_args[0][4] == "STT"


# Error handling tests
class TestSTTServiceErrorHandling:
    """Error handling tests for STT service."""
    
    @pytest.mark.asyncio
    async def test_error_logging_with_context(self, valid_config):
        """Test that errors are logged with proper context."""
        service = WhisperSTTService(valid_config)
        nonexistent_file = Path("/nonexistent/audio.wav")
        
        with patch.object(service, 'log_error_with_context') as mock_log:
            try:
                await service.transcribe(nonexistent_file, "en")
            except AudioProcessingError:
                pass
            
            # Verify error was logged with context
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[0][1].__class__.__name__ == "ValidationError"
            assert "audio_path" in call_args[0][2]
            assert call_args[0][3] == "STT"
