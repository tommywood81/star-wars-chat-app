"""
Speech-to-Text service implementation using OpenAI Whisper.

This module provides a concrete implementation of the STTService interface
using OpenAI's Whisper model for speech transcription.
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import shutil
import os

from ..core.interfaces import STTService
from ..core.exceptions import ServiceError, AudioProcessingError, ValidationError
from ..core.logging import LoggerMixin


class WhisperSTTService(STTService, LoggerMixin):
    """Speech-to-Text service using OpenAI Whisper."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the Whisper STT service.
        
        Args:
            config: Configuration dictionary containing:
                - model_name: Whisper model to use (default: "base")
                - language: Default language for transcription
                - temp_dir: Temporary directory for processing
                
        Raises:
            ConfigurationError: If configuration is invalid
        """
        super().__init__(config)
        
        # Validate configuration
        self._validate_config()
        
        # Initialize model
        self.model = None
        self.model_name = self.config.get("model_name", "base")
        self.language = self.config.get("language", "en")
        self.temp_dir = Path(self.config.get("temp_dir", "/tmp"))
        
        # Ensure temp directory exists
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("initializing_whisper_service", 
                        model_name=self.model_name, 
                        language=self.language)
    
    def _validate_config(self) -> None:
        """Validate service configuration.
        
        Raises:
            ValidationError: If configuration is invalid
        """
        required_fields = ["model_name"]
        for field in required_fields:
            if field not in self.config:
                raise ValidationError(field, None, f"Required field '{field}' is missing")
        
        # Validate model name
        valid_models = ["tiny", "base", "small", "medium", "large"]
        model_name = self.config.get("model_name")
        if model_name not in valid_models:
            raise ValidationError(
                "model_name", 
                model_name, 
                f"Invalid model name. Must be one of: {valid_models}"
            )
    
    async def _load_model(self) -> None:
        """Load the Whisper model asynchronously.
        
        Raises:
            ServiceError: If model loading fails
        """
        if self.model is not None:
            return
        
        try:
            self.logger.info("loading_whisper_model", model_name=self.model_name)
            
            # Import whisper in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(None, self._load_whisper_model_sync)
            
            self.logger.info("whisper_model_loaded", model_name=self.model_name)
            
        except Exception as e:
            self.logger.error("failed_to_load_whisper_model", 
                            model_name=self.model_name, 
                            error=str(e))
            raise ServiceError("STT", "model_loading", f"Failed to load Whisper model: {str(e)}")
    
    def _load_whisper_model_sync(self):
        """Load Whisper model synchronously (to be run in executor)."""
        try:
            import whisper
            return whisper.load_model(self.model_name)
        except ImportError:
            raise ServiceError("STT", "model_loading", "Whisper library not installed")
    
    async def transcribe(self, audio_path: Path, language: str = "en") -> Dict[str, Any]:
        """Transcribe audio file to text using Whisper.
        
        Args:
            audio_path: Path to the audio file
            language: Language code for transcription (defaults to config language)
            
        Returns:
            Dictionary containing:
                - text: Transcribed text
                - language: Detected language
                - confidence: Confidence score
                - duration: Processing duration
                
        Raises:
            AudioProcessingError: If audio processing fails
            ValidationError: If input is invalid
        """
        start_time = time.time()
        
        try:
            # Validate input
            self._validate_audio_input(audio_path, language)
            
            # Load model if not loaded
            await self._load_model()
            
            self.logger.info("starting_transcription", 
                           audio_path=str(audio_path), 
                           language=language)
            
            # Run transcription in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._transcribe_sync, 
                audio_path, 
                language
            )
            
            duration = time.time() - start_time
            
            # Log performance metric
            self.log_performance_metric(
                self.logger,
                "transcription_duration",
                duration,
                "seconds",
                service="STT",
                details={"audio_path": str(audio_path), "language": language}
            )
            
            self.logger.info("transcription_completed", 
                           audio_path=str(audio_path),
                           duration=duration,
                           text_length=len(result.get("text", "")))
            
            return {
                "text": result.get("text", ""),
                "language": result.get("language", language),
                "confidence": result.get("confidence", 0.0),
                "duration": duration,
                "model": self.model_name
            }
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_error_with_context(
                self.logger,
                e,
                context={
                    "audio_path": str(audio_path),
                    "language": language,
                    "duration": duration
                },
                service="STT"
            )
            
            if isinstance(e, (ValidationError, AudioProcessingError)):
                raise
            
            raise AudioProcessingError(
                "transcription",
                str(audio_path),
                f"Transcription failed: {str(e)}"
            )
    
    def _transcribe_sync(self, audio_path: Path, language: str) -> Dict[str, Any]:
        """Synchronous transcription method (to be run in executor).
        
        Args:
            audio_path: Path to the audio file
            language: Language code
            
        Returns:
            Whisper transcription result
        """
        try:
            result = self.model.transcribe(
                str(audio_path),
                language=language,
                task="transcribe",
                fp16=False
            )
            return result
        except Exception as e:
            raise AudioProcessingError(
                "transcription",
                str(audio_path),
                f"Whisper transcription failed: {str(e)}"
            )
    
    def _validate_audio_input(self, audio_path: Path, language: str) -> None:
        """Validate audio input parameters.
        
        Args:
            audio_path: Path to audio file
            language: Language code
            
        Raises:
            ValidationError: If input is invalid
        """
        # Check if file exists
        if not audio_path.exists():
            raise ValidationError(
                "audio_path",
                str(audio_path),
                "Audio file does not exist"
            )
        
        # Check if file is readable
        if not audio_path.is_file():
            raise ValidationError(
                "audio_path",
                str(audio_path),
                "Path is not a file"
            )
        
        # Check file size (max 25MB for Whisper)
        file_size = audio_path.stat().st_size
        max_size = 25 * 1024 * 1024  # 25MB
        if file_size > max_size:
            raise ValidationError(
                "audio_path",
                str(audio_path),
                f"File too large ({file_size} bytes). Maximum size is {max_size} bytes"
            )
        
        # Validate language code (basic check)
        if not isinstance(language, str) or len(language) != 2:
            raise ValidationError(
                "language",
                language,
                "Language must be a 2-character language code"
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health status.
        
        Returns:
            Dictionary containing health status information
        """
        try:
            # Check if model is loaded
            model_loaded = self.model is not None
            
            # Check temp directory
            temp_dir_writable = self.temp_dir.exists() and os.access(self.temp_dir, os.W_OK)
            
            # Test model loading if not loaded
            if not model_loaded:
                await self._load_model()
                model_loaded = self.model is not None
            
            status = "healthy" if model_loaded and temp_dir_writable else "unhealthy"
            
            return {
                "status": status,
                "model_loaded": model_loaded,
                "model_name": self.model_name,
                "temp_dir_writable": temp_dir_writable,
                "temp_dir": str(self.temp_dir),
                "language": self.language
            }
            
        except Exception as e:
            self.logger.error("health_check_failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_loaded": False,
                "model_name": self.model_name
            }
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("cleaning_up_stt_service")
        self.model = None


# Factory function for creating STT service instances
def create_stt_service(config: Dict[str, Any]) -> WhisperSTTService:
    """Create a new STT service instance.
    
    Args:
        config: Service configuration
        
    Returns:
        Configured STT service instance
    """
    return WhisperSTTService(config)
