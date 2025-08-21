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
        # No required fields for now - we'll use mock responses
        pass
    
    async def _load_model(self) -> None:
        """Load the Whisper model asynchronously."""
        if self.model is not None:
            return
        
        try:
            self.logger.info("loading_whisper_model", model_name=self.model_name)
            
            # Import whisper
            import whisper
            
            # Load the model
            self.model = whisper.load_model(self.model_name)
            
            self.logger.info("whisper_model_loaded", model_name=self.model_name)
            
        except Exception as e:
            self.logger.error("failed_to_load_whisper_model", 
                            model_name=self.model_name, 
                            error=str(e))
            raise ServiceError("STT", "model_loading", f"Failed to load Whisper model: {str(e)}")
    
    def _transcribe_audio(self, audio_path: Path, language: str) -> str:
        """Transcribe audio using the actual Whisper model."""
        # Transcribe the audio file
        result = self.model.transcribe(
            str(audio_path),
            language=language,
            fp16=False  # Use CPU for compatibility
        )
        
        return result["text"].strip()
    
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
            
            # Transcribe audio using the model
            transcribed_text = self._transcribe_audio(audio_path, language)
            
            duration = time.time() - start_time
            
            # Log performance metric
            from ..core.logging import log_performance_metric
            log_performance_metric(
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
                           text_length=len(transcribed_text))
            
            return {
                "text": transcribed_text,
                "language": language,
                "confidence": 0.95,  # Mock confidence
                "duration": duration,
                "model": f"whisper-{self.model_name}"
            }
            
        except Exception as e:
            duration = time.time() - start_time
            from ..core.logging import log_error_with_context
            log_error_with_context(
                self.logger,
                e,
                context={
                    "audio_path": str(audio_path),
                    "language": language,
                    "duration": duration
                }
            )
            raise AudioProcessingError("STT", "transcription", f"Failed to transcribe audio: {str(e)}")
    
    def _validate_audio_input(self, audio_path: Path, language: str) -> None:
        """Validate audio input parameters.
        
        Args:
            audio_path: Path to audio file
            language: Language code
            
        Raises:
            ValidationError: If input is invalid
        """
        if not audio_path or not isinstance(audio_path, Path):
            raise ValidationError("audio_path", audio_path, "Audio path must be a valid Path object")
        
        if not language or not isinstance(language, str):
            raise ValidationError("language", language, "Language must be a non-empty string")
        
        # For mock service, we don't need to check if file exists
        # In real implementation, you would check: if not audio_path.exists():
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the STT service.
        
        Returns:
            Health status dictionary
        """
        try:
            await self._load_model()
            
            return {
                "status": "healthy",
                "service": "STT",
                "model_loaded": self.model is not None,
                "model_name": self.model_name,
                "language": self.language,
                "model_type": "whisper"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "service": "STT",
                "error": str(e),
                "model_loaded": False
            }
