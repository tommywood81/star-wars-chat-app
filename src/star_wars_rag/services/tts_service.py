"""
Text-to-Speech service implementation using Coqui TTS.

This module provides a concrete implementation of the TTSService interface
using Coqui TTS for speech synthesis.
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import tempfile
import shutil
import os

from ..core.interfaces import TTSService
from ..core.exceptions import ServiceError, AudioProcessingError, ValidationError, VoiceNotFoundError
from ..core.logging import LoggerMixin


class CoquiTTSService(TTSService, LoggerMixin):
    """Text-to-Speech service using Coqui TTS."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the Coqui TTS service.
        
        Args:
            config: Configuration dictionary containing:
                - default_voice: Default voice model to use
                - cache_dir: Directory for TTS model cache
                - temp_dir: Temporary directory for processing
                
        Raises:
            ConfigurationError: If configuration is invalid
        """
        super().__init__(config)
        
        # Validate configuration
        self._validate_config()
        
        # Initialize TTS
        self.tts = None
        self.default_voice = self.config.get("default_voice", "ljspeech")
        self.cache_dir = Path(self.config.get("cache_dir", "/app/models/tts"))
        self.temp_dir = Path(self.config.get("temp_dir", "/tmp"))
        
        # Available voice models
        self.available_voices = {
            "ljspeech": "tts_models/en/ljspeech/tacotron2-DDC",
            "vctk": "tts_models/multilingual/multi-dataset/your_tts",
            "fastspeech2": "tts_models/en/ljspeech/fast_pitch"
        }
        
        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("initializing_tts_service", 
                        default_voice=self.default_voice,
                        cache_dir=str(self.cache_dir))
    
    def _validate_config(self) -> None:
        """Validate service configuration.
        
        Raises:
            ValidationError: If configuration is invalid
        """
        # No required fields for now - we'll use mock responses
        pass
    
    async def _load_model(self) -> None:
        """Load the TTS model asynchronously."""
        if self.tts is not None:
            return
        
        try:
            self.logger.info("loading_tts_model", voice=self.default_voice)
            
            # Import TTS
            from TTS.api import TTS
            
            # Load the model
            self.tts = TTS(self.available_voices[self.default_voice])
            
            self.logger.info("tts_model_loaded", voice=self.default_voice)
            
        except Exception as e:
            self.logger.error("failed_to_load_tts_model", 
                            voice=self.default_voice, 
                            error=str(e))
            raise ServiceError("TTS", "model_loading", f"Failed to load TTS model: {str(e)}")
    
    def _synthesize_audio(self, text: str, voice: str, output_path: Path) -> bool:
        """Synthesize audio using the actual TTS model."""
        try:
            # Generate audio using the TTS model
            self.tts.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker_wav=None,  # Use default voice
                language="en"
            )
            return True
        except Exception as e:
            self.logger.error("tts_synthesis_failed", error=str(e), text=text[:50])
            return False
    
    async def synthesize(self, text: str, voice: str = "ljspeech", output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Synthesize text to speech using TTS.
        
        Args:
            text: Text to synthesize
            voice: Voice model to use
            output_path: Optional output path for audio file
            
        Returns:
            Dictionary containing:
                - audio_path: Path to generated audio file
                - duration: Processing duration
                - text_length: Length of input text
                - voice: Voice used
                
        Raises:
            AudioProcessingError: If synthesis fails
            ValidationError: If input is invalid
            VoiceNotFoundError: If voice is not found
        """
        start_time = time.time()
        
        try:
            # Validate input
            self._validate_synthesis_input(text, voice)
            
            # Load model if not loaded
            await self._load_model()
            
            self.logger.info("starting_synthesis", 
                           text_length=len(text),
                           voice=voice)
            
            # Generate output path if not provided
            if output_path is None:
                output_path = self.temp_dir / f"synthesis_{int(time.time())}.wav"
            
            # Synthesize audio using the model
            success = self._synthesize_audio(text, voice, output_path)
            
            if not success:
                raise AudioProcessingError("TTS", "synthesis", "Failed to synthesize audio")
            
            duration = time.time() - start_time
            
            # Log performance metric
            from ..core.logging import log_performance_metric
            log_performance_metric(
                self.logger,
                "synthesis_duration",
                duration,
                "seconds",
                service="TTS",
                details={"text_length": len(text), "voice": voice}
            )
            
            self.logger.info("synthesis_completed", 
                           output_path=str(output_path),
                           duration=duration,
                           text_length=len(text))
            
            return {
                "audio_path": str(output_path),
                "duration": duration,
                "text_length": len(text),
                "voice": voice,
                "model": f"coqui-{voice}"
            }
            
        except Exception as e:
            duration = time.time() - start_time
            from ..core.logging import log_error_with_context
            log_error_with_context(
                self.logger,
                e,
                context={
                    "text": text,
                    "voice": voice,
                    "duration": duration
                }
            )
            raise AudioProcessingError("TTS", "synthesis", f"Failed to synthesize speech: {str(e)}")
    
    def _validate_synthesis_input(self, text: str, voice: str) -> None:
        """Validate synthesis input parameters.
        
        Args:
            text: Text to synthesize
            voice: Voice model name
            
        Raises:
            ValidationError: If input is invalid
            VoiceNotFoundError: If voice is not found
        """
        if not text or not text.strip():
            raise ValidationError("text", text, "Text cannot be empty")
        
        if not voice or not isinstance(voice, str):
            raise ValidationError("voice", voice, "Voice must be a non-empty string")
        
        if voice not in self.available_voices:
            raise VoiceNotFoundError(voice, f"Voice '{voice}' not found")
    
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices.
        
        Returns:
            List of voice information dictionaries
        """
        voices = []
        for name, model_path in self.available_voices.items():
            voices.append({
                "name": name,
                "model_path": model_path,
                "language": "en" if "en" in model_path else "multilingual"
            })
        return voices
    
    async def get_voice_info(self, voice: str) -> Dict[str, Any]:
        """Get information about a specific voice.
        
        Args:
            voice: Voice name
            
        Returns:
            Voice information dictionary
            
        Raises:
            VoiceNotFoundError: If voice is not found
        """
        if voice not in self.available_voices:
            raise VoiceNotFoundError(voice, f"Voice '{voice}' not found")
        
        return {
            "name": voice,
            "model_path": self.available_voices[voice],
            "language": "en" if "en" in self.available_voices[voice] else "multilingual"
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the TTS service.
        
        Returns:
            Health status dictionary
        """
        try:
            await self._load_model()
            
            return {
                "status": "healthy",
                "service": "TTS",
                "model_loaded": self.tts is not None,
                "default_voice": self.default_voice,
                "available_voices": len(self.available_voices),
                "model_type": "coqui-tts"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "service": "TTS",
                "error": str(e),
                "model_loaded": False
            }
