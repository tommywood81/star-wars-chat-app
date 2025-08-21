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
        # No required fields for TTS, all have defaults
        pass
    
    async def _load_model(self) -> None:
        """Load the TTS model asynchronously.
        
        Raises:
            ServiceError: If model loading fails
        """
        if self.tts is not None:
            return
        
        try:
            self.logger.info("loading_tts_model", voice=self.default_voice)
            
            # Import TTS in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.tts = await loop.run_in_executor(None, self._load_tts_model_sync)
            
            self.logger.info("tts_model_loaded", voice=self.default_voice)
            
        except Exception as e:
            self.logger.error("failed_to_load_tts_model", 
                            voice=self.default_voice, 
                            error=str(e))
            raise ServiceError("TTS", "model_loading", f"Failed to load TTS model: {str(e)}")
    
    def _load_tts_model_sync(self):
        """Load TTS model synchronously (to be run in executor)."""
        try:
            from TTS.api import TTS
            model_name = self.available_voices.get(self.default_voice, self.available_voices["ljspeech"])
            return TTS(model_name=model_name, progress_bar=False, gpu=False)
        except ImportError:
            raise ServiceError("TTS", "model_loading", "TTS library not installed")
    
    async def synthesize(self, text: str, voice: str, output_path: Path) -> Dict[str, Any]:
        """Synthesize text to speech using Coqui TTS.
        
        Args:
            text: Text to synthesize
            voice: Voice model to use
            output_path: Path where audio file should be saved
            
        Returns:
            Dictionary containing:
                - audio_path: Path to generated audio file
                - duration: Processing duration
                - voice: Voice used
                - text_length: Length of input text
                
        Raises:
            AudioProcessingError: If synthesis fails
            ValidationError: If input is invalid
            VoiceNotFoundError: If voice is not available
        """
        start_time = time.time()
        
        try:
            # Validate input
            self._validate_synthesis_input(text, voice, output_path)
            
            # Load model if not loaded
            await self._load_model()
            
            self.logger.info("starting_synthesis", 
                           text_length=len(text),
                           voice=voice,
                           output_path=str(output_path))
            
            # Run synthesis in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._synthesize_sync, 
                text, 
                voice, 
                output_path
            )
            
            duration = time.time() - start_time
            
            # Log performance metric
            self.log_performance_metric(
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
                "voice": voice,
                "text_length": len(text),
                "model": self.default_voice
            }
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_error_with_context(
                self.logger,
                e,
                context={
                    "text_length": len(text),
                    "voice": voice,
                    "output_path": str(output_path),
                    "duration": duration
                },
                service="TTS"
            )
            
            if isinstance(e, (ValidationError, AudioProcessingError, VoiceNotFoundError)):
                raise
            
            raise AudioProcessingError(
                "synthesis",
                str(output_path),
                f"Speech synthesis failed: {str(e)}"
            )
    
    def _synthesize_sync(self, text: str, voice: str, output_path: Path) -> Dict[str, Any]:
        """Synchronous synthesis method (to be run in executor).
        
        Args:
            text: Text to synthesize
            voice: Voice to use
            output_path: Output file path
            
        Returns:
            Synthesis result
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate speech
            self.tts.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker=None,  # Use default speaker
                language=None  # Use default language
            )
            
            return {"success": True}
        except Exception as e:
            raise AudioProcessingError(
                "synthesis",
                str(output_path),
                f"TTS synthesis failed: {str(e)}"
            )
    
    def _validate_synthesis_input(self, text: str, voice: str, output_path: Path) -> None:
        """Validate synthesis input parameters.
        
        Args:
            text: Text to synthesize
            voice: Voice model
            output_path: Output file path
            
        Raises:
            ValidationError: If input is invalid
            VoiceNotFoundError: If voice is not available
        """
        # Validate text
        if not text or not isinstance(text, str):
            raise ValidationError(
                "text",
                text,
                "Text must be a non-empty string"
            )
        
        if len(text) > 1000:  # Reasonable limit
            raise ValidationError(
                "text",
                text,
                "Text too long (max 1000 characters)"
            )
        
        # Validate voice
        if voice not in self.available_voices:
            available = list(self.available_voices.keys())
            raise VoiceNotFoundError(voice, available)
        
        # Validate output path
        if not isinstance(output_path, Path):
            raise ValidationError(
                "output_path",
                output_path,
                "Output path must be a Path object"
            )
    
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices.
        
        Returns:
            List of voice dictionaries with metadata
        """
        voices = []
        for voice_name, model_path in self.available_voices.items():
            voices.append({
                "name": voice_name,
                "model_path": model_path,
                "language": "en",  # Default assumption
                "description": f"Coqui TTS {voice_name} voice"
            })
        
        return voices
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health status.
        
        Returns:
            Dictionary containing health status information
        """
        try:
            # Check if model is loaded
            model_loaded = self.tts is not None
            
            # Check directories
            cache_dir_exists = self.cache_dir.exists()
            temp_dir_writable = self.temp_dir.exists() and os.access(self.temp_dir, os.W_OK)
            
            # Test model loading if not loaded
            if not model_loaded:
                try:
                    await self._load_model()
                    model_loaded = self.tts is not None
                except Exception:
                    model_loaded = False
            
            status = "healthy" if model_loaded and cache_dir_exists and temp_dir_writable else "unhealthy"
            
            return {
                "status": status,
                "model_loaded": model_loaded,
                "default_voice": self.default_voice,
                "cache_dir_exists": cache_dir_exists,
                "temp_dir_writable": temp_dir_writable,
                "available_voices": list(self.available_voices.keys())
            }
            
        except Exception as e:
            self.logger.error("health_check_failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_loaded": False,
                "default_voice": self.default_voice
            }
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("cleaning_up_tts_service")
        self.tts = None


# Factory function for creating TTS service instances
def create_tts_service(config: Dict[str, Any]) -> CoquiTTSService:
    """Create a new TTS service instance.
    
    Args:
        config: Service configuration
        
    Returns:
        Configured TTS service instance
    """
    return CoquiTTSService(config)
