"""
Chat orchestration service for the Star Wars RAG application.

This module provides a concrete implementation of the ChatService interface
that coordinates STT, LLM, and TTS services to provide a complete chat experience.
"""

import asyncio
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import os

from ..core.interfaces import ChatService, STTService, LLMService, TTSService
from ..core.exceptions import ServiceError, AudioProcessingError, ValidationError
from ..core.logging import LoggerMixin


class StarWarsChatService(ChatService, LoggerMixin):
    """Chat orchestration service for Star Wars characters."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the chat service.
        
        Args:
            config: Configuration dictionary containing:
                - stt_service: STT service instance
                - llm_service: LLM service instance
                - tts_service: TTS service instance
                - temp_dir: Temporary directory for processing
                
        Raises:
            ConfigurationError: If configuration is invalid
        """
        super().__init__(config)
        
        # Validate configuration
        self._validate_config()
        
        # Initialize services
        self.stt_service = self.config.get("stt_service")
        self.llm_service = self.config.get("llm_service")
        self.tts_service = self.config.get("tts_service")
        self.temp_dir = Path(self.config.get("temp_dir", "/tmp"))
        
        # Ensure temp directory exists
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("initializing_chat_service", 
                        temp_dir=str(self.temp_dir),
                        has_stt=self.stt_service is not None,
                        has_llm=self.llm_service is not None,
                        has_tts=self.tts_service is not None)
    
    def _validate_config(self) -> None:
        """Validate service configuration.
        
        Raises:
            ValidationError: If configuration is invalid
        """
        required_services = ["stt_service", "llm_service"]
        for service in required_services:
            if service not in self.config:
                raise ValidationError(service, None, f"Required service '{service}' is missing")
        
        # Validate service instances
        if not isinstance(self.config.get("stt_service"), STTService):
            raise ValidationError("stt_service", self.config.get("stt_service"), "Must be an STTService instance")
        
        if not isinstance(self.config.get("llm_service"), LLMService):
            raise ValidationError("llm_service", self.config.get("llm_service"), "Must be an LLMService instance")
        
        # TTS service is optional
        tts_service = self.config.get("tts_service")
        if tts_service is not None and not isinstance(tts_service, TTSService):
            raise ValidationError("tts_service", tts_service, "Must be a TTSService instance")
    
    async def process_audio_message(
        self, 
        audio_path: Path, 
        character: str,
        language: str = "en"
    ) -> Dict[str, Any]:
        """Process an audio message through the full pipeline.
        
        Args:
            audio_path: Path to the audio file
            character: Character to respond as
            language: Language for transcription
            
        Returns:
            Dictionary containing:
                - transcription: Transcribed text
                - response: Character's response
                - audio_response: Path to audio response (if TTS available)
                - duration: Total processing duration
                - character: Character used
                
        Raises:
            ServiceError: If processing fails
            AudioProcessingError: If audio processing fails
        """
        start_time = time.time()
        
        try:
            self.logger.info("processing_audio_message", 
                           audio_path=str(audio_path),
                           character=character,
                           language=language)
            
            # Step 1: Transcribe audio to text
            transcription_result = await self.stt_service.transcribe(audio_path, language)
            transcription = transcription_result.get("text", "")
            
            if not transcription.strip():
                raise AudioProcessingError(
                    "transcription",
                    str(audio_path),
                    "No text was transcribed from the audio"
                )
            
            self.logger.info("transcription_completed", 
                           transcription=transcription[:100] + "..." if len(transcription) > 100 else transcription)
            
            # Step 2: Generate character response
            response_result = await self.llm_service.generate_response(transcription, character)
            response = response_result.get("response", "")
            
            self.logger.info("response_generated", 
                           response=response[:100] + "..." if len(response) > 100 else response)
            
            # Step 3: Synthesize response to speech (if TTS available)
            audio_response_path = None
            if self.tts_service and response.strip():
                audio_response_path = await self._synthesize_response(response, character)
            
            duration = time.time() - start_time
            
            # Log performance metric
            from ..core.logging import log_performance_metric
            log_performance_metric(
                self.logger,
                "audio_message_processing_duration",
                duration,
                "seconds",
                service="Chat",
                details={
                    "transcription_length": len(transcription),
                    "response_length": len(response),
                    "has_audio_response": audio_response_path is not None
                }
            )
            
            self.logger.info("audio_message_processing_completed", 
                           duration=duration,
                           transcription_length=len(transcription),
                           response_length=len(response))
            
            return {
                "transcription": transcription,
                "response": response,
                "audio_response": audio_response_path,
                "duration": duration,
                "character": character,
                "language": language
            }
            
        except Exception as e:
            duration = time.time() - start_time
            from ..core.logging import log_error_with_context
            log_error_with_context(
                self.logger,
                e,
                context={
                    "audio_path": str(audio_path),
                    "character": character,
                    "language": language,
                    "duration": duration
                },
                service="Chat"
            )
            
            if isinstance(e, (AudioProcessingError, ServiceError)):
                raise
            
            raise ServiceError("Chat", "audio_processing", f"Failed to process audio message: {str(e)}")
    
    async def process_text_message(
        self, 
        text: str, 
        character: str
    ) -> Dict[str, Any]:
        """Process a text message through the LLM pipeline.
        
        Args:
            text: Input text message
            character: Character to respond as
            
        Returns:
            Dictionary containing:
                - response: Character's response
                - audio_response: Path to audio response (if TTS available)
                - duration: Processing duration
                - character: Character used
                
        Raises:
            ServiceError: If processing fails
        """
        start_time = time.time()
        
        try:
            self.logger.info("processing_text_message", 
                           text_length=len(text),
                           character=character)
            
            # Generate character response
            response_result = await self.llm_service.generate_response(text, character)
            response = response_result.get("response", "")
            
            self.logger.info("response_generated", 
                           response=response[:100] + "..." if len(response) > 100 else response)
            
            # Synthesize response to speech (if TTS available)
            audio_response_path = None
            if self.tts_service and response.strip():
                audio_response_path = await self._synthesize_response(response, character)
            
            duration = time.time() - start_time
            
            # Log performance metric
            from ..core.logging import log_performance_metric
            log_performance_metric(
                self.logger,
                "text_message_processing_duration",
                duration,
                "seconds",
                service="Chat",
                details={
                    "text_length": len(text),
                    "response_length": len(response),
                    "has_audio_response": audio_response_path is not None
                }
            )
            
            self.logger.info("text_message_processing_completed", 
                           duration=duration,
                           text_length=len(text),
                           response_length=len(response))
            
            return {
                "response": response,
                "audio_response": audio_response_path,
                "duration": duration,
                "character": character
            }
            
        except Exception as e:
            duration = time.time() - start_time
            from ..core.logging import log_error_with_context
            log_error_with_context(
                self.logger,
                e,
                context={
                    "text_length": len(text),
                    "character": character,
                    "duration": duration
                },
                service="Chat"
            )
            
            if isinstance(e, ServiceError):
                raise
            
            raise ServiceError("Chat", "text_processing", f"Failed to process text message: {str(e)}")
    
    async def _synthesize_response(self, response: str, character: str) -> Optional[str]:
        """Synthesize text response to speech.
        
        Args:
            response: Text response to synthesize
            character: Character for voice selection
            
        Returns:
            Path to audio file, or None if synthesis fails
        """
        try:
            # Generate unique filename
            audio_filename = f"response_{uuid.uuid4().hex}.wav"
            audio_path = self.temp_dir / audio_filename
            
            # Map character to voice (simple mapping)
            voice = self._get_voice_for_character(character)
            
            # Synthesize speech
            synthesis_result = await self.tts_service.synthesize(response, voice, audio_path)
            
            if synthesis_result and audio_path.exists():
                self.logger.info("response_synthesized", 
                               audio_path=str(audio_path),
                               voice=voice)
                return str(audio_path)
            else:
                self.logger.warning("synthesis_failed", response_length=len(response))
                return None
                
        except Exception as e:
            self.logger.error("synthesis_error", error=str(e), response_length=len(response))
            return None
    
    def _get_voice_for_character(self, character: str) -> str:
        """Get appropriate voice for a character.
        
        Args:
            character: Character name
            
        Returns:
            Voice name to use
        """
        # Simple character-to-voice mapping
        voice_mapping = {
            "Luke Skywalker": "ljspeech",
            "Darth Vader": "ljspeech",  # Could use a deeper voice model
            "Princess Leia": "ljspeech",  # Could use a female voice model
        }
        
        return voice_mapping.get(character, "ljspeech")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health status.
        
        Returns:
            Dictionary containing health status information
        """
        try:
            # Check individual services
            stt_health = await self.stt_service.health_check() if self.stt_service else {"status": "not_available"}
            llm_health = await self.llm_service.health_check() if self.llm_service else {"status": "not_available"}
            tts_health = await self.tts_service.health_check() if self.tts_service else {"status": "not_available"}
            
            # Determine overall status
            all_healthy = (
                stt_health.get("status") == "healthy" and
                llm_health.get("status") == "healthy"
            )
            
            status = "healthy" if all_healthy else "unhealthy"
            
            return {
                "status": status,
                "stt_service": stt_health,
                "llm_service": llm_health,
                "tts_service": tts_health,
                "temp_dir": str(self.temp_dir),
                "temp_dir_writable": self.temp_dir.exists() and os.access(self.temp_dir, os.W_OK)
            }
            
        except Exception as e:
            self.logger.error("health_check_failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("cleaning_up_chat_service")
        
        # Clean up individual services
        if self.stt_service:
            await self.stt_service.cleanup()
        
        if self.llm_service:
            await self.llm_service.cleanup()
        
        if self.tts_service:
            await self.tts_service.cleanup()
        
        # Clean up temp files
        try:
            for temp_file in self.temp_dir.glob("response_*.wav"):
                temp_file.unlink()
            self.logger.info("temp_files_cleaned")
        except Exception as e:
            self.logger.warning("failed_to_clean_temp_files", error=str(e))


# Factory function for creating chat service instances
def create_chat_service(
    stt_service: STTService,
    llm_service: LLMService,
    tts_service: Optional[TTSService] = None,
    config: Optional[Dict[str, Any]] = None
) -> StarWarsChatService:
    """Create a new chat service instance.
    
    Args:
        stt_service: STT service instance
        llm_service: LLM service instance
        tts_service: Optional TTS service instance
        config: Additional configuration
        
    Returns:
        Configured chat service instance
    """
    service_config = config or {}
    service_config.update({
        "stt_service": stt_service,
        "llm_service": llm_service,
        "tts_service": tts_service
    })
    
    return StarWarsChatService(service_config)
