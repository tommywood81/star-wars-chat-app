"""
Core interfaces for the Star Wars RAG application.

This module defines the abstract base classes and interfaces that all
service implementations must adhere to.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging


class STTService(ABC):
    """Abstract base class for Speech-to-Text services."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the STT service with configuration.
        
        Args:
            config: Configuration dictionary containing service parameters
        """
        self.config = config
    
    @abstractmethod
    async def transcribe(self, audio_path: Path, language: str = "en") -> Dict[str, Any]:
        """Transcribe audio file to text.
        
        Args:
            audio_path: Path to the audio file
            language: Language code for transcription
            
        Returns:
            Dictionary containing transcription results
            
        Raises:
            ServiceError: If transcription fails
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check service health status.
        
        Returns:
            Dictionary containing health status information
        """
        pass


class TTSService(ABC):
    """Abstract base class for Text-to-Speech services."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the TTS service with configuration.
        
        Args:
            config: Configuration dictionary containing service parameters
        """
        self.config = config
    
    @abstractmethod
    async def synthesize(self, text: str, voice: str, output_path: Path) -> Dict[str, Any]:
        """Synthesize text to speech.
        
        Args:
            text: Text to synthesize
            voice: Voice model to use
            output_path: Path where audio file should be saved
            
        Returns:
            Dictionary containing synthesis results
            
        Raises:
            ServiceError: If synthesis fails
        """
        pass
    
    @abstractmethod
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices.
        
        Returns:
            List of voice dictionaries with metadata
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check service health status.
        
        Returns:
            Dictionary containing health status information
        """
        pass


class LLMService(ABC):
    """Abstract base class for Large Language Model services."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the LLM service with configuration.
        
        Args:
            config: Configuration dictionary containing service parameters
        """
        self.config = config
    
    @abstractmethod
    async def generate_response(
        self, 
        prompt: str, 
        character: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a response using the LLM.
        
        Args:
            prompt: Input prompt for the model
            character: Character persona to use
            context: Optional context information
            
        Returns:
            Dictionary containing the generated response
            
        Raises:
            ServiceError: If generation fails
        """
        pass
    
    @abstractmethod
    async def get_available_characters(self) -> List[Dict[str, Any]]:
        """Get list of available characters.
        
        Returns:
            List of character dictionaries with metadata
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check service health status.
        
        Returns:
            Dictionary containing health status information
        """
        pass


class ChatService(ABC):
    """Abstract base class for Chat orchestration services."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the chat service with configuration.
        
        Args:
            config: Configuration dictionary containing service parameters
        """
        self.config = config
    
    @abstractmethod
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
            Dictionary containing the complete response
            
        Raises:
            ServiceError: If processing fails
        """
        pass
    
    @abstractmethod
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
            Dictionary containing the response
            
        Raises:
            ServiceError: If processing fails
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check service health status.
        
        Returns:
            Dictionary containing health status information
        """
        pass
