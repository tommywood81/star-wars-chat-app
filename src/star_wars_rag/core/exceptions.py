"""
Custom exceptions for the Star Wars RAG application.

This module defines application-specific exceptions that provide
clear error handling and debugging information.
"""

from typing import Optional, Any, Dict


class StarWarsRAGError(Exception):
    """Base exception for all Star Wars RAG application errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the exception.
        
        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ServiceError(StarWarsRAGError):
    """Exception raised when a service operation fails."""
    
    def __init__(self, service_name: str, operation: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize service error.
        
        Args:
            service_name: Name of the service that failed
            operation: Operation that failed
            message: Error message
            details: Optional error details
        """
        full_message = f"{service_name} service failed during {operation}: {message}"
        super().__init__(full_message, details)
        self.service_name = service_name
        self.operation = operation


class ConfigurationError(StarWarsRAGError):
    """Exception raised when configuration is invalid or missing."""
    
    def __init__(self, config_key: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize configuration error.
        
        Args:
            config_key: Configuration key that caused the error
            message: Error message
            details: Optional error details
        """
        full_message = f"Configuration error for '{config_key}': {message}"
        super().__init__(full_message, details)
        self.config_key = config_key


class ValidationError(StarWarsRAGError):
    """Exception raised when input validation fails."""
    
    def __init__(self, field: str, value: Any, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize validation error.
        
        Args:
            field: Field name that failed validation
            value: Value that failed validation
            message: Error message
            details: Optional error details
        """
        full_message = f"Validation error for field '{field}' with value '{value}': {message}"
        super().__init__(full_message, details)
        self.field = field
        self.value = value


class ModelError(StarWarsRAGError):
    """Exception raised when model operations fail."""
    
    def __init__(self, model_name: str, operation: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize model error.
        
        Args:
            model_name: Name of the model that failed
            operation: Operation that failed
            message: Error message
            details: Optional error details
        """
        full_message = f"Model '{model_name}' failed during {operation}: {message}"
        super().__init__(full_message, details)
        self.model_name = model_name
        self.operation = operation


class AudioProcessingError(StarWarsRAGError):
    """Exception raised when audio processing fails."""
    
    def __init__(self, operation: str, file_path: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize audio processing error.
        
        Args:
            operation: Audio operation that failed
            file_path: Path to the audio file
            message: Error message
            details: Optional error details
        """
        full_message = f"Audio processing failed during {operation} for file '{file_path}': {message}"
        super().__init__(full_message, details)
        self.operation = operation
        self.file_path = file_path


class CharacterNotFoundError(StarWarsRAGError):
    """Exception raised when a requested character is not found."""
    
    def __init__(self, character_name: str, available_characters: Optional[list] = None) -> None:
        """Initialize character not found error.
        
        Args:
            character_name: Name of the character that was not found
            available_characters: List of available character names
        """
        message = f"Character '{character_name}' not found"
        if available_characters:
            message += f". Available characters: {', '.join(available_characters)}"
        
        super().__init__(message, {"available_characters": available_characters})
        self.character_name = character_name
        self.available_characters = available_characters


class VoiceNotFoundError(StarWarsRAGError):
    """Exception raised when a requested voice is not found."""
    
    def __init__(self, voice_name: str, available_voices: Optional[list] = None) -> None:
        """Initialize voice not found error.
        
        Args:
            voice_name: Name of the voice that was not found
            available_voices: List of available voice names
        """
        message = f"Voice '{voice_name}' not found"
        if available_voices:
            message += f". Available voices: {', '.join(available_voices)}"
        
        super().__init__(message, {"available_voices": available_voices})
        self.voice_name = voice_name
        self.available_voices = available_voices
