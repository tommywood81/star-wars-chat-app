"""
Configuration management for the Star Wars RAG application.

This module provides a centralized configuration system using Pydantic
for validation and environment variable support.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
from pydantic import Field, validator
from pydantic_settings import BaseSettings
import os


class STTConfig(BaseSettings):
    """Configuration for Speech-to-Text service."""
    
    model_name: str = Field(default="base", description="Whisper model to use")
    language: str = Field(default="en", description="Default language for transcription")
    temp_dir: Path = Field(default=Path("/tmp"), description="Temporary directory for audio files")
    
    class Config:
        env_prefix = "STT_"


class TTSConfig(BaseSettings):
    """Configuration for Text-to-Speech service."""
    
    default_voice: str = Field(default="ljspeech", description="Default voice model")
    cache_dir: Path = Field(default=Path("/app/models/tts"), description="TTS model cache directory")
    temp_dir: Path = Field(default=Path("/tmp"), description="Temporary directory for audio files")
    
    class Config:
        env_prefix = "TTS_"


class LLMConfig(BaseSettings):
    """Configuration for Large Language Model service."""
    
    model_path: Path = Field(description="Path to the LLM model file")
    n_ctx: int = Field(default=2048, description="Context window size")
    n_threads: Optional[int] = Field(default=None, description="Number of threads to use")
    n_gpu_layers: int = Field(default=0, description="Number of GPU layers")
    verbose: bool = Field(default=False, description="Enable verbose logging")
    
    @validator('model_path')
    def validate_model_path(cls, v):
        if not v.exists():
            raise ValueError(f"Model path does not exist: {v}")
        return v
    
    class Config:
        env_prefix = "LLM_"


class CharacterConfig(BaseSettings):
    """Configuration for character management."""
    
    characters_file: Path = Field(default=Path("characters.json"), description="Path to characters configuration")
    default_character: str = Field(default="Luke Skywalker", description="Default character")
    
    class Config:
        env_prefix = "CHARACTER_"


class LoggingConfig(BaseSettings):
    """Configuration for logging."""
    
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    file_path: Optional[Path] = Field(default=None, description="Log file path")
    
    @validator('level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()
    
    class Config:
        env_prefix = "LOG_"


class AppConfig(BaseSettings):
    """Main application configuration."""
    
    # Service configurations
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    character: CharacterConfig = Field(default_factory=CharacterConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Application settings
    debug: bool = Field(default=False, description="Enable debug mode")
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=8000, description="Port to bind to")
    workers: int = Field(default=1, description="Number of worker processes")
    
    # Data directories
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    models_dir: Path = Field(default=Path("models"), description="Models directory")
    static_dir: Path = Field(default=Path("static"), description="Static files directory")
    
    @validator('data_dir', 'models_dir', 'static_dir')
    def create_directories(cls, v):
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def get_service_config(self, service_name: str) -> Dict[str, Any]:
        """Get configuration for a specific service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Dictionary containing service configuration
        """
        service_configs = {
            'stt': self.stt.dict(),
            'tts': self.tts.dict(),
            'llm': self.llm.dict(),
            'character': self.character.dict(),
        }
        
        if service_name not in service_configs:
            raise ValueError(f"Unknown service: {service_name}")
        
        return service_configs[service_name]
    
    def validate(self) -> None:
        """Validate the complete configuration."""
        # Check if required files exist
        if not self.llm.model_path.exists():
            raise ValueError(f"LLM model not found: {self.llm.model_path}")
        
        if not self.character.characters_file.exists():
            raise ValueError(f"Characters file not found: {self.character.characters_file}")


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Load application configuration from file and environment variables.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        AppConfig instance with loaded configuration
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    try:
        if config_path and config_path.exists():
            # Load from specific config file
            config = AppConfig(_env_file=config_path)
        else:
            # Load from environment variables and .env file
            config = AppConfig()
        
        config.validate()
        return config
        
    except Exception as e:
        raise ConfigurationError("app", f"Failed to load configuration: {str(e)}")


def get_default_config() -> AppConfig:
    """Get default configuration for testing and development.
    
    Returns:
        AppConfig instance with default values
    """
    return AppConfig(
        llm=LLMConfig(model_path=Path("models/phi-2.Q4_K_M.gguf")),
        debug=True
    )
