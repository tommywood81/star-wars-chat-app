"""
Large Language Model service implementation using local models.

This module provides a concrete implementation of the LLMService interface
using local LLM models for text generation.
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

from ..core.interfaces import LLMService
from ..core.exceptions import ServiceError, ModelError, ValidationError, CharacterNotFoundError
from ..core.logging import LoggerMixin


class LocalLLMService(LLMService, LoggerMixin):
    """Large Language Model service using local models."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the Local LLM service.
        
        Args:
            config: Configuration dictionary containing:
                - model_path: Path to the LLM model file
                - n_ctx: Context window size
                - n_threads: Number of threads to use
                - n_gpu_layers: Number of GPU layers
                - verbose: Enable verbose logging
                
        Raises:
            ConfigurationError: If configuration is invalid
        """
        super().__init__(config)
        
        # Validate configuration
        self._validate_config()
        
        # Initialize model
        self.llm = None
        self.model_path = Path(self.config["model_path"])
        self.n_ctx = self.config.get("n_ctx", 2048)
        self.n_threads = self.config.get("n_threads")
        self.n_gpu_layers = self.config.get("n_gpu_layers", 0)
        self.verbose = self.config.get("verbose", False)
        
        # Character management
        self.characters = {}
        self.characters_file = Path("characters.json")
        self._load_characters()
        
        self.logger.info("initializing_llm_service", 
                        model_path=str(self.model_path),
                        n_ctx=self.n_ctx)
    
    def _validate_config(self) -> None:
        """Validate service configuration.
        
        Raises:
            ValidationError: If configuration is invalid
        """
        required_fields = ["model_path"]
        for field in required_fields:
            if field not in self.config:
                raise ValidationError(field, None, f"Required field '{field}' is missing")
        
        # Validate model path
        model_path = Path(self.config["model_path"])
        if not model_path.exists():
            raise ValidationError(
                "model_path",
                str(model_path),
                "Model file does not exist"
            )
    
    def _load_characters(self) -> None:
        """Load character definitions from file."""
        try:
            if self.characters_file.exists():
                with open(self.characters_file, 'r') as f:
                    self.characters = json.load(f)
                self.logger.info("characters_loaded", count=len(self.characters))
            else:
                # Create default characters
                self.characters = {
                    "Luke Skywalker": {
                        "description": "A young Jedi Knight with a strong connection to the Force",
                        "personality": "Optimistic, brave, and determined to do what's right",
                        "speaking_style": "Direct and earnest, often asking questions"
                    },
                    "Darth Vader": {
                        "description": "A powerful Sith Lord and former Jedi",
                        "personality": "Intimidating, authoritative, and conflicted",
                        "speaking_style": "Deep, commanding voice with dramatic pauses"
                    },
                    "Princess Leia": {
                        "description": "A brave leader and diplomat",
                        "personality": "Strong-willed, intelligent, and compassionate",
                        "speaking_style": "Confident and articulate, often using wit"
                    }
                }
                self.logger.info("default_characters_created", count=len(self.characters))
        except Exception as e:
            self.logger.error("failed_to_load_characters", error=str(e))
            self.characters = {}
    
    async def _load_model(self) -> None:
        """Load the LLM model asynchronously.
        
        Raises:
            ServiceError: If model loading fails
        """
        if self.llm is not None:
            return
        
        try:
            self.logger.info("loading_llm_model", model_path=str(self.model_path))
            
            # Import and load model in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.llm = await loop.run_in_executor(None, self._load_llm_model_sync)
            
            self.logger.info("llm_model_loaded", model_path=str(self.model_path))
            
        except Exception as e:
            self.logger.error("failed_to_load_llm_model", 
                            model_path=str(self.model_path), 
                            error=str(e))
            raise ServiceError("LLM", "model_loading", f"Failed to load LLM model: {str(e)}")
    
    def _load_llm_model_sync(self):
        """Load LLM model synchronously (to be run in executor)."""
        try:
            from ..llm import LocalLLM
            return LocalLLM(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose
            )
        except ImportError:
            raise ServiceError("LLM", "model_loading", "LLM library not installed")
    
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
            Dictionary containing:
                - response: Generated text response
                - character: Character used
                - duration: Processing duration
                - prompt_length: Length of input prompt
                
        Raises:
            ModelError: If generation fails
            ValidationError: If input is invalid
            CharacterNotFoundError: If character is not found
        """
        start_time = time.time()
        
        try:
            # Validate input
            self._validate_generation_input(prompt, character)
            
            # Load model if not loaded
            await self._load_model()
            
            # Build character-specific prompt
            full_prompt = self._build_character_prompt(prompt, character, context)
            
            self.logger.info("starting_generation", 
                           prompt_length=len(prompt),
                           character=character,
                           full_prompt_length=len(full_prompt))
            
            # Run generation in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                self._generate_sync, 
                full_prompt
            )
            
            duration = time.time() - start_time
            
            # Log performance metric
            self.log_performance_metric(
                self.logger,
                "generation_duration",
                duration,
                "seconds",
                service="LLM",
                details={"prompt_length": len(prompt), "character": character}
            )
            
            self.logger.info("generation_completed", 
                           character=character,
                           duration=duration,
                           response_length=len(response))
            
            return {
                "response": response,
                "character": character,
                "duration": duration,
                "prompt_length": len(prompt),
                "model": str(self.model_path)
            }
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_error_with_context(
                self.logger,
                e,
                context={
                    "prompt_length": len(prompt),
                    "character": character,
                    "duration": duration
                },
                service="LLM"
            )
            
            if isinstance(e, (ValidationError, ModelError, CharacterNotFoundError)):
                raise
            
            raise ModelError(
                str(self.model_path),
                "generation",
                f"Response generation failed: {str(e)}"
            )
    
    def _generate_sync(self, prompt: str) -> str:
        """Synchronous generation method (to be run in executor).
        
        Args:
            prompt: Full prompt for generation
            
        Returns:
            Generated text response
        """
        try:
            response = self.llm.generate(prompt, max_tokens=500, temperature=0.7)
            return response.strip()
        except Exception as e:
            raise ModelError(
                str(self.model_path),
                "generation",
                f"LLM generation failed: {str(e)}"
            )
    
    def _build_character_prompt(self, prompt: str, character: str, context: Optional[str] = None) -> str:
        """Build a character-specific prompt.
        
        Args:
            prompt: User's input prompt
            character: Character to respond as
            context: Optional context information
            
        Returns:
            Formatted prompt for the LLM
        """
        if character not in self.characters:
            raise CharacterNotFoundError(character, list(self.characters.keys()))
        
        char_info = self.characters[character]
        
        # Build character prompt
        char_prompt = f"""You are {character}, a character from Star Wars.

Description: {char_info['description']}
Personality: {char_info['personality']}
Speaking Style: {char_info['speaking_style']}

{context or ''}

User: {prompt}

{character}:"""
        
        return char_prompt
    
    def _validate_generation_input(self, prompt: str, character: str) -> None:
        """Validate generation input parameters.
        
        Args:
            prompt: Input prompt
            character: Character name
            
        Raises:
            ValidationError: If input is invalid
            CharacterNotFoundError: If character is not found
        """
        # Validate prompt
        if not prompt or not isinstance(prompt, str):
            raise ValidationError(
                "prompt",
                prompt,
                "Prompt must be a non-empty string"
            )
        
        if len(prompt) > 2000:  # Reasonable limit
            raise ValidationError(
                "prompt",
                prompt,
                "Prompt too long (max 2000 characters)"
            )
        
        # Validate character
        if not character or not isinstance(character, str):
            raise ValidationError(
                "character",
                character,
                "Character must be a non-empty string"
            )
        
        if character not in self.characters:
            available = list(self.characters.keys())
            raise CharacterNotFoundError(character, available)
    
    async def get_available_characters(self) -> List[Dict[str, Any]]:
        """Get list of available characters.
        
        Returns:
            List of character dictionaries with metadata
        """
        characters = []
        for name, info in self.characters.items():
            characters.append({
                "name": name,
                "description": info.get("description", ""),
                "personality": info.get("personality", ""),
                "speaking_style": info.get("speaking_style", "")
            })
        
        return characters
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health status.
        
        Returns:
            Dictionary containing health status information
        """
        try:
            # Check if model is loaded
            model_loaded = self.llm is not None
            
            # Check model file
            model_file_exists = self.model_path.exists()
            
            # Test model loading if not loaded
            if not model_loaded:
                try:
                    await self._load_model()
                    model_loaded = self.llm is not None
                except Exception:
                    model_loaded = False
            
            status = "healthy" if model_loaded and model_file_exists else "unhealthy"
            
            return {
                "status": status,
                "model_loaded": model_loaded,
                "model_path": str(self.model_path),
                "model_file_exists": model_file_exists,
                "available_characters": list(self.characters.keys()),
                "character_count": len(self.characters)
            }
            
        except Exception as e:
            self.logger.error("health_check_failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_loaded": False,
                "model_path": str(self.model_path)
            }
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("cleaning_up_llm_service")
        self.llm = None


# Factory function for creating LLM service instances
def create_llm_service(config: Dict[str, Any]) -> LocalLLMService:
    """Create a new LLM service instance.
    
    Args:
        config: Service configuration
        
    Returns:
        Configured LLM service instance
    """
    return LocalLLMService(config)
