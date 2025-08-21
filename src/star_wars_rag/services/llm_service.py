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
        self.model_path = Path(self.config.get("model_path", "models/phi-2.Q4_K_M.gguf"))
        self.n_ctx = self.config.get("n_ctx", 2048)
        self.n_threads = self.config.get("n_threads")
        self.n_gpu_layers = self.config.get("n_gpu_layers", 0)
        self.verbose = self.config.get("verbose", False)
        
        # Character management
        self.characters = {}
        self.characters_file = Path("characters.json")
        self._load_characters()
        
        # If no characters loaded, create defaults
        if not self.characters:
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
        
        self.logger.info("initializing_llm_service", 
                        model_path=str(self.model_path),
                        n_ctx=self.n_ctx)
    
    def _validate_config(self) -> None:
        """Validate service configuration.
        
        Raises:
            ValidationError: If configuration is invalid
        """
        # No required fields for now - we'll use mock responses
        pass
    
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
        """Load the LLM model asynchronously."""
        if self.llm is not None:
            return
        
        try:
            self.logger.info("loading_llm_model", model_path=str(self.model_path))
            
            # Import llama-cpp-python
            from llama_cpp import Llama
            
            # Load the model
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose
            )
            
            self.logger.info("llm_model_loaded", model_path=str(self.model_path))
            
        except Exception as e:
            self.logger.error("failed_to_load_llm_model", 
                            model_path=str(self.model_path), 
                            error=str(e))
            raise ServiceError("LLM", "model_loading", f"Failed to load LLM model: {str(e)}")
    
    def _generate_response(self, prompt: str, character: str) -> str:
        """Generate a response using the actual LLM model."""
        character_info = self.characters.get(character, {})
        personality = character_info.get("personality", "Neutral")
        speaking_style = character_info.get("speaking_style", "Direct")
        
        # Create a context-aware prompt
        system_prompt = f"""You are {character}, a character from Star Wars. 
        Personality: {personality}
        Speaking style: {speaking_style}
        
        Respond to the user's message in character, staying true to your personality and speaking style.
        Keep responses concise but engaging, as if you're having a real conversation."""
        
        full_prompt = f"{system_prompt}\n\nUser: {prompt}\n{character}:"
        
        # Generate response using the model
        response = self.llm(
            full_prompt,
            max_tokens=150,
            temperature=0.7,
            stop=["User:", "\n\n"]
        )
        
        return response['choices'][0]['text'].strip()
    
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
            
            self.logger.info("generating_response", 
                           prompt_length=len(prompt),
                           character=character)
            
            # Generate response using the model
            response_text = self._generate_response(prompt, character)
            
            duration = time.time() - start_time
            
            # Log performance metric
            from ..core.logging import log_performance_metric
            log_performance_metric(
                self.logger,
                "response_generation_duration",
                duration,
                "seconds",
                service="LLM",
                details={"prompt_length": len(prompt), "character": character}
            )
            
            self.logger.info("response_generated", 
                           character=character,
                           duration=duration,
                           response_length=len(response_text))
            
            return {
                "response": response_text,
                "character": character,
                "duration": duration,
                "prompt_length": len(prompt),
                "model": "phi-2"
            }
            
        except Exception as e:
            duration = time.time() - start_time
            from ..core.logging import log_error_with_context
            log_error_with_context(
                self.logger,
                e,
                context={
                    "prompt": prompt,
                    "character": character,
                    "duration": duration
                }
            )
            raise ModelError("LLM", "generation", f"Failed to generate response: {str(e)}")
    
    def _validate_generation_input(self, prompt: str, character: str) -> None:
        """Validate input for response generation.
        
        Args:
            prompt: Input prompt
            character: Character name
            
        Raises:
            ValidationError: If input is invalid
            CharacterNotFoundError: If character is not found
        """
        if not prompt or not prompt.strip():
            raise ValidationError("prompt", prompt, "Prompt cannot be empty")
        
        if not character or not character.strip():
            raise ValidationError("character", character, "Character cannot be empty")
        
        if character not in self.characters:
            raise CharacterNotFoundError(character, f"Character '{character}' not found")
    
    async def get_available_characters(self) -> List[str]:
        """Get list of available characters.
        
        Returns:
            List of character names
        """
        return list(self.characters.keys())
    
    async def get_character_info(self, character: str) -> Dict[str, Any]:
        """Get information about a specific character.
        
        Args:
            character: Character name
            
        Returns:
            Character information dictionary
            
        Raises:
            CharacterNotFoundError: If character is not found
        """
        if character not in self.characters:
            raise CharacterNotFoundError(character, f"Character '{character}' not found")
        
        return self.characters[character]
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the LLM service.
        
        Returns:
            Health status dictionary
        """
        try:
            await self._load_model()
            
            return {
                "status": "healthy",
                "service": "LLM",
                "model_loaded": self.llm is not None,
                "available_characters": len(self.characters),
                "model_type": "phi-2"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "service": "LLM",
                "error": str(e),
                "model_loaded": False
            }
