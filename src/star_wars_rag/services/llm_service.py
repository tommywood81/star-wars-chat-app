"""
Large Language Model service implementation using Transformers.

This module provides a concrete implementation of the LLMService interface
using Hugging Face Transformers for text generation.
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import torch

from ..core.interfaces import LLMService
from ..core.exceptions import ServiceError, ModelError, ValidationError, CharacterNotFoundError
from ..core.logging import LoggerMixin


class LocalLLMService(LLMService, LoggerMixin):
    """Large Language Model service using Transformers."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the Local LLM service.
        
        Args:
            config: Configuration dictionary containing:
                - model_name: Hugging Face model name
                - max_length: Maximum generation length
                - temperature: Sampling temperature
                - verbose: Enable verbose logging
                
        Raises:
            ConfigurationError: If configuration is invalid
        """
        super().__init__(config)
        
        # Validate configuration
        self._validate_config()
        
        # Initialize model
        self.model = None
        self.tokenizer = None
        self.model_name = self.config.get("model_name", "microsoft/DialoGPT-medium")
        self.max_length = self.config.get("max_length", 150)
        self.temperature = self.config.get("temperature", 0.7)
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
                        model_name=self.model_name,
                        max_length=self.max_length)
    
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
        if self.model is not None and self.tokenizer is not None:
            return
        
        try:
            self.logger.info("loading_llm_model", model_name=self.model_name)
            
            # Import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Load the model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.logger.info("llm_model_loaded", model_name=self.model_name)
            
        except Exception as e:
            self.logger.error("failed_to_load_llm_model", 
                            model_name=self.model_name, 
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
        
        # Tokenize input
        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate response using the model
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=inputs.input_ids.shape[1] + self.max_length,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the response
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return response.strip()
    
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
            # Validate inputs
            if not prompt or not prompt.strip():
                raise ValidationError("Prompt cannot be empty")
            
            if character not in self.characters:
                raise CharacterNotFoundError(f"Character '{character}' not found")
            
            # Load model if not loaded
            await self._load_model()
            
            # Generate response
            response = self._generate_response(prompt, character)
            
            duration = time.time() - start_time
            
            # Log performance metric
            from ..core.logging import log_performance_metric
            log_performance_metric(self.logger, "llm_generation", duration, {
                "character": character,
                "prompt_length": len(prompt),
                "response_length": len(response)
            })
            
            return {
                "response": response,
                "character": character,
                "duration": duration,
                "prompt_length": len(prompt),
                "model": "transformers-star-wars",
                "model_type": "transformers"
            }
            
        except (ValidationError, CharacterNotFoundError):
            raise
        except Exception as e:
            from ..core.logging import log_error_with_context
            log_error_with_context(self.logger, "llm_generation_failed", str(e), {
                "character": character,
                "prompt_length": len(prompt)
            })
            raise ModelError("LLM", "generation", f"Failed to generate response: {str(e)}")
    
    async def get_characters(self) -> List[str]:
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
            raise CharacterNotFoundError(f"Character '{character}' not found")
        
        return self.characters[character]
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check.
        
        Returns:
            Health status dictionary
        """
        try:
            # Try to load model if not loaded
            if self.model is None:
                await self._load_model()
            
            return {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "characters_available": len(self.characters),
                "model": "transformers-star-wars",
                "model_type": "transformers"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_loaded": self.model is not None,
                "characters_available": len(self.characters)
            }
