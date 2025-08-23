"""
LLM Service for Star Wars Chat App.

This module provides FastAPI endpoints for character chat.
Requires proper model setup - no fallbacks, fails fast on errors.
"""

import os
import logging
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Star Wars Chat - LLM Service",
    description="LLM service for Star Wars character chat",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    """Request model for character chat."""
    message: str
    character: str
    context: Optional[str] = None
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7

class ChatResponse(BaseModel):
    """Response model for character chat."""
    response: str
    character: str
    metadata: Dict[str, Any]

class CharacterInfo(BaseModel):
    """Model for character information."""
    name: str
    description: str
    personality: str
    speaking_style: str

class LLMService:
    """LLM service that requires proper model setup."""
    
    def __init__(self):
        """Initialize the LLM service."""
        self.characters = self._load_characters()
        self.model = None
        self.model_path = None
        self._load_model()
        logger.info("LLM service initialized")
    
    def _load_characters(self) -> Dict[str, Dict[str, str]]:
        """Load character definitions."""
        characters_file = Path("characters.json")
        
        if not characters_file.exists():
            raise RuntimeError(f"Characters file not found: {characters_file}")
        
        try:
            with open(characters_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load characters: {e}")
    
    def _load_model(self):
        """Load the Phi-2 model - no fallbacks."""
        try:
            # Try to import llama_cpp
            import llama_cpp
            
            # Look for the model in the models directory
            model_path = Path("models/phi-2.Q4_K_M.gguf")
            
            if not model_path.exists():
                raise RuntimeError(f"Model not found at {model_path}")
            
            logger.info(f"Loading Phi-2 model from {model_path}")
            
            # Load the model
            self.model = llama_cpp.Llama(
                model_path=str(model_path),
                n_ctx=1024,
                n_threads=2,
                n_batch=256,
                n_gpu_layers=0,
                verbose=False
            )
            
            self.model_path = str(model_path)
            logger.info("Phi-2 model loaded successfully")
            
        except ImportError as e:
            raise RuntimeError(f"Required library not available: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _generate_response(self, message: str, character: str, context: str = None,
                          max_tokens: int = 200, temperature: float = 0.7) -> str:
        """Generate a response using the loaded model."""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        try:
            prompt = f"""You are {character}, a character from Star Wars.

Character Description: {self.characters.get(character, {}).get('description', '')}
Personality: {self.characters.get(character, {}).get('personality', '')}
Speaking Style: {self.characters.get(character, {}).get('speaking_style', '')}

{context + chr(10) if context else ''}User: {message}

{character}:"""
            
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["User:", "\n\n", "###"],
                echo=False
            )
            
            generated_text = response['choices'][0]['text'].strip()
            if generated_text.startswith(f"{character}:"):
                generated_text = generated_text[len(f"{character}:"):].strip()
            
            if not generated_text:
                raise RuntimeError("Model generated empty response")
            
            return generated_text
            
        except Exception as e:
            raise RuntimeError(f"Model generation failed: {e}")
    
    def chat_with_character(self, message: str, character: str, context: str = None,
                           max_tokens: int = 200, temperature: float = 0.7) -> Dict[str, Any]:
        """Chat with a Star Wars character.
        
        Args:
            message: User's message
            character: Character to chat with
            context: Optional conversation context
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            
        Returns:
            Dictionary with response and metadata
            
        Raises:
            RuntimeError: If model or character setup fails
        """
        if character not in self.characters:
            raise RuntimeError(f"Unknown character: {character}")
        
        start_time = time.time()
        
        try:
            response = self._generate_response(message, character, context, max_tokens, temperature)
            
            processing_time = time.time() - start_time
            
            return {
                "response": response,
                "character": character,
                "metadata": {
                    "model": "phi-2",
                    "model_path": self.model_path,
                    "processing_time": processing_time,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "tokens_generated": len(response.split())
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"Chat generation failed: {e}")

# Initialize service
try:
    llm_service = LLMService()
except Exception as e:
    logger.error(f"Failed to initialize LLM service: {e}")
    llm_service = None

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not llm_service:
        raise HTTPException(status_code=503, detail="LLM service not initialized")
    
    return {
        "status": "healthy",
        "service": "llm",
        "model": "phi-2",
        "model_path": llm_service.model_path,
        "characters": list(llm_service.characters.keys())
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with a Star Wars character."""
    if not llm_service:
        raise HTTPException(status_code=503, detail="LLM service not available")
    
    try:
        result = llm_service.chat_with_character(
            message=request.message,
            character=request.character,
            context=request.context,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return ChatResponse(
            response=result["response"],
            character=result["character"],
            metadata=result["metadata"]
        )
        
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/characters")
async def get_characters():
    """Get available characters."""
    if not llm_service:
        raise HTTPException(status_code=503, detail="LLM service not available")
    
    return {
        "characters": list(llm_service.characters.keys()),
        "count": len(llm_service.characters)
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Star Wars Chat - LLM Service",
        "version": "1.0.0",
        "model": "phi-2",
        "status": "running" if llm_service else "failed",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "characters": "/characters"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5003)
