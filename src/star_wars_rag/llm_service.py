"""
LLM Service for Star Wars Chat App.

This module provides FastAPI endpoints for character chat using the local LLM.
It integrates with the existing LLM module and provides character-specific responses.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .llm import LocalLLM
from .prompt import StarWarsPromptBuilder

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
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40
    repeat_penalty: Optional[float] = 1.1

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
    voice_style: Optional[str] = None

class LLMService:
    """LLM service for Star Wars character chat."""
    
    def __init__(self, model_path: str = "models/phi-2.Q4_K_M.gguf"):
        """Initialize the LLM service.
        
        Args:
            model_path: Path to the GGUF model file
        """
        self.model_path = Path(model_path)
        self.llm = None
        self.prompt_builder = StarWarsPromptBuilder()
        self._load_model()
    
    def _load_model(self):
        """Load the LLM model."""
        try:
            logger.info(f"Loading LLM model: {self.model_path}")
            self.llm = LocalLLM(
                model_path=self.model_path,
                n_ctx=2048,
                n_threads=None,  # Auto-detect
                n_gpu_layers=0,  # CPU-only for now
                verbose=False
            )
            logger.info("LLM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            raise RuntimeError(f"Could not load LLM model: {e}")
    
    def chat_with_character(self, message: str, character: str, context: str = None,
                           max_tokens: int = 200, temperature: float = 0.7,
                           top_p: float = 0.9, top_k: int = 40,
                           repeat_penalty: float = 1.1) -> Dict[str, Any]:
        """Chat with a Star Wars character.
        
        Args:
            message: User's message
            character: Character name to chat with
            context: Optional conversation context
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling limit
            repeat_penalty: Repetition penalty factor
            
        Returns:
            Dictionary containing character response and metadata
        """
        try:
            logger.info(f"Chatting with {character}: {message[:50]}...")
            
            # Build character-specific prompt
            prompt = self.prompt_builder.build_character_prompt(
                character=character,
                user_message=message,
                context=context
            )
            
            # Generate response
            result = self.llm.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stop=["\n", "User:", "Human:", "Assistant:"]
            )
            
            # Extract and clean response
            response_text = result['response'].strip()
            
            # Remove any remaining prompt artifacts
            if "Assistant:" in response_text:
                response_text = response_text.split("Assistant:")[-1].strip()
            
            logger.info(f"Generated response for {character}: {response_text[:50]}...")
            
            return {
                "response": response_text,
                "character": character,
                "metadata": {
                    **result['metadata'],
                    "prompt_length": len(prompt),
                    "response_length": len(response_text)
                }
            }
            
        except Exception as e:
            logger.error(f"Character chat failed: {e}")
            raise RuntimeError(f"Character chat failed: {e}")
    
    def get_available_characters(self) -> List[CharacterInfo]:
        """Get list of available characters."""
        return [
            CharacterInfo(
                name="Luke Skywalker",
                description="Jedi Knight and hero of the Rebellion",
                personality="Brave, idealistic, and determined. Speaks with hope and conviction.",
                voice_style="young_male"
            ),
            CharacterInfo(
                name="Darth Vader",
                description="Dark Lord of the Sith and former Jedi",
                personality="Intimidating, powerful, and conflicted. Speaks with authority and menace.",
                voice_style="deep_male"
            ),
            CharacterInfo(
                name="Princess Leia",
                description="Princess of Alderaan and Rebel leader",
                personality="Strong-willed, intelligent, and diplomatic. Speaks with confidence and grace.",
                voice_style="female"
            ),
            CharacterInfo(
                name="Han Solo",
                description="Smuggler and captain of the Millennium Falcon",
                personality="Confident, sarcastic, and loyal. Speaks with wit and bravado.",
                voice_style="male"
            ),
            CharacterInfo(
                name="Yoda",
                description="Wise Jedi Master and teacher",
                personality="Ancient, wise, and philosophical. Speaks in unique sentence structure.",
                voice_style="elderly_male"
            ),
            CharacterInfo(
                name="Obi-Wan Kenobi",
                description="Jedi Master and mentor",
                personality="Wise, patient, and honorable. Speaks with wisdom and calm authority.",
                voice_style="mature_male"
            ),
            CharacterInfo(
                name="Chewbacca",
                description="Wookiee warrior and co-pilot",
                personality="Loyal, fierce, and protective. Communicates through growls and gestures.",
                voice_style="wookiee"
            ),
            CharacterInfo(
                name="R2-D2",
                description="Astromech droid and loyal companion",
                personality="Brave, resourceful, and expressive. Communicates through beeps and whistles.",
                voice_style="droid"
            )
        ]
    
    def get_character_info(self, character_name: str) -> CharacterInfo:
        """Get information about a specific character."""
        characters = self.get_available_characters()
        for character in characters:
            if character.name.lower() == character_name.lower():
                return character
        raise ValueError(f"Character '{character_name}' not found")

# Initialize LLM service
llm_service = LLMService()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "service": "llm", 
        "model": llm_service.model_path.name
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_with_character(request: ChatRequest):
    """Chat with a Star Wars character.
    
    Args:
        request: ChatRequest containing message and parameters
        
    Returns:
        ChatResponse with character response and metadata
    """
    try:
        result = llm_service.chat_with_character(
            message=request.message,
            character=request.character,
            context=request.context,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repeat_penalty=request.repeat_penalty
        )
        
        return ChatResponse(
            response=result["response"],
            character=result["character"],
            metadata=result["metadata"]
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/characters")
async def list_characters():
    """List available characters."""
    characters = llm_service.get_available_characters()
    return {
        "characters": [char.dict() for char in characters],
        "total_count": len(characters)
    }

@app.get("/characters/{character_name}")
async def get_character_info(character_name: str):
    """Get information about a specific character."""
    try:
        character = llm_service.get_character_info(character_name)
        return character.dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model."""
    return {
        "model_path": str(llm_service.model_path),
        "model_name": llm_service.model_path.name,
        "context_size": llm_service.llm.n_ctx if llm_service.llm else None,
        "threads": llm_service.llm.n_threads if llm_service.llm else None,
        "gpu_layers": llm_service.llm.n_gpu_layers if llm_service.llm else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5003)
