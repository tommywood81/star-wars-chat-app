"""
LLM Service for Star Wars Chat App.

This module provides FastAPI endpoints for character chat using a hybrid approach:
- Tries to load Phi-2 model if available
- Falls back to intelligent character-based responses if model fails
"""

import os
import logging
import time
import json
import random
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Star Wars Chat - LLM Service",
    description="LLM service for Star Wars character chat using hybrid approach",
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

class HybridLLMService:
    """Hybrid LLM service with Phi-2 fallback to intelligent character responses."""
    
    def __init__(self):
        """Initialize the hybrid LLM service."""
        self.characters = self._load_characters()
        self.character_responses = self._load_character_responses()
        self.model = None
        self.model_path = None
        self._load_model()
        logger.info("Hybrid LLM service initialized")
    
    def _load_characters(self) -> Dict[str, Dict[str, str]]:
        """Load character definitions."""
        characters_file = Path("characters.json")
        
        if characters_file.exists():
            try:
                with open(characters_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load characters: {e}")
        
        # Default characters
        return {
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
            },
            "Han Solo": {
                "description": "Smuggler and captain of the Millennium Falcon",
                "personality": "Confident, sarcastic, and loyal",
                "speaking_style": "Witty and bravado-filled"
            },
            "Yoda": {
                "description": "Wise Jedi Master and teacher",
                "personality": "Ancient, wise, and philosophical",
                "speaking_style": "Unique sentence structure, wise and cryptic"
            }
        }
    
    def _load_character_responses(self) -> Dict[str, List[str]]:
        """Load character-specific response patterns."""
        return {
            "Luke Skywalker": [
                "The Force is strong with you. I can feel it.",
                "I believe there's good in everyone, even in the darkest places.",
                "I'm not afraid. I'm ready to face whatever comes.",
                "The Force will guide us. We just need to trust in it.",
                "Sometimes the hardest choices are the ones that define us.",
                "I am a Jedi, like my father before me.",
                "The Force flows through all living things.",
                "I'll never turn to the dark side.",
                "I have a bad feeling about this.",
                "The Force is what gives a Jedi his power."
            ],
            "Darth Vader": [
                "I find your lack of faith... disturbing.",
                "The Force is strong with this one.",
                "You underestimate the power of the dark side.",
                "I am your father. Search your feelings, you know it to be true.",
                "The Emperor will show you the true nature of the Force.",
                "Luke, you can destroy the Emperor. He has foreseen this.",
                "The dark side of the Force is a pathway to many abilities some consider to be unnatural.",
                "You don't know the power of the dark side.",
                "It is your destiny.",
                "The circle is now complete."
            ],
            "Princess Leia": [
                "Help me, Obi-Wan Kenobi. You're my only hope.",
                "I'd just as soon kiss a Wookiee!",
                "Someone has to save our skins.",
                "I love you. I know.",
                "The more you tighten your grip, the more star systems will slip through your fingers.",
                "Aren't you a little short for a stormtrooper?",
                "I am a member of the Imperial Senate on a diplomatic mission.",
                "Why, you stuck-up, half-witted, scruffy-looking nerf herder!",
                "I'm not afraid to die.",
                "You have your moments."
            ],
            "Han Solo": [
                "I've got a bad feeling about this.",
                "Great, kid! Don't get cocky.",
                "Hokey religions and ancient weapons are no match for a good blaster at your side.",
                "Never tell me the odds!",
                "I know.",
                "Chewie, we're home.",
                "I can arrange that. He could use a good kiss!",
                "I'm a simple man trying to make my way in the universe.",
                "I'm responsible for these people.",
                "I love you too."
            ],
            "Yoda": [
                "Do or do not. There is no try.",
                "Fear is the path to the dark side.",
                "Size matters not. Look at me. Judge me by my size, do you?",
                "The greatest teacher, failure is.",
                "When you look at the dark side, careful you must be.",
                "The Force surrounds us, penetrates us, binds the galaxy together.",
                "Wars not make one great.",
                "Much to learn, you still have.",
                "Patience you must have, my young padawan.",
                "In a dark place we find ourselves, and a little more knowledge lights our way."
            ]
        }
    
    def _load_model(self):
        """Try to load the Phi-2 model, fallback gracefully if it fails."""
        try:
            # Try to import llama_cpp
            import llama_cpp
            
            # Look for the model in the models directory
            model_path = Path("models/phi-2.Q4_K_M.gguf")
            
            if not model_path.exists():
                logger.warning(f"Model not found at {model_path}")
                return
            
            logger.info(f"Attempting to load Phi-2 model from {model_path}")
            
            # Try to load the model
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
            
        except Exception as e:
            logger.warning(f"Phi-2 model loading failed: {str(e)}")
            logger.info("Falling back to intelligent character responses")
            self.model = None
    
    def _generate_intelligent_response(self, message: str, character: str, context: str = None) -> str:
        """Generate an intelligent response based on character personality and message content."""
        message_lower = message.lower()
        character_info = self.characters.get(character, {})
        responses = self.character_responses.get(character, ["I understand."])
        
        # Character-specific keyword matching
        if "force" in message_lower:
            if character == "Luke Skywalker":
                return "The Force flows through all living things. It's what gives a Jedi his power."
            elif character == "Darth Vader":
                return "The dark side of the Force is a pathway to many abilities some consider to be unnatural."
            elif character == "Yoda":
                return "The Force surrounds us, penetrates us, binds the galaxy together."
        
        if "father" in message_lower or "dad" in message_lower:
            if character == "Luke Skywalker":
                return "I am a Jedi, like my father before me."
            elif character == "Darth Vader":
                return "Luke, I am your father. Search your feelings, you know it to be true."
        
        if "hope" in message_lower:
            if character == "Princess Leia":
                return "Help me, Obi-Wan Kenobi. You're my only hope."
        
        if "odds" in message_lower or "chance" in message_lower:
            if character == "Han Solo":
                return "Never tell me the odds!"
        
        if "fear" in message_lower or "afraid" in message_lower:
            if character == "Yoda":
                return "Fear is the path to the dark side."
            elif character == "Luke Skywalker":
                return "I'm not afraid. I'm ready to face whatever comes."
        
        if "love" in message_lower:
            if character == "Princess Leia":
                return "I love you. I know."
            elif character == "Han Solo":
                return "I know."
        
        if "bad feeling" in message_lower:
            if character == "Han Solo":
                return "I've got a bad feeling about this."
            elif character == "Luke Skywalker":
                return "I have a bad feeling about this."
        
        # Personality-based responses
        personality = character_info.get("personality", "").lower()
        if "optimistic" in personality and any(word in message_lower for word in ["hope", "future", "better"]):
            return "I believe there's good in everyone, even in the darkest places."
        
        if "intimidating" in personality and any(word in message_lower for word in ["power", "strength", "control"]):
            return "You underestimate the power of the dark side."
        
        if "sarcastic" in personality and any(word in message_lower for word in ["religion", "faith", "belief"]):
            return "Hokey religions and ancient weapons are no match for a good blaster at your side."
        
        # Default to random character response
        return random.choice(responses)
    
    def _generate_response(self, message: str, character: str, context: str = None,
                          max_tokens: int = 200, temperature: float = 0.7) -> str:
        """Generate a response using available methods."""
        if self.model:
            try:
                # Try to use Phi-2 model
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
                
                return generated_text if generated_text else self._generate_intelligent_response(message, character, context)
                
            except Exception as e:
                logger.warning(f"Phi-2 generation failed: {e}")
                return self._generate_intelligent_response(message, character, context)
        else:
            return self._generate_intelligent_response(message, character, context)
    
    def chat_with_character(self, message: str, character: str, context: str = None,
                           max_tokens: int = 200, temperature: float = 0.7) -> Dict[str, Any]:
        """Chat with a Star Wars character using hybrid approach.
        
        Args:
            message: User's message
            character: Character name to chat with
            context: Optional conversation context
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dictionary containing character response and metadata
        """
        start_time = time.time()
        
        if character not in self.characters:
            raise ValueError(f"Character '{character}' not found")
        
        # Generate response
        response = self._generate_response(message, character, context, max_tokens, temperature)
        
        duration = time.time() - start_time
        
        return {
            "response": response,
            "character": character,
            "metadata": {
                "model": "phi-2-gguf" if self.model else "intelligent-character",
                "model_type": "llama_cpp" if self.model else "hybrid",
                "model_path": self.model_path,
                "duration": duration,
                "prompt_length": len(message),
                "response_length": len(response),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "method": "phi2" if self.model else "intelligent_fallback"
            }
        }

# Initialize the service
llm_service = HybridLLMService()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        return {
            "status": "healthy",
            "service": "LLM",
            "model_loaded": llm_service.model is not None,
            "characters_available": len(llm_service.characters),
            "model": "phi-2-gguf" if llm_service.model else "intelligent-character",
            "model_type": "llama_cpp" if llm_service.model else "hybrid",
            "model_path": llm_service.model_path
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "service": "LLM"
        }

@app.get("/characters")
async def get_characters():
    """Get available characters."""
    return {
        "characters": list(llm_service.characters.keys()),
        "count": len(llm_service.characters)
    }

@app.get("/characters/{character}")
async def get_character_info(character: str):
    """Get information about a specific character."""
    if character not in llm_service.characters:
        raise HTTPException(status_code=404, detail=f"Character '{character}' not found")
    
    return {
        "name": character,
        **llm_service.characters[character]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_with_character(request: ChatRequest):
    """Chat with a Star Wars character."""
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
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5003)
