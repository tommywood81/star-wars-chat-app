"""
Standalone LLM Service for Star Wars Chat App.

This is a simplified LLM service that provides character responses
without requiring the heavy Phi-2 model for basic functionality.
"""

import os
import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Star Wars Chat - LLM Service",
    description="LLM service for Star Wars character responses",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    """Request model for chat."""
    message: str
    character: str = "Luke Skywalker"

class ChatResponse(BaseModel):
    """Response model for chat."""
    response: str
    character: str
    model: str = "character_responses"

class CharacterResponses:
    """Simple character response system."""
    
    def __init__(self):
        self.characters = {
            "Luke Skywalker": {
                "personality": "Optimistic, brave, and determined Jedi Knight",
                "responses": [
                    "The Force is strong with you, young one. Trust in it.",
                    "I believe there's good in everyone, even in the darkest places.",
                    "May the Force be with you, always.",
                    "I'm a Jedi, like my father before me.",
                    "The Force will guide us through this challenge."
                ]
            },
            "Darth Vader": {
                "personality": "Dark, intimidating, and powerful Sith Lord",
                "responses": [
                    "I find your lack of faith disturbing.",
                    "The Force is strong with this one.",
                    "You underestimate the power of the dark side.",
                    "I am your father.",
                    "The Emperor will not be pleased with this."
                ]
            },
            "Yoda": {
                "personality": "Wise, ancient Jedi Master with unique speech pattern",
                "responses": [
                    "Do or do not, there is no try.",
                    "Fear is the path to the dark side.",
                    "Size matters not. Look at me.",
                    "A Jedi uses the Force for knowledge and defense, never for attack.",
                    "Patience you must have, my young padawan."
                ]
            },
            "Han Solo": {
                "personality": "Confident, sarcastic smuggler and captain",
                "responses": [
                    "I've got a bad feeling about this.",
                    "Great, kid! Don't get cocky.",
                    "I know.",
                    "Hokey religions and ancient weapons are no match for a good blaster at your side.",
                    "Let's blow this thing and go home!"
                ]
            },
            "Princess Leia": {
                "personality": "Strong-willed, diplomatic leader and princess",
                "responses": [
                    "Help me, Obi-Wan Kenobi. You're my only hope.",
                    "I love you.",
                    "I know.",
                    "Aren't you a little short for a stormtrooper?",
                    "The more you tighten your grip, the more star systems will slip through your fingers."
                ]
            },
            "Obi-Wan Kenobi": {
                "personality": "Wise, experienced Jedi Master and mentor",
                "responses": [
                    "The Force will be with you, always.",
                    "These aren't the droids you're looking for.",
                    "Hello there!",
                    "You were the chosen one!",
                    "Use the Force, Luke."
                ]
            }
        }
    
    def get_response(self, message: str, character: str) -> str:
        """Get a character-appropriate response."""
        import random
        
        if character not in self.characters:
            character = "Luke Skywalker"  # Default fallback
        
        char_data = self.characters[character]
        responses = char_data["responses"]
        
        # Simple response selection based on message content
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["force", "jedi", "light"]):
            return random.choice([r for r in responses if "Force" in r or "Jedi" in r])
        elif any(word in message_lower for word in ["dark", "sith", "evil"]):
            return random.choice([r for r in responses if "dark" in r.lower() or "Sith" in r])
        elif any(word in message_lower for word in ["love", "feelings"]):
            return random.choice([r for r in responses if "love" in r.lower()])
        else:
            return random.choice(responses)

# Initialize character responses
character_responses = CharacterResponses()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "service": "llm", 
        "model": "character_responses",
        "characters": list(character_responses.characters.keys())
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Get a character response."""
    try:
        response = character_responses.get_response(request.message, request.character)
        
        return ChatResponse(
            response=response,
            character=request.character,
            model="character_responses"
        )
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/characters")
async def get_characters():
    """Get available characters."""
    return {
        "characters": list(character_responses.characters.keys()),
        "count": len(character_responses.characters)
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Star Wars Chat - LLM Service",
        "version": "1.0.0",
        "model": "character_responses",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "characters": "/characters"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5003)
