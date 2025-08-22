"""
FastAPI backend for Star Wars RAG chat application.

This module provides REST API endpoints for the complete chat system.
"""

import logging
import time
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Star Wars RAG API",
    description="API for Star Wars character chat system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    """Chat request model."""
    character: str = Field(..., description="Character to chat with")
    message: str = Field(..., min_length=1, description="User message")
    session_id: str = Field(..., description="Session identifier")
    max_tokens: Optional[int] = Field(150, ge=1, le=500, description="Maximum response tokens")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="LLM temperature")

class ChatResponse(BaseModel):
    """Chat response model."""
    character: str
    response: str
    user_message: str
    session_id: str
    metadata: Dict[str, Any]
    timestamp: float

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    service: str
    timestamp: float

class CharacterInfo(BaseModel):
    """Character information model."""
    name: str
    description: str
    personality: str

# Mock data for testing
MOCK_CHARACTERS = {
    "Luke Skywalker": {
        "description": "A young Jedi Knight with a strong connection to the Force",
        "personality": "Optimistic, brave, and determined to do what's right"
    },
    "Darth Vader": {
        "description": "A powerful Sith Lord and former Jedi",
        "personality": "Intimidating, authoritative, and conflicted"
    },
    "Princess Leia": {
        "description": "A brave leader and diplomat",
        "personality": "Strong-willed, intelligent, and compassionate"
    }
}

# Global session store
session_store: Dict[str, List[Dict[str, Any]]] = {}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="Star Wars RAG API",
        timestamp=time.time()
    )

@app.get("/characters")
async def get_characters():
    """Get available characters."""
    return {
        "characters": list(MOCK_CHARACTERS.keys()),
        "count": len(MOCK_CHARACTERS)
    }

@app.get("/characters/{character}")
async def get_character_info(character: str):
    """Get information about a specific character."""
    if character not in MOCK_CHARACTERS:
        raise HTTPException(status_code=404, detail=f"Character '{character}' not found")
    
    return {
        "name": character,
        **MOCK_CHARACTERS[character]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_with_character(request: ChatRequest):
    """Chat with a Star Wars character."""
    if request.character not in MOCK_CHARACTERS:
        raise HTTPException(status_code=400, detail=f"Character '{request.character}' not found")
    
    # Mock response based on character
    mock_responses = {
        "Luke Skywalker": "The Force is strong with you. I can feel it.",
        "Darth Vader": "I find your lack of faith... disturbing.",
        "Princess Leia": "Help me, Obi-Wan Kenobi. You're my only hope."
    }
    
    response = mock_responses.get(request.character, "I understand.")
    
    # Store in session
    if request.session_id not in session_store:
        session_store[request.session_id] = []
    
    session_store[request.session_id].append({
        "character": request.character,
        "user_message": request.message,
        "response": response,
        "timestamp": time.time()
    })
    
    return ChatResponse(
        character=request.character,
        response=response,
        user_message=request.message,
        session_id=request.session_id,
        metadata={
            "model": "mock-api",
            "duration": 0.1,
            "tokens_used": len(response.split())
        },
        timestamp=time.time()
    )

@app.get("/sessions/{session_id}")
async def get_session_history(session_id: str):
    """Get chat history for a session."""
    if session_id not in session_store:
        return {"messages": [], "count": 0}
    
    return {
        "messages": session_store[session_id],
        "count": len(session_store[session_id])
    }

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a chat session."""
    if session_id in session_store:
        del session_store[session_id]
    
    return {"message": "Session cleared", "session_id": session_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
