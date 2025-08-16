"""
FastAPI backend for Star Wars RAG chat application.

This module provides REST API endpoints for the complete chat system including
character interactions, system management, and real-time features.
"""

import logging
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from star_wars_rag.chat import StarWarsChatApp
from star_wars_rag.models import ModelManager
import numpy as np

logger = logging.getLogger(__name__)

# Global application state
chat_app: Optional[StarWarsChatApp] = None
session_store: Dict[str, List[Dict[str, Any]]] = {}


# Pydantic models for request/response
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
    context_used: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: float


class StreamChatEvent(BaseModel):
    """Streaming chat event model."""
    token: str
    accumulated_response: str
    is_complete: bool
    character: str
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    models_loaded: bool
    database_connected: bool
    data_loaded: bool
    timestamp: float


class CharacterInfo(BaseModel):
    """Character information model."""
    name: str
    dialogue_count: int
    movies: List[str]
    sample_dialogue: List[str]


class SystemInfo(BaseModel):
    """System information model."""
    dialogue_lines: int
    characters_count: int
    movies_count: int
    embedding_model: str
    llm_model: str
    version: str


class DataLoadRequest(BaseModel):
    """Data loading request model."""
    script_path: Optional[str] = None
    force_reload: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Star Wars RAG API...")
    
    global chat_app
    try:
        # Initialize chat application
        chat_app = StarWarsChatApp(auto_download=True)
        
        # Load default data if available
        data_dir = Path("data/raw")
        if data_dir.exists():
            script_files = list(data_dir.glob("*.txt"))
            if script_files:
                # Load A New Hope by default
                default_script = None
                for script in script_files:
                    if "NEW HOPE" in script.name.upper():
                        default_script = script
                        break
                
                if default_script is None:
                    default_script = script_files[0]
                
                logger.info(f"Loading default data: {default_script.name}")
                
                # Create temp directory for loading
                import tempfile
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_script_dir = Path(temp_dir) / "scripts"
                    temp_script_dir.mkdir()
                    
                    temp_script = temp_script_dir / default_script.name
                    temp_script.write_text(
                        default_script.read_text(encoding='utf-8'), 
                        encoding='utf-8'
                    )
                    
                    chat_app.load_from_scripts(temp_script_dir, pattern=default_script.name)
                
                logger.info("Default data loaded successfully")
        
        logger.info("Star Wars RAG API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        chat_app = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down Star Wars RAG API...")
    if chat_app and hasattr(chat_app, 'cleanup'):
        await chat_app.cleanup()


# Create FastAPI application
app = FastAPI(
    title="Star Wars RAG Chat API",
    description="REST API for chatting with Star Wars characters using RAG and local LLM",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Utility functions
def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def get_chat_app() -> StarWarsChatApp:
    """Get the global chat application instance."""
    if chat_app is None:
        raise HTTPException(
            status_code=503,
            detail="Chat application not initialized"
        )
    return chat_app


def get_session_history(session_id: str) -> List[Dict[str, Any]]:
    """Get conversation history for a session."""
    return session_store.get(session_id, [])


def store_session_message(session_id: str, role: str, content: str, character: str = None):
    """Store a message in session history."""
    if session_id not in session_store:
        session_store[session_id] = []
    
    message = {
        "role": role,
        "content": content,
        "timestamp": time.time()
    }
    if character:
        message["character"] = character
    
    session_store[session_id].append(message)
    
    # Keep only last 20 messages per session
    if len(session_store[session_id]) > 20:
        session_store[session_id] = session_store[session_id][-20:]


# API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    models_loaded = chat_app is not None and chat_app.llm is not None
    data_loaded = chat_app is not None and chat_app.is_loaded
    
    return HealthResponse(
        status="healthy" if (models_loaded and data_loaded) else "degraded",
        models_loaded=models_loaded,
        database_connected=True,  # TODO: Add actual database check
        data_loaded=data_loaded,
        timestamp=time.time()
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_with_character(request: ChatRequest):
    """Chat with a Star Wars character."""
    app_instance = get_chat_app()
    
    # Validate character
    available_characters = app_instance.get_available_characters()
    if request.character not in available_characters:
        raise HTTPException(
            status_code=400,
            detail=f"Character '{request.character}' not available. Available: {available_characters}"
        )
    
    # Validate message
    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )
    
    try:
        # Get conversation history
        conversation_history = []
        session_history = get_session_history(request.session_id)
        
        # Convert session history to conversation format
        for msg in session_history[-6:]:  # Last 6 messages for context
            if msg["role"] == "user":
                conversation_history.append({
                    "user": msg["content"],
                    "character_name": request.character
                })
            elif msg["role"] == "character":
                if conversation_history:
                    conversation_history[-1]["character"] = msg["content"]
        
        # Generate response
        response = app_instance.chat_with_character(
            user_message=request.message,
            character=request.character,
            conversation_history=conversation_history,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Store messages in session
        store_session_message(request.session_id, "user", request.message)
        store_session_message(
            request.session_id, 
            "character", 
            response["response"], 
            request.character
        )
        
        # Convert numpy types to Python types for JSON serialization
        clean_response = convert_numpy_types(response)
        
        return ChatResponse(
            character=clean_response["character"],
            response=clean_response["response"],
            user_message=request.message,
            session_id=request.session_id,
            context_used=clean_response["context_used"],
            metadata=clean_response.get("conversation_metadata", {}),
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat generation failed: {str(e)}"
        )


@app.post("/chat/stream")
async def stream_chat_with_character(request: ChatRequest):
    """Stream chat response with a Star Wars character."""
    app_instance = get_chat_app()
    
    # Validate character
    available_characters = app_instance.get_available_characters()
    if request.character not in available_characters:
        raise HTTPException(
            status_code=400,
            detail=f"Character '{request.character}' not available"
        )
    
    # Validate message
    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )
    
    async def generate_stream():
        """Generate streaming response."""
        try:
            # Get conversation history
            conversation_history = []
            session_history = get_session_history(request.session_id)
            
            for msg in session_history[-6:]:
                if msg["role"] == "user":
                    conversation_history.append({
                        "user": msg["content"],
                        "character_name": request.character
                    })
                elif msg["role"] == "character":
                    if conversation_history:
                        conversation_history[-1]["character"] = msg["content"]
            
            # Store user message
            store_session_message(request.session_id, "user", request.message)
            
            # Generate streaming response
            full_response = ""
            
            for token_data in app_instance.stream_chat_with_character(
                user_message=request.message,
                character=request.character,
                conversation_history=conversation_history,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            ):
                full_response = token_data.get("accumulated_response", "")
                
                event = StreamChatEvent(
                    token=token_data["token"],
                    accumulated_response=full_response,
                    is_complete=token_data["is_complete"],
                    character=request.character,
                    metadata=token_data.get("metadata", {})
                )
                
                yield f"data: {event.json()}\n\n"
                
                if token_data["is_complete"]:
                    break
            
            # Store final response
            if full_response:
                store_session_message(
                    request.session_id, 
                    "character", 
                    full_response, 
                    request.character
                )
            
        except Exception as e:
            logger.error(f"Streaming chat error: {e}")
            error_event = StreamChatEvent(
                token="",
                accumulated_response=f"Error: {str(e)}",
                is_complete=True,
                character=request.character,
                metadata={"error": str(e)}
            )
            yield f"data: {error_event.json()}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/characters")
async def get_characters():
    """Get available characters."""
    app_instance = get_chat_app()
    
    characters = app_instance.get_available_characters()
    character_info = []
    
    for character in characters:
        try:
            # Get character statistics
            stats = app_instance.get_system_stats()
            character_count = stats.get("top_characters", {}).get(character, 0)
            
            # Get sample dialogue
            sample_dialogue = []
            try:
                sample_results = app_instance.retrieve_similar_dialogue(
                    "hello", 
                    character_filter=character, 
                    top_k=3
                )
                sample_dialogue = [result["dialogue"][:100] for result in sample_results[:2]]
            except:
                pass
            
            character_info.append(CharacterInfo(
                name=character,
                dialogue_count=character_count,
                movies=stats.get("movies", []),
                sample_dialogue=sample_dialogue
            ))
            
        except Exception as e:
            logger.warning(f"Error getting info for character {character}: {e}")
            character_info.append(CharacterInfo(
                name=character,
                dialogue_count=0,
                movies=[],
                sample_dialogue=[]
            ))
    
    return {"characters": character_info}


@app.get("/characters/{character_name}/stats")
async def get_character_stats(character_name: str):
    """Get statistics for a specific character."""
    app_instance = get_chat_app()
    
    available_characters = app_instance.get_available_characters()
    if character_name not in available_characters:
        raise HTTPException(
            status_code=404,
            detail=f"Character '{character_name}' not found"
        )
    
    try:
        # Get character dialogue
        dialogue_results = app_instance.retrieve_similar_dialogue(
            "hello",  # Generic query to get character dialogue
            character_filter=character_name,
            top_k=10
        )
        
        stats = app_instance.get_system_stats()
        character_count = stats.get("top_characters", {}).get(character_name, 0)
        
        sample_dialogue = [result["dialogue"] for result in dialogue_results[:5]]
        
        return {
            "character": character_name,
            "dialogue_count": character_count,
            "sample_dialogue": sample_dialogue,
            "movies": stats.get("movies", [])
        }
        
    except Exception as e:
        logger.error(f"Error getting character stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get character statistics: {str(e)}"
        )


@app.get("/system/info", response_model=SystemInfo)
async def get_system_info():
    """Get system information."""
    app_instance = get_chat_app()
    
    try:
        stats = app_instance.get_system_stats()
        llm_info = app_instance.get_llm_info()
        
        return SystemInfo(
            dialogue_lines=stats.get("total_dialogue_lines", 0),
            characters_count=stats.get("num_characters", 0),
            movies_count=stats.get("num_movies", 0),
            embedding_model=stats.get("embedding_model", "unknown"),
            llm_model=llm_info.get("model_name", "unknown"),
            version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system information: {str(e)}"
        )


@app.post("/admin/load-data")
async def load_data(request: DataLoadRequest, background_tasks: BackgroundTasks):
    """Load dialogue data from scripts (admin endpoint)."""
    app_instance = get_chat_app()
    
    try:
        if request.script_path:
            script_path = Path(request.script_path)
            if not script_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Script file not found: {script_path}"
                )
            
            # Load specific script
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_script_dir = Path(temp_dir) / "scripts"
                temp_script_dir.mkdir()
                
                temp_script = temp_script_dir / script_path.name
                temp_script.write_text(
                    script_path.read_text(encoding='utf-8'), 
                    encoding='utf-8'
                )
                
                app_instance.load_from_scripts(
                    temp_script_dir, 
                    pattern=script_path.name
                )
        else:
            # Load all available scripts
            data_dir = Path("data/raw")
            if data_dir.exists():
                app_instance.load_from_scripts(data_dir)
        
        stats = app_instance.get_system_stats()
        
        return {
            "status": "success",
            "message": "Data loaded successfully",
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Data loading error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load data: {str(e)}"
        )


@app.get("/admin/models")
async def list_models():
    """List available and downloaded models (admin endpoint)."""
    app_instance = get_chat_app()
    
    try:
        model_info = app_instance.list_available_models()
        return model_info
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list models: {str(e)}"
        )


@app.post("/admin/download-model/{model_name}")
async def download_model(model_name: str, background_tasks: BackgroundTasks):
    """Download a specific model (admin endpoint)."""
    app_instance = get_chat_app()
    
    def download_task():
        """Background task for model download."""
        try:
            success = app_instance.download_model(model_name, force=False)
            if success:
                logger.info(f"Model {model_name} downloaded successfully")
            else:
                logger.error(f"Failed to download model {model_name}")
        except Exception as e:
            logger.error(f"Model download error: {e}")
    
    background_tasks.add_task(download_task)
    
    return {
        "status": "started",
        "message": f"Model {model_name} download started in background"
    }


# Exception handlers
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return HTTPException(
        status_code=500,
        detail="Internal server error"
    )


# Application factory
def create_app() -> FastAPI:
    """Create FastAPI application."""
    return app


if __name__ == "__main__":
    # For development
    uvicorn.run(
        "src.star_wars_rag.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
