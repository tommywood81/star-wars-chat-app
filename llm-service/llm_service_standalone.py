"""
LLM Service for Star Wars Chat App with RAG (Retrieval-Augmented Generation).

This module provides FastAPI endpoints for character chat using PostgreSQL + pgvector
for retrieving relevant Star Wars dialogue context.
"""

import os
import logging
import time
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Star Wars Chat - LLM Service",
    description="LLM service for Star Wars character chat with RAG",
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

class RAGLLMService:
    """LLM service with RAG capabilities using PostgreSQL + pgvector."""
    
    def __init__(self):
        """Initialize the RAG LLM service."""
        self.characters = self._load_characters()
        self.model = None
        self.model_path = None
        self.embedding_model = None
        self.db_pool = None
        self._load_model()
        self._load_embedding_model()
        asyncio.create_task(self._initialize_database())
        logger.info("RAG LLM service initialized")
    
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
        """Load the Phi-2 model."""
        try:
            import llama_cpp
            
            model_path = Path("models/phi-2.Q4_K_M.gguf")
            
            if not model_path.exists():
                raise RuntimeError(f"Model not found at {model_path}")
            
            logger.info(f"Loading Phi-2 model from {model_path}")
            
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
    
    def _load_embedding_model(self):
        """Load the embedding model for generating embeddings."""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            logger.info("Loading embedding model")
            model_name = 'sentence-transformers/all-MiniLM-L6-v2'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.embedding_model = AutoModel.from_pretrained(model_name)
            logger.info("Embedding model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")
    
    async def _initialize_database(self):
        """Initialize database connection."""
        try:
            import asyncpg
            
            # Get database connection details from environment
            host = os.getenv("POSTGRES_HOST", "localhost")
            port = int(os.getenv("POSTGRES_PORT", "5432"))
            database = os.getenv("POSTGRES_DB", "star_wars_rag")
            user = os.getenv("POSTGRES_USER", "postgres")
            password = os.getenv("POSTGRES_PASSWORD", "password")
            
            connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
            
            self.db_pool = await asyncpg.create_pool(
                connection_string,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            
            logger.info("Database connection pool created")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise RuntimeError(f"Database initialization failed: {e}")
    
    async def _get_relevant_context(self, message: str, character: str, top_k: int = 6) -> List[Dict[str, Any]]:
        """Retrieve relevant context from the database using vector similarity."""
        if not self.db_pool or not self.embedding_model:
            return []
        
        try:
            # Generate embedding for the user message
            import torch
            
            # Tokenize and encode
            inputs = self.tokenizer(message, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                # Use mean pooling
                attention_mask = inputs['attention_mask']
                embeddings = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
                embeddings = embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                message_embedding = embeddings[0].numpy()
            
            # Query database for similar dialogue lines
            async with self.db_pool.acquire() as conn:
                # First get character ID
                character_id = await conn.fetchval(
                    "SELECT id FROM characters WHERE name = $1",
                    character
                )
                
                if not character_id:
                    logger.warning(f"Character {character} not found in database")
                    return []
                
                # Query for similar dialogue lines
                query = """
                SELECT dl.dialogue, dl.cleaned_dialogue, dl.scene_info, m.title as movie_title
                FROM dialogue_lines dl
                JOIN movies m ON dl.movie_id = m.id
                WHERE dl.character_id = $1
                ORDER BY dl.embedding <=> $2
                LIMIT $3
                """
                
                rows = await conn.fetch(query, character_id, message_embedding.tolist(), top_k)
                
                context_lines = []
                for row in rows:
                    context_lines.append({
                        "dialogue": row['dialogue'],
                        "cleaned_dialogue": row['cleaned_dialogue'],
                        "scene_info": row['scene_info'],
                        "movie_title": row['movie_title']
                    })
                
                logger.info(f"Retrieved {len(context_lines)} relevant context lines for {character}")
                return context_lines
                
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def _generate_response(self, message: str, character: str, context_lines: List[Dict[str, Any]] = None,
                          max_tokens: int = 200, temperature: float = 0.7) -> str:
        """Generate a response using the loaded model with RAG context."""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        try:
            # Build context from retrieved lines
            context_text = ""
            if context_lines:
                context_text = "Here are some relevant quotes from the Star Wars movies:\n\n"
                for i, line in enumerate(context_lines, 1):
                    context_text += f"{i}. \"{line['dialogue']}\" (from {line['movie_title']})\n"
                context_text += "\n"
            
            prompt = f"""You are {character}, a character from Star Wars.

Character Description: {self.characters.get(character, {}).get('description', '')}
Personality: {self.characters.get(character, {}).get('personality', '')}
Speaking Style: {self.characters.get(character, {}).get('speaking_style', '')}

{context_text}User: {message}

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
    
    async def chat_with_character(self, message: str, character: str, context: str = None,
                                 max_tokens: int = 200, temperature: float = 0.7) -> Dict[str, Any]:
        """Chat with a Star Wars character using RAG."""
        if character not in self.characters:
            raise RuntimeError(f"Unknown character: {character}")
        
        start_time = time.time()
        
        try:
            # Retrieve relevant context
            context_lines = await self._get_relevant_context(message, character)
            
            # Generate response with context
            response = self._generate_response(message, character, context_lines, max_tokens, temperature)
            
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
                    "tokens_generated": len(response.split()),
                    "context_lines_retrieved": len(context_lines),
                    "rag_enabled": True
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"Chat generation failed: {e}")

# Initialize service
try:
    llm_service = RAGLLMService()
except Exception as e:
    logger.error(f"Failed to initialize LLM service: {e}")
    llm_service = None

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not llm_service:
        raise HTTPException(status_code=503, detail="LLM service not initialized")
    
    db_status = "connected" if llm_service.db_pool else "disconnected"
    
    return {
        "status": "healthy",
        "service": "llm",
        "model": "phi-2",
        "model_path": llm_service.model_path,
        "database": db_status,
        "characters": list(llm_service.characters.keys())
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with a Star Wars character using RAG."""
    if not llm_service:
        raise HTTPException(status_code=503, detail="LLM service not available")
    
    try:
        result = await llm_service.chat_with_character(
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
        "service": "Star Wars Chat - LLM Service with RAG",
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
