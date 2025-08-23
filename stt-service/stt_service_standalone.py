"""
Speech-to-Text (STT) Service for Star Wars Chat App.

This module provides FastAPI endpoints for converting audio input to text
using OpenAI's Whisper model. It's designed to run as a separate microservice.
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import whisper
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import structlog

# Configure logging
logger = structlog.get_logger()

app = FastAPI(
    title="Star Wars Chat - STT Service",
    description="Speech-to-Text service using Whisper",
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

class TranscriptionRequest(BaseModel):
    """Request model for transcription."""
    language: Optional[str] = "en"
    task: Optional[str] = "transcribe"

class TranscriptionResponse(BaseModel):
    """Response model for transcription."""
    text: str
    language: str
    confidence: Optional[float] = None
    duration: Optional[float] = None

class STTService:
    """Speech-to-Text service using Whisper."""
    
    def __init__(self, model_name: str = None):
        """Initialize the STT service with Whisper model.
        
        Args:
            model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        # Use environment variable if not specified, default to 'base'
        self.model_name = model_name or os.getenv("WHISPER_MODEL", "base")
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model."""
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise RuntimeError(f"Could not load Whisper model: {e}")
    
    def transcribe_audio(self, audio_path: str, language: str = "en", task: str = "transcribe") -> Dict[str, Any]:
        """Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'es', 'fr')
            task: Task type ('transcribe' or 'translate')
            
        Returns:
            Dictionary containing transcription results
        """
        try:
            logger.info(f"Transcribing audio: {audio_path}")
            
            # Validate file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Transcribe with Whisper
            result = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                fp16=False  # Use CPU
            )
            
            logger.info("Transcription completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {e}")

# Initialize STT service
stt_service = STTService()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "stt", "model": stt_service.model_name}

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = "en",
    task: str = "transcribe"
):
    """Transcribe uploaded audio file to text.
    
    Args:
        file: Audio file upload
        language: Language code for transcription
        task: Task type ('transcribe' or 'translate')
        
    Returns:
        TranscriptionResponse with text and metadata
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Please upload an audio file."
            )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            # Write uploaded audio to temp file
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Transcribe audio
            result = stt_service.transcribe_audio(temp_file_path, language, task)
            
            # Extract text and metadata
            text = result.get("text", "").strip()
            detected_language = result.get("language", language)
            confidence = result.get("avg_logprob", None)
            duration = result.get("duration", None)
            
            if not text:
                raise HTTPException(
                    status_code=400,
                    detail="No speech detected in audio file"
                )
            
            return TranscriptionResponse(
                text=text,
                language=detected_language,
                confidence=confidence,
                duration=duration
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/models")
async def list_models():
    """List available Whisper models."""
    models = ["tiny", "base", "small", "medium", "large"]
    return {
        "available_models": models,
        "current_model": stt_service.model_name
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "STT Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "transcribe": "/transcribe",
            "models": "/models"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
