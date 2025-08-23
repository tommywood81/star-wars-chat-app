"""
Speech-to-Text Service using OpenAI Whisper.

This service requires proper Whisper setup - no fallbacks, fails fast on errors.
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import structlog

# Configure logging
logger = structlog.get_logger()

app = FastAPI(title="STT Service", version="1.0.0")

class TranscriptionRequest(BaseModel):
    """Request model for speech-to-text conversion."""
    language: Optional[str] = "en"
    task: Optional[str] = "transcribe"

class TranscriptionResponse(BaseModel):
    """Response model for speech-to-text conversion."""
    text: str
    language: str
    duration: Optional[float] = None

class STTService:
    """Speech-to-Text service using OpenAI Whisper."""
    
    def __init__(self, model_name: str = "base"):
        """Initialize the STT service.
        
        Args:
            model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            
        Raises:
            RuntimeError: If Whisper cannot be loaded or initialized
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model - no fallbacks."""
        try:
            import whisper
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded successfully")
        except ImportError as e:
            raise RuntimeError(f"Whisper library not available: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")
    
    def transcribe_audio(self, audio_path: str, language: str = "en", task: str = "transcribe") -> Dict[str, Any]:
        """Transcribe audio using Whisper.
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'es', 'fr')
            task: Task type ('transcribe' or 'translate')
            
        Returns:
            Dictionary containing transcription results
            
        Raises:
            RuntimeError: If transcription fails
        """
        if not self.model:
            raise RuntimeError("Whisper model not loaded")
        
        try:
            logger.info(f"Transcribing audio: {audio_path}")
            
            # Transcribe the audio file
            result = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                fp16=False  # Use CPU for compatibility
            )
            
            return {
                "text": result["text"].strip(),
                "language": result.get("language", language),
                "duration": result.get("duration", None)
            }
            
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")

# Initialize STT service
try:
    stt_service = STTService(model_name=os.getenv("WHISPER_MODEL", "base"))
except Exception as e:
    logger.error(f"Failed to initialize STT service: {e}")
    stt_service = None

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not stt_service:
        raise HTTPException(status_code=503, detail="STT service not initialized")
    
    return {
        "status": "healthy",
        "service": "stt",
        "model": stt_service.model_name
    }

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = "en",
    task: str = "transcribe"
):
    """Transcribe uploaded audio file.
    
    Args:
        audio: Audio file to transcribe
        language: Language code (e.g., 'en', 'es', 'fr')
        task: Task type ('transcribe' or 'translate')
        
    Returns:
        Transcription result
        
    Raises:
        HTTPException: If transcription fails
    """
    if not stt_service:
        raise HTTPException(status_code=503, detail="STT service not available")
    
    try:
        # Validate file type
        if not audio.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg')):
            raise HTTPException(status_code=400, detail="Unsupported audio format")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Transcribe the audio
            result = stt_service.transcribe_audio(temp_file_path, language, task)
            
            return TranscriptionResponse(
                text=result["text"],
                language=result["language"],
                duration=result.get("duration")
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "STT Service",
        "version": "1.0.0",
        "model": stt_service.model_name if stt_service else "not loaded",
        "status": "running" if stt_service else "failed",
        "endpoints": {
            "health": "/health",
            "transcribe": "/transcribe"
        }
    }
