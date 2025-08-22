"""
Standalone Speech-to-Text (STT) Service for Star Wars Chat App.

This is a completely isolated STT service that doesn't import any other services
to avoid loading heavy models like Phi-2.
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Star Wars Chat - STT Service",
    description="Speech-to-Text service using OpenAI Whisper (CPU-only)",
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
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model."""
        try:
            import whisper
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise RuntimeError(f"Could not load Whisper model: {e}")
    
    def transcribe_audio(self, audio_path: str, language: str = "en", task: str = "transcribe") -> Dict[str, Any]:
        """Transcribe audio using Whisper.
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'es', 'fr')
            task: Task type ('transcribe' or 'translate')
            
        Returns:
            Dictionary containing transcription results
        """
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
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {e}")

# Initialize STT service
stt_service = STTService(model_name=os.getenv("WHISPER_MODEL", "base"))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "stt", "model": stt_service.model_name}

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
    """
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
                
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Star Wars Chat - STT Service",
        "version": "1.0.0",
        "model": stt_service.model_name,
        "endpoints": {
            "health": "/health",
            "transcribe": "/transcribe"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
