"""
Text-to-Speech Service using gTTS.

This service requires proper gTTS setup - no fallbacks, fails fast on errors.
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import structlog

# Configure logging
logger = structlog.get_logger()

app = FastAPI(title="TTS Service", version="1.0.0")

class SynthesisRequest(BaseModel):
    """Request model for text-to-speech conversion."""
    text: str
    voice: Optional[str] = "en"
    language: Optional[str] = "en"

class SynthesisResponse(BaseModel):
    """Response model for text-to-speech conversion."""
    audio_path: str
    duration: Optional[float] = None
    language: str

class TTSService:
    """Text-to-Speech service using gTTS."""
    
    def __init__(self, default_voice: str = "en"):
        """Initialize the TTS service.
        
        Args:
            default_voice: Default voice/language to use
            
        Raises:
            RuntimeError: If gTTS cannot be loaded or initialized
        """
        self.default_voice = default_voice
        self._load_tts()
    
    def _load_tts(self):
        """Load the gTTS library - no fallbacks."""
        try:
            from gtts import gTTS
            self.gtts = gTTS
            logger.info("gTTS library loaded successfully")
        except ImportError as e:
            raise RuntimeError(f"gTTS library not available: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load gTTS: {e}")
    
    def synthesize_speech(self, text: str, voice: str = "en", language: str = "en") -> Dict[str, Any]:
        """Synthesize speech from text using gTTS.
        
        Args:
            text: Text to convert to speech
            voice: Voice identifier
            language: Language code
            
        Returns:
            Dictionary containing audio file path and metadata
            
        Raises:
            RuntimeError: If synthesis fails
        """
        if not hasattr(self, 'gtts'):
            raise RuntimeError("gTTS not loaded")
        
        try:
            logger.info(f"Synthesizing speech for text: {text[:50]}...")
            
            # Create temporary file for audio
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_path = temp_file.name
            temp_file.close()
            
            # Generate speech
            tts = self.gtts(text=text, lang=language, slow=False)
            tts.save(temp_path)
            
            # Get file size for duration estimation (rough approximation)
            file_size = os.path.getsize(temp_path)
            estimated_duration = file_size / 16000  # Rough estimate: 16kbps
            
            return {
                "audio_path": temp_path,
                "duration": estimated_duration,
                "language": language,
                "file_size": file_size
            }
            
        except Exception as e:
            # Clean up temp file if it exists
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            raise RuntimeError(f"Speech synthesis failed: {e}")

# Initialize TTS service
try:
    tts_service = TTSService(default_voice=os.getenv("TTS_DEFAULT_VOICE", "en"))
except Exception as e:
    logger.error(f"Failed to initialize TTS service: {e}")
    tts_service = None

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not tts_service:
        raise HTTPException(status_code=503, detail="TTS service not initialized")
    
    return {
        "status": "healthy",
        "service": "tts",
        "default_voice": tts_service.default_voice
    }

@app.post("/synthesize", response_model=SynthesisResponse)
async def synthesize_speech(request: SynthesisRequest):
    """Synthesize speech from text.
    
    Args:
        request: Synthesis request with text and voice parameters
        
    Returns:
        Synthesis result with audio file path
        
    Raises:
        HTTPException: If synthesis fails
    """
    if not tts_service:
        raise HTTPException(status_code=503, detail="TTS service not available")
    
    try:
        result = tts_service.synthesize_speech(
            text=request.text,
            voice=request.voice,
            language=request.language
        )
        
        return SynthesisResponse(
            audio_path=result["audio_path"],
            duration=result["duration"],
            language=result["language"]
        )
        
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in synthesis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "TTS Service",
        "version": "1.0.0",
        "default_voice": tts_service.default_voice if tts_service else "not loaded",
        "status": "running" if tts_service else "failed",
        "endpoints": {
            "health": "/health",
            "synthesize": "/synthesize"
        }
    }
