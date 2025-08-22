"""
Standalone Text-to-Speech (TTS) Service for Star Wars Chat App.

This is a completely isolated TTS service that doesn't import any other services
to avoid loading heavy models like Whisper or Phi-2.
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Star Wars Chat - TTS Service",
    description="Text-to-Speech service using Google TTS (CPU-only)",
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

class TTSRequest(BaseModel):
    """Request model for text-to-speech conversion."""
    text: str
    voice: Optional[str] = "en"
    speed: Optional[float] = 1.0
    emotion: Optional[str] = None

class TTSResponse(BaseModel):
    """Response model for text-to-speech conversion."""
    audio_file: str
    duration: Optional[float] = None
    voice: str
    text_length: int

class TTSService:
    """Text-to-Speech service using Google TTS (gTTS)."""
    
    def __init__(self, default_voice: str = "en"):
        """Initialize the TTS service.
        
        Args:
            default_voice: Default voice language to use
        """
        self.default_voice = default_voice
        self.available_voices = {
            "en": "English",
            "en-gb": "English (UK)",
            "en-au": "English (Australia)",
            "en-us": "English (US)",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese"
        }
        self._test_tts()
    
    def _test_tts(self):
        """Test TTS functionality."""
        try:
            from gtts import gTTS
            logger.info("gTTS imported successfully - TTS service ready")
        except Exception as e:
            logger.error(f"Failed to import gTTS: {e}")
            raise RuntimeError(f"Could not initialize TTS service: {e}")
    
    def synthesize_speech(self, text: str, voice: str = None, speed: float = 1.0, 
                         emotion: str = None) -> Dict[str, Any]:
        """Synthesize speech from text using Google TTS.
        
        Args:
            text: Text to convert to speech
            voice: Voice language to use
            speed: Speech speed multiplier (not supported by gTTS, kept for compatibility)
            emotion: Emotion to apply (not supported by gTTS, kept for compatibility)
            
        Returns:
            Dictionary containing audio file path and metadata
        """
        try:
            logger.info(f"Synthesizing speech for text: {text[:50]}...")
            
            # Import gTTS here to avoid import issues during testing
            from gtts import gTTS
            
            # Validate text
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")
            
            # Use default voice if none specified
            if voice is None:
                voice = self.default_voice
            
            # Validate voice
            if voice not in self.available_voices:
                voice = self.default_voice
                logger.warning(f"Voice '{voice}' not found, using default: {self.default_voice}")
            
            # Create temporary file for audio output
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                output_path = temp_file.name
            
            # Synthesize speech using Google TTS
            tts = gTTS(text=text, lang=voice, slow=False)
            tts.save(output_path)
            
            # Get file size and estimate duration
            file_size = os.path.getsize(output_path)
            estimated_duration = len(text.split()) * 0.5  # Rough estimate: 0.5s per word
            
            logger.info(f"Speech synthesis completed: {output_path}")
            
            return {
                "audio_file": output_path,
                "duration": estimated_duration,
                "voice": voice,
                "text_length": len(text),
                "file_size": file_size
            }
            
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            # Clean up temporary file if it exists
            if 'output_path' in locals() and os.path.exists(output_path):
                os.unlink(output_path)
            raise RuntimeError(f"Speech synthesis failed: {e}")
    
    def get_available_voices(self) -> List[str]:
        """Get list of available voices."""
        return list(self.available_voices.keys())
    
    def get_voice_info(self, voice: str) -> Dict[str, Any]:
        """Get information about a specific voice."""
        if voice not in self.available_voices:
            raise ValueError(f"Voice '{voice}' not found")
        
        return {
            "name": voice,
            "language": self.available_voices[voice],
            "supported_features": ["text"],
            "device": "cpu",
            "provider": "google"
        }

# Initialize TTS service
try:
    tts_service = TTSService()
    logger.info("TTS service initialized successfully")
except Exception as e:
    logger.error(f"TTS service initialization failed: {e}")
    tts_service = None

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if tts_service is None:
        return {
            "status": "unhealthy", 
            "service": "tts", 
            "error": "TTS service not initialized - missing dependencies",
            "device": "cpu",
            "provider": "google"
        }
    return {
        "status": "healthy", 
        "service": "tts", 
        "default_voice": tts_service.default_voice,
        "device": "cpu",
        "provider": "google"
    }

@app.post("/synthesize", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest):
    """Synthesize speech from text.
    
    Args:
        request: TTSRequest containing text and parameters
        
    Returns:
        TTSResponse with audio file path and metadata
    """
    if tts_service is None:
        raise HTTPException(
            status_code=503,
            detail="TTS service not available - missing dependencies"
        )
    
    try:
        result = tts_service.synthesize_speech(
            text=request.text,
            voice=request.voice,
            speed=request.speed,
            emotion=request.emotion
        )
        
        return TTSResponse(
            audio_file=result["audio_file"],
            duration=result["duration"],
            voice=result["voice"],
            text_length=result["text_length"]
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Synthesis endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """Serve audio file.
    
    Args:
        filename: Name of the audio file to serve
        
    Returns:
        Audio file response
    """
    try:
        # Security: only allow .mp3 files
        if not filename.endswith('.mp3'):
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        file_path = Path("/tmp") / filename  # Assuming files are in /tmp
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        return FileResponse(
            path=str(file_path),
            media_type="audio/mpeg",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio file serving error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/voices")
async def list_voices():
    """List available voices."""
    if tts_service is None:
        return {
            "available_voices": [],
            "default_voice": "none",
            "error": "TTS service not available",
            "provider": "google"
        }
    
    voices = tts_service.get_available_voices()
    return {
        "available_voices": voices,
        "default_voice": tts_service.default_voice,
        "device": "cpu",
        "provider": "google"
    }

@app.get("/voices/{voice}")
async def get_voice_info(voice: str):
    """Get information about a specific voice."""
    if tts_service is None:
        raise HTTPException(
            status_code=503,
            detail="TTS service not available"
        )
    
    try:
        info = tts_service.get_voice_info(voice)
        return info
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/synthesize-simple")
async def synthesize_speech_simple(
    text: str = Query(..., description="Text to convert to speech"),
    voice: str = Query("en", description="Voice language to use")
):
    """Simple synthesis endpoint for direct text input."""
    if tts_service is None:
        raise HTTPException(
            status_code=503,
            detail="TTS service not available"
        )
    
    try:
        result = tts_service.synthesize_speech(text=text, voice=voice)
        
        # Return the audio file directly
        return FileResponse(
            path=result["audio_file"],
            media_type="audio/mpeg",
            filename="synthesized_speech.mp3"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Simple synthesis endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)
