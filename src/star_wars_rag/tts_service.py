"""
Text-to-Speech (TTS) Service for Star Wars Chat App.

This module provides FastAPI endpoints for converting text to speech
using Coqui TTS. It's designed to run as a separate microservice.
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Star Wars Chat - TTS Service",
    description="Text-to-Speech service using Coqui TTS",
    version="1.0.0"
)

class TTSRequest(BaseModel):
    """Request model for text-to-speech conversion."""
    text: str
    voice: Optional[str] = "ljspeech"
    speed: Optional[float] = 1.0
    emotion: Optional[str] = None

class TTSResponse(BaseModel):
    """Response model for text-to-speech conversion."""
    audio_file: str
    duration: Optional[float] = None
    voice: str
    text_length: int

class TTSService:
    """Text-to-Speech service using Coqui TTS."""
    
    def __init__(self, default_voice: str = "ljspeech"):
        """Initialize the TTS service.
        
        Args:
            default_voice: Default voice model to use
        """
        self.default_voice = default_voice
        self.tts = None
        self.available_voices = {
            "ljspeech": "tts_models/en/ljspeech/tacotron2-DDC",
            "vctk": "tts_models/multilingual/multi-dataset/your_tts",
            "fastspeech2": "tts_models/en/ljspeech/fast_pitch"
        }
        self._load_model()
    
    def _load_model(self):
        """Load the TTS model."""
        try:
            logger.info(f"Loading TTS model: {self.default_voice}")
            # Import TTS here to avoid import issues during testing
            from TTS.api import TTS
            
            model_name = self.available_voices.get(self.default_voice, self.available_voices["ljspeech"])
            self.tts = TTS(model_name=model_name, progress_bar=False, gpu=False)
            logger.info("TTS model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            raise RuntimeError(f"Could not load TTS model: {e}")
    
    def synthesize_speech(self, text: str, voice: str = None, speed: float = 1.0, 
                         emotion: str = None) -> Dict[str, Any]:
        """Synthesize speech from text.
        
        Args:
            text: Text to convert to speech
            voice: Voice model to use
            speed: Speech speed multiplier
            emotion: Emotion to apply (if supported)
            
        Returns:
            Dictionary containing audio file path and metadata
        """
        try:
            logger.info(f"Synthesizing speech for text: {text[:50]}...")
            
            # Validate text
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")
            
            # Use default voice if none specified
            if voice is None:
                voice = self.default_voice
            
            # Create temporary file for audio output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                output_path = temp_file.name
            
            # Synthesize speech
            self.tts.tts_to_file(
                text=text,
                file_path=output_path,
                speed=speed
            )
            
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
            "model": self.available_voices[voice],
            "supported_features": ["speed", "emotion"] if voice == "vctk" else ["speed"]
        }

# Initialize TTS service (with error handling for missing dependencies)
try:
    tts_service = TTSService()
except Exception as e:
    logger.warning(f"TTS service initialization failed: {e}")
    tts_service = None

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if tts_service is None:
        return {
            "status": "unhealthy", 
            "service": "tts", 
            "error": "TTS service not initialized - missing dependencies"
        }
    return {
        "status": "healthy", 
        "service": "tts", 
        "default_voice": tts_service.default_voice
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
        # Security: only allow .wav files
        if not filename.endswith('.wav'):
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        file_path = Path("/tmp") / filename  # Assuming files are in /tmp
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        return FileResponse(
            path=str(file_path),
            media_type="audio/wav",
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
    voices = tts_service.get_available_voices()
    return {
        "available_voices": voices,
        "default_voice": tts_service.default_voice
    }

@app.get("/voices/{voice}")
async def get_voice_info(voice: str):
    """Get information about a specific voice."""
    try:
        info = tts_service.get_voice_info(voice)
        return info
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/synthesize-simple")
async def synthesize_speech_simple(
    text: str = Query(..., description="Text to convert to speech"),
    voice: str = Query("ljspeech", description="Voice to use")
):
    """Simple synthesis endpoint for direct text input."""
    try:
        result = tts_service.synthesize_speech(text=text, voice=voice)
        
        # Return the audio file directly
        return FileResponse(
            path=result["audio_file"],
            media_type="audio/wav",
            filename="synthesized_speech.wav"
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
