"""
Standalone STT Service - Minimal Version
Provides a lightweight speech-to-text service without heavy ML dependencies.
For production, this can be replaced with a full Whisper implementation.
"""

import os
import tempfile
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import structlog

# Configure logging
logger = structlog.get_logger()

app = FastAPI(title="STT Service", version="1.0.0")

class TranscriptionRequest(BaseModel):
    """Request model for transcription."""
    language: Optional[str] = "en"

class TranscriptionResponse(BaseModel):
    """Response model for transcription."""
    text: str
    language: str
    confidence: float
    duration: Optional[float] = None

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "stt"}

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = "en"
):
    """
    Transcribe uploaded audio file.
    
    This is a minimal implementation. For production use,
    replace with actual Whisper or other STT service.
    """
    try:
        logger.info("Received transcription request", 
                   filename=file.filename, 
                   language=language)
        
        # Validate file type
        if not file.filename or not file.filename.lower().endswith(('.wav', '.mp3', '.m4a')):
            raise HTTPException(status_code=400, detail="Invalid audio file format")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # For now, return a placeholder response
            # In production, this would call actual STT service
            logger.info("Processing audio file", path=temp_file_path)
            
            # Simulate processing time
            await asyncio.sleep(0.1)
            
            # Placeholder response - replace with actual transcription
            response = TranscriptionResponse(
                text="[Audio transcription placeholder - implement actual STT]",
                language=language,
                confidence=0.95,
                duration=1.0
            )
            
            logger.info("Transcription completed", 
                       text_length=len(response.text),
                       confidence=response.confidence)
            
            return response
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error("Transcription failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "STT Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "transcribe": "/transcribe"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
