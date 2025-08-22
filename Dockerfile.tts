# TTS Service Dockerfile - Optimized (CPU Only)
FROM python:3.10-slim

# Install system dependencies (minimal for gTTS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-tts.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-tts.txt

# Copy the standalone TTS service
COPY tts_service_standalone.py /app/tts_service.py

# Create necessary directories
RUN mkdir -p /app/temp /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV TEMP_DIR=/app/temp

# Expose port
EXPOSE 5002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5002/health || exit 1

# Run the application
CMD ["uvicorn", "tts_service:app", "--host", "0.0.0.0", "--port", "5002"]
