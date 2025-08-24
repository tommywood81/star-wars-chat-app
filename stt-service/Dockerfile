# STT Service Dockerfile - Ultra Lightweight CPU Only
FROM python:3.10-slim

# Set CPU-only environment variables
ENV CUDA_VISIBLE_DEVICES=""
ENV PYTORCH_CUDA_ALLOC_CONF=""
ENV TORCH_CUDA_ARCH_LIST=""
ENV WHISPER_MODEL=base

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-stt.txt .

# Install Python dependencies (CPU only, no cache)
RUN pip install --no-cache-dir fastapi==0.104.1 uvicorn==0.24.0 pydantic==2.5.0 structlog==23.2.0 python-multipart==0.0.6 && \
    pip install --no-cache-dir "numpy<2.0.0" && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.1.0+cpu torchaudio==2.1.0+cpu && \
    pip install --no-cache-dir openai-whisper==20231117

# Copy the standalone STT service
COPY stt_service_standalone.py /app/stt_service.py

# Create necessary directories
RUN mkdir -p /app/temp

# Set environment variables
ENV PYTHONPATH=/app
ENV TEMP_DIR=/app/temp

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Run the application
CMD ["uvicorn", "stt_service:app", "--host", "0.0.0.0", "--port", "5001"]
