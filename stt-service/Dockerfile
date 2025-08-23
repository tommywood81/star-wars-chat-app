# STT Service Dockerfile - Optimized (CPU Only)
FROM python:3.10-slim

# Set CPU-only environment variables
ENV CUDA_VISIBLE_DEVICES=""
ENV PYTORCH_CUDA_ALLOC_CONF=""
ENV TORCH_CUDA_ARCH_LIST=""

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-stt.txt .

# Install Python dependencies (CPU only)
RUN pip install --no-cache-dir -r requirements-stt.txt

# Copy the standalone STT service
COPY stt_service_standalone.py /app/stt_service.py

# Create necessary directories
RUN mkdir -p /app/temp /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV WHISPER_MODEL=base
ENV TEMP_DIR=/app/temp

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Run the application
CMD ["uvicorn", "stt_service:app", "--host", "0.0.0.0", "--port", "5001"]
