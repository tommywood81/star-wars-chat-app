# LLM Service Dockerfile - Optimized (CPU Only)
FROM python:3.10-slim

# Set CPU-only environment variables
ENV CUDA_VISIBLE_DEVICES=""
ENV PYTORCH_CUDA_ALLOC_CONF=""
ENV TORCH_CUDA_ARCH_LIST=""

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-llm.txt .

# Install Python dependencies (CPU only)
RUN pip install --no-cache-dir -r requirements-llm.txt

# Copy the standalone LLM service
COPY llm_service_standalone.py /app/llm_service.py

# Create necessary directories
RUN mkdir -p /app/temp /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV TEMP_DIR=/app/temp

# Expose port
EXPOSE 5003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5003/health || exit 1

# Run the application
CMD ["uvicorn", "llm_service:app", "--host", "0.0.0.0", "--port", "5003"]
