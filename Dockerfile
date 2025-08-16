# Star Wars RAG Chat App Docker Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies in stages for better caching and faster builds
# First install the heaviest/slowest dependencies
RUN pip install --no-cache-dir torch>=1.12.0 --index-url https://download.pytorch.org/whl/cpu

# Install other heavy ML dependencies
RUN pip install --no-cache-dir sentence-transformers>=2.2.0

# Install remaining dependencies
RUN pip install --no-cache-dir \
    pandas>=1.5.0 \
    numpy>=1.21.0 \
    scikit-learn>=1.1.0 \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    pydantic>=2.5.0 \
    python-multipart>=0.0.6 \
    psycopg2-binary>=2.9.0 \
    asyncpg>=0.29.0 \
    streamlit>=1.28.0 \
    python-dotenv>=1.0.0 \
    requests>=2.31.0

# Skip heavy optional dependencies for faster builds
# RUN pip install --no-cache-dir llama-cpp-python>=0.2.0
# RUN pip install --no-cache-dir huggingface-hub>=0.20.0

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY *.py ./

# Create models directory
RUN mkdir -p models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI application
CMD ["uvicorn", "src.star_wars_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
