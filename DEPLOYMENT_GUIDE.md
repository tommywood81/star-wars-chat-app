# Star Wars Chat App - Deployment Guide

## 🚀 Overview

This guide covers the deployment of the Star Wars Chat App with Speech-to-Text (STT), Text-to-Speech (TTS), and LLM services. The application is designed as a microservices architecture with separate containers for each service.

## 📋 Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   STT Service   │    │   TTS Service   │
│   (React)       │    │   (Whisper)     │    │   (Coqui TTS)   │
│   Port: 3000    │    │   Port: 5001    │    │   Port: 5002    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   LLM Service   │
                    │   (Phi-2)       │
                    │   Port: 5003    │
                    └─────────────────┘
```

## 🛠️ Prerequisites

- Docker and Docker Compose
- Python 3.10+
- Node.js 20+ (for frontend development)
- At least 8GB RAM (16GB recommended)
- 20GB+ free disk space

## 📦 Services

### 1. STT Service (Speech-to-Text)
- **Technology**: OpenAI Whisper
- **Port**: 5001
- **Features**:
  - Real-time audio transcription
  - Multiple language support
  - Confidence scoring
  - Audio file processing

### 2. TTS Service (Text-to-Speech)
- **Technology**: Coqui TTS
- **Port**: 5002
- **Features**:
  - Multiple voice models
  - Speed control
  - Emotion support (for some voices)
  - Audio file generation

### 3. LLM Service (Language Model)
- **Technology**: Phi-2 (via llama.cpp)
- **Port**: 5003
- **Features**:
  - Character-specific responses
  - Context-aware conversations
  - Multiple Star Wars characters
  - Configurable generation parameters

### 4. Frontend Service
- **Technology**: React + Nginx
- **Port**: 3000
- **Features**:
  - Modern UI with Star Wars theme
  - Real-time chat interface
  - Voice recording and playback
  - Character selection
  - Service status monitoring

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd star-wars-chat-app
```

### 2. Environment Setup

```bash
# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: .\env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Models

```bash
# Create models directory
mkdir -p models

# Download Phi-2 model (if not already present)
# The model should be placed in models/phi-2.Q4_K_M.gguf
```

### 4. Run with Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

### 5. Access the Application

- **Frontend**: http://localhost:3000
- **STT API**: http://localhost:5001
- **TTS API**: http://localhost:5002
- **LLM API**: http://localhost:5003

## 🔧 Manual Service Startup

### STT Service
```bash
cd src/star_wars_rag
python stt_service.py
```

### TTS Service
```bash
cd src/star_wars_rag
python tts_service.py
```

### LLM Service
```bash
cd src/star_wars_rag
python llm_service.py
```

### Frontend (Development)
```bash
npm install
npm start
```

## 🧪 Testing

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Service Tests
```bash
# STT Service tests
python -m pytest tests/test_stt_service.py -v

# TTS Service tests
python -m pytest tests/test_tts_service.py -v

# LLM Service tests
python -m pytest tests/test_llm_service.py -v
```

### Quick Health Check
```bash
python test_simple.py
```

## 📡 API Endpoints

### STT Service (Port 5001)

- `GET /health` - Service health check
- `POST /transcribe` - Transcribe audio file
- `GET /models` - List available Whisper models

### TTS Service (Port 5002)

- `GET /health` - Service health check
- `POST /synthesize` - Convert text to speech
- `GET /voices` - List available voices
- `GET /voices/{voice}` - Get voice information
- `POST /synthesize-simple` - Simple text-to-speech

### LLM Service (Port 5003)

- `GET /health` - Service health check
- `POST /chat` - Chat with Star Wars character
- `GET /characters` - List available characters
- `GET /characters/{name}` - Get character information
- `GET /model-info` - Get model information

## 🎭 Available Characters

1. **Luke Skywalker** - Jedi Knight and hero
2. **Darth Vader** - Dark Lord of the Sith
3. **Princess Leia** - Rebel leader
4. **Han Solo** - Smuggler and captain
5. **Yoda** - Wise Jedi Master
6. **Obi-Wan Kenobi** - Jedi Master and mentor
7. **Chewbacca** - Wookiee warrior
8. **R2-D2** - Astromech droid

## 🔧 Configuration

### Environment Variables

```bash
# STT Service
WHISPER_MODEL=base
TEMP_DIR=/app/temp

# TTS Service
TTS_CACHE_DIR=/app/models/tts
DEFAULT_VOICE=ljspeech

# LLM Service
MODEL_PATH=/app/models/phi-2.Q4_K_M.gguf
HF_HOME=/app/models/hf

# Frontend
REACT_APP_STT_URL=http://localhost:5001
REACT_APP_TTS_URL=http://localhost:5002
REACT_APP_LLM_URL=http://localhost:5003
```

### Docker Configuration

The application uses separate Dockerfiles for each service:

- `Dockerfile.stt` - STT service with Whisper
- `Dockerfile.tts` - TTS service with Coqui TTS
- `Dockerfile.llm` - LLM service with Phi-2
- `Dockerfile.frontend` - React frontend with Nginx

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce model size (use 'tiny' instead of 'base' for Whisper)
   - Increase Docker memory limits
   - Use CPU-only builds

2. **Model Loading Failures**
   - Check model file paths
   - Verify model file integrity
   - Ensure sufficient disk space

3. **Service Communication Issues**
   - Check port availability
   - Verify Docker network configuration
   - Check service health endpoints

4. **Audio Issues**
   - Ensure microphone permissions
   - Check browser audio support
   - Verify audio file formats

### Health Checks

```bash
# Check STT service
curl http://localhost:5001/health

# Check TTS service
curl http://localhost:5002/health

# Check LLM service
curl http://localhost:5003/health
```

### Logs

```bash
# View all service logs
docker-compose logs

# View specific service logs
docker-compose logs stt
docker-compose logs tts
docker-compose logs llm
docker-compose logs frontend
```

## 🔒 Security Considerations

1. **API Security**
   - Implement authentication for production
   - Use HTTPS in production
   - Rate limiting for API endpoints

2. **Model Security**
   - Keep models in secure locations
   - Implement model access controls
   - Regular security updates

3. **Data Privacy**
   - Audio data is processed locally
   - No data is stored permanently
   - Implement data retention policies

## 📈 Performance Optimization

1. **Model Optimization**
   - Use quantized models for faster inference
   - GPU acceleration for supported models
   - Model caching and warm-up

2. **Service Optimization**
   - Load balancing for multiple instances
   - Connection pooling
   - Response caching

3. **Frontend Optimization**
   - Code splitting
   - Asset optimization
   - CDN for static assets

## 🚀 Production Deployment

### Digital Ocean Deployment

1. **Create Droplet**
   ```bash
   # Use Ubuntu 22.04 LTS
   # Minimum specs: 4GB RAM, 2 vCPUs, 80GB SSD
   ```

2. **Install Docker**
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   ```

3. **Deploy Application**
   ```bash
   git clone <repository-url>
   cd star-wars-chat-app
   docker-compose -f docker-compose.production.yml up -d
   ```

### AWS Deployment

1. **EC2 Instance Setup**
   - Use t3.large or larger
   - Ubuntu 22.04 LTS
   - Security groups for ports 80, 443, 3000

2. **Load Balancer**
   - Application Load Balancer
   - SSL/TLS termination
   - Health checks

3. **Auto Scaling**
   - Scale based on CPU/memory usage
   - Minimum 2 instances
   - Maximum 10 instances

## 📊 Monitoring

### Metrics to Monitor

1. **Service Health**
   - Response times
   - Error rates
   - Service availability

2. **Resource Usage**
   - CPU utilization
   - Memory usage
   - Disk space

3. **User Experience**
   - Chat response times
   - Audio quality
   - User satisfaction

### Monitoring Tools

- **Prometheus** - Metrics collection
- **Grafana** - Visualization
- **ELK Stack** - Log management
- **Health checks** - Service monitoring

## 🔄 Updates and Maintenance

### Regular Maintenance

1. **Security Updates**
   - Update dependencies monthly
   - Monitor for vulnerabilities
   - Apply patches promptly

2. **Model Updates**
   - Update Whisper models quarterly
   - Test new TTS voices
   - Evaluate new LLM models

3. **Performance Monitoring**
   - Monitor response times
   - Track resource usage
   - Optimize bottlenecks

### Backup Strategy

1. **Configuration Backup**
   - Docker Compose files
   - Environment variables
   - Model configurations

2. **Data Backup**
   - User conversations (if stored)
   - Custom models
   - Configuration files

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Review service logs
3. Test individual components
4. Create detailed bug reports

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

---

**Happy chatting with Star Wars characters! May the Force be with you! 🌟**
