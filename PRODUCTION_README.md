# 🌟 Star Wars RAG - Production Quality Implementation

A production-ready Star Wars chat application with Speech-to-Text (STT), Text-to-Speech (TTS), and Large Language Model (LLM) capabilities, allowing users to have voice conversations with Star Wars characters.

## 🏗️ Architecture Overview

The application follows a **clean architecture** pattern with clear separation of concerns:

```
src/star_wars_rag/
├── core/                    # Core interfaces and utilities
│   ├── interfaces.py       # Abstract base classes for services
│   ├── exceptions.py       # Custom exception hierarchy
│   ├── config.py          # Configuration management with Pydantic
│   └── logging.py         # Structured logging system
├── services/               # Concrete service implementations
│   ├── stt_service.py     # Whisper-based Speech-to-Text
│   ├── tts_service.py     # Coqui TTS-based Text-to-Speech
│   ├── llm_service.py     # Local LLM for character responses
│   └── chat_service.py    # Orchestration service
└── tests/                 # Comprehensive test suite
```

## 🎯 Key Features

### ✅ Production Quality Standards
- **Clean Architecture**: Clear separation of concerns with interfaces and implementations
- **Type Safety**: Full type hints throughout the codebase
- **Error Handling**: Comprehensive exception hierarchy with context
- **Logging**: Structured logging with performance metrics
- **Configuration**: Environment-based configuration with validation
- **Testing**: Unit tests with mocking and integration tests
- **Documentation**: Comprehensive docstrings and inline comments

### 🎤 Speech-to-Text (STT)
- **Model**: OpenAI Whisper (base model)
- **Features**: 
  - Async processing with thread pool execution
  - Input validation (file size, format, language)
  - Performance metrics and logging
  - Health checks and error recovery

### 🔊 Text-to-Speech (TTS)
- **Model**: Coqui TTS with multiple voice options
- **Features**:
  - Character-specific voice mapping
  - Async synthesis with caching
  - Output format validation
  - Graceful degradation when TTS unavailable

### 🤖 Large Language Model (LLM)
- **Model**: Local Phi-2 model via llama.cpp
- **Features**:
  - Character-specific prompt engineering
  - Context-aware responses
  - Configurable generation parameters
  - Character management system

### 💬 Chat Orchestration
- **Pipeline**: STT → LLM → TTS
- **Features**:
  - End-to-end audio processing
  - Error handling and recovery
  - Performance monitoring
  - Resource cleanup

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- FFmpeg (for audio processing)
- 8GB+ RAM (for LLM model)

### Installation

1. **Clone and setup environment**:
```bash
git clone <repository>
cd star-wars-chat-app
python -m venv env
source env/bin/activate  # Linux/Mac
# or
.\env\Scripts\activate   # Windows
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download models**:
```bash
# Download Phi-2 model (if not already present)
# Place in models/phi-2.Q4_K_M.gguf
```

4. **Run tests**:
```bash
python test_implementation.py
```

### Basic Usage

```python
import asyncio
from pathlib import Path
from src.star_wars_rag.core.config import get_default_config
from src.star_wars_rag.services import (
    WhisperSTTService,
    LocalLLMService,
    CoquiTTSService,
    StarWarsChatService
)

async def main():
    # Load configuration
    config = get_default_config()
    
    # Create services
    stt_service = WhisperSTTService(config.get_service_config('stt'))
    llm_service = LocalLLMService(config.get_service_config('llm'))
    tts_service = CoquiTTSService(config.get_service_config('tts'))
    
    # Create chat service
    chat_service = StarWarsChatService({
        'stt_service': stt_service,
        'llm_service': llm_service,
        'tts_service': tts_service
    })
    
    # Process audio message
    audio_path = Path("user_audio.wav")
    result = await chat_service.process_audio_message(
        audio_path, 
        character="Luke Skywalker"
    )
    
    print(f"Transcription: {result['transcription']}")
    print(f"Response: {result['response']}")
    print(f"Audio response: {result['audio_response']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🧪 Testing

### Running Tests
```bash
# Run core implementation tests
python test_implementation.py

# Run unit tests
pytest tests/

# Run specific service tests
pytest tests/test_stt_service.py
pytest tests/test_llm_service.py
pytest tests/test_tts_service.py
```

### Test Coverage
- ✅ Configuration system
- ✅ Logging system  
- ✅ Exception handling
- ✅ Interface definitions
- ✅ Service implementations
- ✅ Error scenarios
- ✅ Performance metrics

## ⚙️ Configuration

### Environment Variables
```bash
# STT Configuration
STT_MODEL_NAME=base
STT_LANGUAGE=en
STT_TEMP_DIR=/tmp

# TTS Configuration  
TTS_DEFAULT_VOICE=ljspeech
TTS_CACHE_DIR=/app/models/tts
TTS_TEMP_DIR=/tmp

# LLM Configuration
LLM_MODEL_PATH=models/phi-2.Q4_K_M.gguf
LLM_N_CTX=2048
LLM_N_THREADS=4
LLM_N_GPU_LAYERS=0

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE_PATH=logs/app.log
```

### Configuration Validation
The system validates all configuration on startup:
- Model file existence
- Directory permissions
- Required dependencies
- Service connectivity

## 📊 Monitoring & Logging

### Structured Logging
All services use structured logging with:
- Service identification
- Performance metrics
- Error context
- Request/response correlation

### Performance Metrics
- Transcription duration
- Synthesis duration  
- Generation duration
- End-to-end processing time

### Health Checks
Each service provides health check endpoints:
```python
health = await service.health_check()
# Returns: {"status": "healthy", "details": {...}}
```

## 🔧 Extending the System

### Adding New Characters
1. Update `characters.json`:
```json
{
  "Han Solo": {
    "description": "A smuggler turned hero",
    "personality": "Confident, sarcastic, and loyal",
    "speaking_style": "Direct and witty with Corellian accent"
  }
}
```

2. The system automatically loads new characters on startup.

### Adding New Voices
1. Update voice mapping in `tts_service.py`:
```python
voice_mapping = {
    "Han Solo": "male_voice_model",
    # ... existing mappings
}
```

### Adding New Models
1. Implement new service class inheriting from base interface
2. Add factory function for service creation
3. Update configuration schema
4. Add tests for new implementation

## 🐳 Docker Deployment

### Building Services
```bash
# Build all services
docker-compose build

# Build specific service
docker build -f Dockerfile.stt -t star-wars-stt .
docker build -f Dockerfile.tts -t star-wars-tts .
docker build -f Dockerfile.llm -t star-wars-llm .
```

### Running with Docker Compose
```bash
docker-compose up -d
```

### Service Endpoints
- **STT Service**: `http://localhost:5001`
- **TTS Service**: `http://localhost:5002`  
- **LLM Service**: `http://localhost:5003`
- **Frontend**: `http://localhost:3000`

## 🔒 Security Considerations

### Input Validation
- Audio file size limits (25MB max)
- Text length limits (2000 chars max)
- File format validation
- Language code validation

### Error Handling
- Graceful degradation when services unavailable
- No sensitive information in error messages
- Proper resource cleanup on errors

### Configuration Security
- Environment variable validation
- No hardcoded secrets
- Secure default configurations

## 📈 Performance Optimization

### Async Processing
- All I/O operations are async
- Thread pool execution for CPU-intensive tasks
- Non-blocking service communication

### Resource Management
- Lazy model loading
- Automatic cleanup of temporary files
- Memory-efficient audio processing

### Caching
- TTS model caching
- Character prompt caching
- Health check result caching

## 🐛 Troubleshooting

### Common Issues

**TTS Service Not Available**
```
ERROR: No module named 'TTS'
```
**Solution**: Install TTS library or run without TTS (system degrades gracefully)

**Model Loading Fails**
```
ERROR: Model file not found
```
**Solution**: Ensure model file exists at configured path

**Audio Processing Fails**
```
ERROR: File too large
```
**Solution**: Check audio file size (max 25MB)

### Debug Mode
Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
```

### Health Checks
Check service health:
```python
health = await service.health_check()
print(health)
```

## 🤝 Contributing

### Code Standards
- Follow PEP 8 style guide
- Use type hints for all functions
- Write docstrings for all classes/methods
- Include unit tests for new features
- Use structured logging

### Testing Requirements
- Unit tests for all new code
- Integration tests for service interactions
- Performance tests for critical paths
- Error scenario testing

### Pull Request Process
1. Create feature branch
2. Implement changes with tests
3. Run full test suite
4. Update documentation
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for Whisper model
- Coqui AI for TTS library
- Microsoft for Phi-2 model
- The Star Wars universe for inspiration

---

**Ready for production deployment!** 🚀

This implementation provides a solid foundation for a production Star Wars chat application with proper error handling, monitoring, and extensibility.
