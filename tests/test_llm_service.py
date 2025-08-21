"""
Tests for the LLM service.

This module contains unit tests and integration tests for the LLM service
for Star Wars character chat.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

from src.star_wars_rag.llm_service import (
    app, LLMService, ChatRequest, ChatResponse, CharacterInfo
)


class TestLLMService:
    """Test cases for LLMService class."""
    
    @patch('src.star_wars_rag.llm_service.LocalLLM')
    @patch('src.star_wars_rag.llm_service.CharacterPromptBuilder')
    def test_llm_service_initialization(self, mock_prompt_builder, mock_llm_class):
        """Test LLM service initialization."""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_prompt_builder_instance = Mock()
        mock_prompt_builder.return_value = mock_prompt_builder_instance
        
        service = LLMService(model_path="test_model.gguf")
        
        assert service.model_path.name == "test_model.gguf"
        assert service.llm == mock_llm
        assert service.prompt_builder == mock_prompt_builder_instance
        mock_llm_class.assert_called_once()
    
    @patch('src.star_wars_rag.llm_service.LocalLLM')
    @patch('src.star_wars_rag.llm_service.CharacterPromptBuilder')
    def test_llm_service_initialization_failure(self, mock_prompt_builder, mock_llm_class):
        """Test LLM service initialization failure."""
        mock_llm_class.side_effect = Exception("Model loading failed")
        
        with pytest.raises(RuntimeError, match="Could not load LLM model"):
            LLMService(model_path="test_model.gguf")
    
    @patch('src.star_wars_rag.llm_service.LocalLLM')
    @patch('src.star_wars_rag.llm_service.CharacterPromptBuilder')
    def test_chat_with_character_success(self, mock_prompt_builder, mock_llm_class):
        """Test successful character chat."""
        # Mock LLM
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        # Mock prompt builder
        mock_prompt_builder_instance = Mock()
        mock_prompt_builder.return_value = mock_prompt_builder_instance
        mock_prompt_builder_instance.build_character_prompt.return_value = "Test prompt"
        
        # Mock LLM response
        mock_llm.generate.return_value = {
            'response': 'May the Force be with you, young one.',
            'metadata': {
                'prompt_tokens': 50,
                'completion_tokens': 10,
                'total_tokens': 60,
                'generation_time_seconds': 2.5,
                'tokens_per_second': 4.0,
                'model': 'test_model.gguf',
                'stop_reason': 'stop'
            }
        }
        
        service = LLMService(model_path="test_model.gguf")
        
        result = service.chat_with_character(
            message="Hello Luke!",
            character="Luke Skywalker",
            context="Previous conversation context"
        )
        
        assert result["response"] == "May the Force be with you, young one."
        assert result["character"] == "Luke Skywalker"
        assert "metadata" in result
        assert result["metadata"]["prompt_tokens"] == 50
        assert result["metadata"]["prompt_length"] == 11  # "Test prompt"
        assert result["metadata"]["response_length"] == 30  # Response length
        
        # Verify prompt builder was called
        mock_prompt_builder_instance.build_character_prompt.assert_called_once_with(
            character="Luke Skywalker",
            user_message="Hello Luke!",
            context="Previous conversation context"
        )
        
        # Verify LLM was called
        mock_llm.generate.assert_called_once_with(
            prompt="Test prompt",
            max_tokens=200,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            stop=["\n", "User:", "Human:", "Assistant:"]
        )
    
    @patch('src.star_wars_rag.llm_service.LocalLLM')
    @patch('src.star_wars_rag.llm_service.CharacterPromptBuilder')
    def test_chat_with_character_response_cleaning(self, mock_prompt_builder, mock_llm_class):
        """Test response cleaning when Assistant: prefix is present."""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        mock_prompt_builder_instance = Mock()
        mock_prompt_builder.return_value = mock_prompt_builder_instance
        mock_prompt_builder_instance.build_character_prompt.return_value = "Test prompt"
        
        # Mock response with Assistant: prefix
        mock_llm.generate.return_value = {
            'response': 'Assistant: Hello there! General Kenobi.',
            'metadata': {
                'prompt_tokens': 50,
                'completion_tokens': 10,
                'total_tokens': 60,
                'generation_time_seconds': 2.5,
                'tokens_per_second': 4.0,
                'model': 'test_model.gguf',
                'stop_reason': 'stop'
            }
        }
        
        service = LLMService(model_path="test_model.gguf")
        
        result = service.chat_with_character(
            message="Hello Obi-Wan!",
            character="Obi-Wan Kenobi"
        )
        
        assert result["response"] == "Hello there! General Kenobi."
    
    @patch('src.star_wars_rag.llm_service.LocalLLM')
    @patch('src.star_wars_rag.llm_service.CharacterPromptBuilder')
    def test_chat_with_character_llm_failure(self, mock_prompt_builder, mock_llm_class):
        """Test character chat when LLM fails."""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        mock_prompt_builder_instance = Mock()
        mock_prompt_builder.return_value = mock_prompt_builder_instance
        mock_prompt_builder_instance.build_character_prompt.return_value = "Test prompt"
        
        mock_llm.generate.side_effect = Exception("LLM generation failed")
        
        service = LLMService(model_path="test_model.gguf")
        
        with pytest.raises(RuntimeError, match="Character chat failed"):
            service.chat_with_character(
                message="Hello!",
                character="Luke Skywalker"
            )
    
    def test_get_available_characters(self):
        """Test getting available characters."""
        with patch('src.star_wars_rag.llm_service.LocalLLM') as mock_llm_class:
            mock_llm = Mock()
            mock_llm_class.return_value = mock_llm
            
            service = LLMService(model_path="test_model.gguf")
            characters = service.get_available_characters()
            
            assert len(characters) == 8
            character_names = [char.name for char in characters]
            assert "Luke Skywalker" in character_names
            assert "Darth Vader" in character_names
            assert "Princess Leia" in character_names
            assert "Han Solo" in character_names
            assert "Yoda" in character_names
            assert "Obi-Wan Kenobi" in character_names
            assert "Chewbacca" in character_names
            assert "R2-D2" in character_names
    
    def test_get_character_info_valid(self):
        """Test getting character info for valid character."""
        with patch('src.star_wars_rag.llm_service.LocalLLM') as mock_llm_class:
            mock_llm = Mock()
            mock_llm_class.return_value = mock_llm
            
            service = LLMService(model_path="test_model.gguf")
            character = service.get_character_info("Luke Skywalker")
            
            assert character.name == "Luke Skywalker"
            assert "Jedi Knight" in character.description
            assert "Brave" in character.personality
            assert character.voice_style == "young_male"
    
    def test_get_character_info_case_insensitive(self):
        """Test getting character info with case insensitive matching."""
        with patch('src.star_wars_rag.llm_service.LocalLLM') as mock_llm_class:
            mock_llm = Mock()
            mock_llm_class.return_value = mock_llm
            
            service = LLMService(model_path="test_model.gguf")
            character = service.get_character_info("luke skywalker")
            
            assert character.name == "Luke Skywalker"
    
    def test_get_character_info_invalid(self):
        """Test getting character info for invalid character."""
        with patch('src.star_wars_rag.llm_service.LocalLLM') as mock_llm_class:
            mock_llm = Mock()
            mock_llm_class.return_value = mock_llm
            
            service = LLMService(model_path="test_model.gguf")
            
            with pytest.raises(ValueError, match="Character 'Invalid Character' not found"):
                service.get_character_info("Invalid Character")


class TestLLMServiceEndpoints:
    """Test cases for LLM service FastAPI endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "llm"
        assert "model" in data
    
    def test_list_characters(self, client):
        """Test characters listing endpoint."""
        response = client.get("/characters")
        assert response.status_code == 200
        data = response.json()
        assert "characters" in data
        assert "total_count" in data
        assert data["total_count"] == 8
        assert len(data["characters"]) == 8
    
    def test_get_character_info_valid(self, client):
        """Test getting character info for valid character."""
        response = client.get("/characters/Luke%20Skywalker")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Luke Skywalker"
        assert "description" in data
        assert "personality" in data
        assert "voice_style" in data
    
    def test_get_character_info_invalid(self, client):
        """Test getting character info for invalid character."""
        response = client.get("/characters/Invalid%20Character")
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]
    
    def test_get_model_info(self, client):
        """Test model info endpoint."""
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "model_path" in data
        assert "model_name" in data
        assert "context_size" in data
    
    @patch('src.star_wars_rag.llm_service.llm_service.chat_with_character')
    def test_chat_with_character_success(self, mock_chat, client):
        """Test successful character chat endpoint."""
        # Mock chat result
        mock_result = {
            "response": "May the Force be with you, young one.",
            "character": "Luke Skywalker",
            "metadata": {
                "prompt_tokens": 50,
                "completion_tokens": 10,
                "total_tokens": 60,
                "generation_time_seconds": 2.5,
                "tokens_per_second": 4.0,
                "model": "test_model.gguf",
                "stop_reason": "stop",
                "prompt_length": 100,
                "response_length": 30
            }
        }
        mock_chat.return_value = mock_result
        
        request_data = {
            "message": "Hello Luke!",
            "character": "Luke Skywalker",
            "context": "Previous conversation",
            "max_tokens": 150,
            "temperature": 0.8
        }
        
        response = client.post("/chat", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "May the Force be with you, young one."
        assert data["character"] == "Luke Skywalker"
        assert "metadata" in data
        
        # Verify service was called correctly
        mock_chat.assert_called_once_with(
            message="Hello Luke!",
            character="Luke Skywalker",
            context="Previous conversation",
            max_tokens=150,
            temperature=0.8,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1
        )
    
    @patch('src.star_wars_rag.llm_service.llm_service.chat_with_character')
    def test_chat_with_character_validation_error(self, mock_chat, client):
        """Test character chat with validation error."""
        mock_chat.side_effect = ValueError("Character not found")
        
        request_data = {
            "message": "Hello!",
            "character": "Invalid Character"
        }
        
        response = client.post("/chat", json=request_data)
        
        assert response.status_code == 400
        data = response.json()
        assert "Character not found" in data["detail"]
    
    @patch('src.star_wars_rag.llm_service.llm_service.chat_with_character')
    def test_chat_with_character_service_error(self, mock_chat, client):
        """Test character chat with service error."""
        mock_chat.side_effect = RuntimeError("Service error")
        
        request_data = {
            "message": "Hello!",
            "character": "Luke Skywalker"
        }
        
        response = client.post("/chat", json=request_data)
        
        assert response.status_code == 500
        data = response.json()
        assert "Internal server error" in data["detail"]


class TestChatRequest:
    """Test cases for ChatRequest model."""
    
    def test_chat_request_creation(self):
        """Test ChatRequest model creation."""
        request = ChatRequest(
            message="Hello Luke!",
            character="Luke Skywalker",
            context="Previous conversation",
            max_tokens=150,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            repeat_penalty=1.2
        )
        
        assert request.message == "Hello Luke!"
        assert request.character == "Luke Skywalker"
        assert request.context == "Previous conversation"
        assert request.max_tokens == 150
        assert request.temperature == 0.8
        assert request.top_p == 0.95
        assert request.top_k == 50
        assert request.repeat_penalty == 1.2
    
    def test_chat_request_defaults(self):
        """Test ChatRequest with default values."""
        request = ChatRequest(
            message="Hello!",
            character="Luke Skywalker"
        )
        
        assert request.message == "Hello!"
        assert request.character == "Luke Skywalker"
        assert request.context is None
        assert request.max_tokens == 200
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.top_k == 40
        assert request.repeat_penalty == 1.1


class TestChatResponse:
    """Test cases for ChatResponse model."""
    
    def test_chat_response_creation(self):
        """Test ChatResponse model creation."""
        metadata = {
            "prompt_tokens": 50,
            "completion_tokens": 10,
            "total_tokens": 60,
            "generation_time_seconds": 2.5
        }
        
        response = ChatResponse(
            response="May the Force be with you!",
            character="Luke Skywalker",
            metadata=metadata
        )
        
        assert response.response == "May the Force be with you!"
        assert response.character == "Luke Skywalker"
        assert response.metadata == metadata


class TestCharacterInfo:
    """Test cases for CharacterInfo model."""
    
    def test_character_info_creation(self):
        """Test CharacterInfo model creation."""
        character = CharacterInfo(
            name="Luke Skywalker",
            description="Jedi Knight and hero of the Rebellion",
            personality="Brave and idealistic",
            voice_style="young_male"
        )
        
        assert character.name == "Luke Skywalker"
        assert character.description == "Jedi Knight and hero of the Rebellion"
        assert character.personality == "Brave and idealistic"
        assert character.voice_style == "young_male"
    
    def test_character_info_optional_voice_style(self):
        """Test CharacterInfo with optional voice_style."""
        character = CharacterInfo(
            name="Luke Skywalker",
            description="Jedi Knight and hero of the Rebellion",
            personality="Brave and idealistic"
        )
        
        assert character.name == "Luke Skywalker"
        assert character.voice_style is None


if __name__ == "__main__":
    pytest.main([__file__])
