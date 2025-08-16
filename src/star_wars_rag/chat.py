"""
Enhanced chat system integrating RAG retrieval with local LLM generation.

This module combines the existing RAG system with local LLM for full character chat.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Iterator
import time

from star_wars_rag.app import StarWarsRAGApp
from star_wars_rag.llm import create_llm, LocalLLM, MockLLM
from star_wars_rag.prompt import StarWarsPromptBuilder, SafetyFilter
from star_wars_rag.models import ModelManager, auto_setup_model

logger = logging.getLogger(__name__)


class StarWarsChatApp(StarWarsRAGApp):
    """Enhanced Star Wars RAG app with local LLM chat capabilities."""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_model_path: Optional[Union[str, Path]] = None,
                 models_dir: str = "models",
                 auto_download: bool = True):
        """Initialize the chat application.
        
        Args:
            embedding_model: Embedding model name for RAG
            llm_model_path: Path to LLM model file (auto-setup if None)
            models_dir: Directory for model storage
            auto_download: Automatically download model if not found
        """
        # Initialize base RAG system
        super().__init__(embedding_model)
        
        # Initialize chat components
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.model_manager = ModelManager(str(self.models_dir))
        self.prompt_builder = StarWarsPromptBuilder()
        self.safety_filter = SafetyFilter()
        
        # Initialize LLM
        self.llm: Optional[Union[LocalLLM, MockLLM]] = None
        self.llm_model_path: Optional[Path] = None
        
        if llm_model_path:
            self._load_llm(Path(llm_model_path))
        elif auto_download:
            self._auto_setup_llm()
        
        logger.info("StarWarsChatApp initialized")
    
    def _auto_setup_llm(self) -> None:
        """Automatically setup the best available LLM."""
        try:
            logger.info("Auto-setting up LLM...")
            model_path = auto_setup_model(str(self.models_dir))
            
            if model_path:
                self._load_llm(model_path)
            else:
                logger.warning("Could not auto-setup LLM, using MockLLM")
                self.llm = create_llm("mock_model.gguf")
                
        except Exception as e:
            logger.error(f"LLM auto-setup failed: {e}")
            logger.info("Using MockLLM as fallback")
            self.llm = create_llm("mock_model.gguf")
    
    def _load_llm(self, model_path: Path) -> None:
        """Load LLM from model path.
        
        Args:
            model_path: Path to model file
        """
        try:
            logger.info(f"Loading LLM from {model_path}")
            self.llm = create_llm(model_path, n_ctx=2048, n_threads=2)
            self.llm_model_path = model_path
            
            # Validate model
            if hasattr(self.llm, 'validate_model') and not self.llm.validate_model():
                raise RuntimeError("Model validation failed")
                
            logger.info("LLM loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            logger.info("Using MockLLM as fallback")
            self.llm = create_llm("mock_model.gguf")
    
    def chat_with_character(self, 
                           user_message: str, 
                           character: str,
                           max_context_lines: int = 6,
                           conversation_history: Optional[List[Dict[str, str]]] = None,
                           temperature: float = 0.7,
                           max_tokens: int = 150) -> Dict[str, Any]:
        """Enhanced character chat using RAG + LLM.
        
        Args:
            user_message: User's input message
            character: Character to chat with
            max_context_lines: Maximum retrieved context lines
            conversation_history: Previous conversation turns
            temperature: LLM sampling temperature
            max_tokens: Maximum response tokens
            
        Returns:
            Dictionary with character response and metadata
            
        Raises:
            RuntimeError: If system is not loaded or LLM not available
        """
        if not self.is_loaded:
            raise RuntimeError("RAG system not loaded. Call load_from_scripts() or load_from_processed_data() first.")
        
        if self.llm is None:
            raise RuntimeError("LLM not available. Check model setup.")
        
        start_time = time.time()
        
        try:
            # Step 1: Retrieve relevant context using RAG
            logger.debug(f"Retrieving context for character '{character}' and query '{user_message[:50]}...'")
            
            retrieved_context = self.search_dialogue(
                user_message,
                top_k=max_context_lines * 2,  # Get more, then filter
                character_filter=character
            )
            
            # Step 2: Build character-specific prompt
            prompt = self.prompt_builder.build_character_prompt(
                character=character,
                user_message=user_message,
                retrieved_context=retrieved_context[:max_context_lines],
                conversation_history=conversation_history,
                max_context_lines=max_context_lines
            )
            
            # Step 3: Apply safety filtering to prompt
            prompt = self.safety_filter.filter_prompt(prompt)
            
            # Step 4: Generate response with LLM
            logger.debug(f"Generating response with LLM (temp={temperature}, max_tokens={max_tokens})")
            
            llm_response = self.llm.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                stop=["\n", "User:", "Human:", f"{character}:", "Character:"]
            )
            
            # Step 5: Extract and filter response
            response_text = llm_response['response']
            response_text = self.safety_filter.filter_response(response_text, character)
            
            # Step 6: Build result
            total_time = time.time() - start_time
            
            result = {
                'character': character,
                'response': response_text,
                'user_message': user_message,
                'context_used': retrieved_context[:max_context_lines],
                'conversation_metadata': {
                    'retrieval_results': len(retrieved_context),
                    'context_lines_used': min(len(retrieved_context), max_context_lines),
                    'prompt_length': len(prompt),
                    'total_time_seconds': round(total_time, 3)
                },
                'llm_metadata': llm_response.get('metadata', {}),
                'model_info': {
                    'embedding_model': self.embedder.model_name,
                    'llm_model': getattr(self.llm, 'model_path', 'mock_model') if hasattr(self.llm, 'model_path') else 'mock'
                }
            }
            
            logger.debug(f"Chat completed in {total_time:.3f}s - {character}: {response_text[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Chat generation failed: {e}")
            
            # Fallback response
            return {
                'character': character,
                'response': f"I'm having trouble responding right now. Perhaps we could talk about something else?",
                'user_message': user_message,
                'context_used': [],
                'error': str(e),
                'conversation_metadata': {
                    'retrieval_results': 0,
                    'context_lines_used': 0,
                    'total_time_seconds': time.time() - start_time
                },
                'llm_metadata': {},
                'model_info': {}
            }
    
    def stream_chat_with_character(self,
                                  user_message: str,
                                  character: str,
                                  max_context_lines: int = 6,
                                  conversation_history: Optional[List[Dict[str, str]]] = None,
                                  temperature: float = 0.7,
                                  max_tokens: int = 150) -> Iterator[Dict[str, Any]]:
        """Stream character chat responses token by token.
        
        Args:
            user_message: User's input message
            character: Character to chat with
            max_context_lines: Maximum retrieved context lines
            conversation_history: Previous conversation turns
            temperature: LLM sampling temperature
            max_tokens: Maximum response tokens
            
        Yields:
            Dictionary with streaming response data
        """
        if not self.is_loaded:
            raise RuntimeError("RAG system not loaded.")
        
        if self.llm is None or not hasattr(self.llm, 'stream_generate'):
            # Fallback to non-streaming
            result = self.chat_with_character(
                user_message, character, max_context_lines, 
                conversation_history, temperature, max_tokens
            )
            yield {
                'token': result['response'],
                'is_complete': True,
                'metadata': result.get('llm_metadata', {})
            }
            return
        
        try:
            # Get context and build prompt (same as regular chat)
            retrieved_context = self.search_dialogue(
                user_message,
                top_k=max_context_lines * 2,
                character_filter=character
            )
            
            prompt = self.prompt_builder.build_character_prompt(
                character=character,
                user_message=user_message,
                retrieved_context=retrieved_context[:max_context_lines],
                conversation_history=conversation_history,
                max_context_lines=max_context_lines
            )
            
            prompt = self.safety_filter.filter_prompt(prompt)
            
            # Stream response
            accumulated_response = ""
            for token_data in self.llm.stream_generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                stop=["\n", "User:", "Human:", f"{character}:", "Character:"]
            ):
                token = token_data['token']
                accumulated_response += token
                
                # Apply safety filtering to accumulated response
                filtered_response = self.safety_filter.filter_response(accumulated_response, character)
                
                yield {
                    'token': token,
                    'accumulated_response': filtered_response,
                    'is_complete': token_data['is_complete'],
                    'character': character,
                    'metadata': token_data.get('metadata', {})
                }
                
        except Exception as e:
            logger.error(f"Streaming chat failed: {e}")
            yield {
                'token': f"I'm having trouble responding right now.",
                'accumulated_response': f"I'm having trouble responding right now.",
                'is_complete': True,
                'character': character,
                'error': str(e),
                'metadata': {}
            }
    
    def get_llm_info(self) -> Dict[str, Any]:
        """Get information about the loaded LLM.
        
        Returns:
            Dictionary with LLM information
        """
        if self.llm is None:
            return {"llm_loaded": False}
        
        info = {"llm_loaded": True}
        
        if hasattr(self.llm, 'get_model_info'):
            info.update(self.llm.get_model_info())
        
        if self.llm_model_path:
            model_info = self.model_manager.get_model_info(self.llm_model_path)
            info['model_file_info'] = model_info
        
        return info
    
    def list_available_models(self) -> Dict[str, Any]:
        """List available and downloaded models.
        
        Returns:
            Dictionary with model information
        """
        return {
            'available_configs': self.model_manager.list_available_models(),
            'downloaded_models': self.model_manager.list_downloaded_models(),
            'current_model': self.get_llm_info()
        }
    
    def download_model(self, model_name: str, force: bool = False) -> bool:
        """Download a specific model.
        
        Args:
            model_name: Name of model to download
            force: Re-download even if exists
            
        Returns:
            True if successful
        """
        try:
            model_path = self.model_manager.download_model(model_name, force)
            logger.info(f"Model {model_name} downloaded to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            return False
    
    def switch_model(self, model_path: Union[str, Path]) -> bool:
        """Switch to a different LLM model.
        
        Args:
            model_path: Path to new model file
            
        Returns:
            True if successful
        """
        try:
            # Unload current model
            if self.llm and hasattr(self.llm, 'unload_model'):
                self.llm.unload_model()
            
            # Load new model
            self._load_llm(Path(model_path))
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            return False
    
    def get_available_characters(self) -> List[str]:
        """Get list of available characters from the loaded data.
        
        Returns:
            List of character names
        """
        if not self.is_loaded or self.retriever is None:
            return []
        
        return self.retriever.get_available_characters()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics.
        
        Returns:
            Dictionary with system stats
        """
        if not self.is_loaded:
            return {"error": "System not loaded"}
        
        return super().get_system_stats()
    
    def test_chat_quality(self, test_queries: Optional[List[str]] = None) -> Dict[str, Any]:
        """Test chat quality with sample queries.
        
        Args:
            test_queries: Custom test queries (uses defaults if None)
            
        Returns:
            Dictionary with test results
        """
        if test_queries is None:
            test_queries = [
                "Tell me about the Force",
                "What do you think about the Empire?",
                "Can you help me with something?",
                "I need guidance on my path"
            ]
        
        characters = self.get_available_characters()[:3]  # Test with top 3 characters
        if not characters:
            return {"error": "No characters available for testing"}
        
        results = {}
        total_tests = 0
        successful_tests = 0
        total_time = 0
        
        for character in characters:
            char_results = []
            
            for query in test_queries:
                try:
                    start_time = time.time()
                    response = self.chat_with_character(query, character, max_tokens=100)
                    test_time = time.time() - start_time
                    
                    char_results.append({
                        'query': query,
                        'response': response['response'][:100] + "..." if len(response['response']) > 100 else response['response'],
                        'success': True,
                        'time_seconds': round(test_time, 3),
                        'context_lines': response['conversation_metadata']['context_lines_used']
                    })
                    
                    successful_tests += 1
                    total_time += test_time
                    
                except Exception as e:
                    char_results.append({
                        'query': query,
                        'response': None,
                        'success': False,
                        'error': str(e),
                        'time_seconds': 0
                    })
                
                total_tests += 1
            
            results[character] = char_results
        
        return {
            'test_summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
                'average_time_seconds': total_time / successful_tests if successful_tests > 0 else 0
            },
            'character_results': results,
            'llm_info': self.get_llm_info()
        }
    
    def retrieve_similar_dialogue(self, query: str, character_filter: Optional[str] = None, 
                                 top_k: int = 5, min_similarity: float = 0.0) -> List[Dict]:
        """Retrieve similar dialogue - wrapper for search_dialogue method.
        
        This method provides compatibility for the web interface.
        
        Args:
            query: Search query
            character_filter: Filter by specific character
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of relevant dialogue results
        """
        return self.search_dialogue(
            query=query,
            character_filter=character_filter,
            top_k=top_k,
            min_similarity=min_similarity
        )
