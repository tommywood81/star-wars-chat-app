"""
Local LLM module for Star Wars RAG chat using llama.cpp and Phi-2.

This module provides a wrapper for running quantized LLMs locally for character chat.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Iterator, Union
import json
import time

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not available. LLM functionality will be disabled.")


class LocalLLM:
    """Local LLM wrapper for Star Wars character chat using llama.cpp."""
    
    def __init__(self, 
                 model_path: Union[str, Path],
                 n_ctx: int = 2048,
                 n_threads: Optional[int] = None,
                 n_gpu_layers: int = 0,
                 verbose: bool = False):
        """Initialize the local LLM.
        
        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context window size (tokens)
            n_threads: Number of CPU threads (auto-detect if None)
            n_gpu_layers: Number of layers to offload to GPU (0 for CPU-only)
            verbose: Enable verbose logging from llama.cpp
            
        Raises:
            ImportError: If llama-cpp-python is not installed
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is required for LLM functionality. "
                "Install with: pip install llama-cpp-python"
            )
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        
        # Initialize model
        self.llm: Optional[Llama] = None
        self._load_model()
        
        logger.info(f"LocalLLM initialized with model: {self.model_path.name}")
    
    def _load_model(self) -> None:
        """Load the GGUF model using llama.cpp."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            start_time = time.time()
            
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose
            )
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def generate(self, 
                prompt: str, 
                max_tokens: int = 200,
                temperature: float = 0.7,
                top_p: float = 0.9,
                top_k: int = 40,
                repeat_penalty: float = 1.1,
                stop: Optional[list] = None) -> Dict[str, Any]:
        """Generate a response from the prompt.
        
        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling limit
            repeat_penalty: Repetition penalty factor
            stop: List of stop sequences
            
        Returns:
            Dictionary with response and metadata
            
        Raises:
            RuntimeError: If model is not loaded
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded")
        
        if stop is None:
            stop = ["\n", "User:", "Human:"]
        
        try:
            start_time = time.time()
            
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stop=stop,
                echo=False
            )
            
            generation_time = time.time() - start_time
            
            # Extract response text
            response_text = response['choices'][0]['text'].strip()
            
            # Build result
            result = {
                'response': response_text,
                'metadata': {
                    'prompt_tokens': response['usage']['prompt_tokens'],
                    'completion_tokens': response['usage']['completion_tokens'],
                    'total_tokens': response['usage']['total_tokens'],
                    'generation_time_seconds': generation_time,
                    'tokens_per_second': response['usage']['completion_tokens'] / generation_time if generation_time > 0 else 0,
                    'model': self.model_path.name,
                    'stop_reason': response['choices'][0].get('finish_reason', 'unknown')
                }
            }
            
            logger.debug(f"Generated {response['usage']['completion_tokens']} tokens in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Text generation failed: {e}")
    
    def stream_generate(self, 
                       prompt: str,
                       max_tokens: int = 200,
                       temperature: float = 0.7,
                       top_p: float = 0.9,
                       top_k: int = 40,
                       repeat_penalty: float = 1.1,
                       stop: Optional[list] = None) -> Iterator[Dict[str, Any]]:
        """Generate response with streaming (token by token).
        
        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling limit
            repeat_penalty: Repetition penalty factor
            stop: List of stop sequences
            
        Yields:
            Dictionary with token and metadata for each generated token
            
        Raises:
            RuntimeError: If model is not loaded
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded")
        
        if stop is None:
            stop = ["\n", "User:", "Human:"]
        
        try:
            start_time = time.time()
            token_count = 0
            
            for output in self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stop=stop,
                stream=True,
                echo=False
            ):
                token_count += 1
                current_time = time.time()
                
                yield {
                    'token': output['choices'][0]['text'],
                    'is_complete': output['choices'][0].get('finish_reason') is not None,
                    'metadata': {
                        'tokens_generated': token_count,
                        'elapsed_time': current_time - start_time,
                        'tokens_per_second': token_count / (current_time - start_time) if current_time > start_time else 0
                    }
                }
                
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise RuntimeError(f"Streaming generation failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_path': str(self.model_path),
            'model_name': self.model_path.name,
            'model_size_mb': self.model_path.stat().st_size / (1024 * 1024) if self.model_path.exists() else 0,
            'context_size': self.n_ctx,
            'threads': self.n_threads,
            'gpu_layers': self.n_gpu_layers,
            'is_loaded': self.llm is not None
        }
    
    def validate_model(self) -> bool:
        """Validate that the model is working correctly.
        
        Returns:
            True if model validation passes
        """
        try:
            test_prompt = "Hello"
            response = self.generate(test_prompt, max_tokens=5, temperature=0.1)
            return len(response['response']) > 0
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.llm is not None:
            del self.llm
            self.llm = None
            logger.info("Model unloaded")


class MockLLM:
    """Mock LLM for testing when llama-cpp-python is not available."""
    
    def __init__(self, *args, **kwargs):
        """Initialize mock LLM."""
        self.model_path = Path("mock_model.gguf")
        logger.warning("Using MockLLM - llama-cpp-python not available")
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate mock response."""
        # Simple mock responses based on prompt content
        if "Force" in prompt:
            response = "The Force is what gives a Jedi his power."
        elif "Dark Side" in prompt or "Vader" in prompt:
            response = "You underestimate the power of the Dark Side."
        elif "help" in prompt.lower():
            response = "Help you I can, but first, patience you must learn."
        else:
            response = "Mock response from local LLM."
        
        return {
            'response': response,
            'metadata': {
                'prompt_tokens': len(prompt.split()),
                'completion_tokens': len(response.split()),
                'total_tokens': len(prompt.split()) + len(response.split()),
                'generation_time_seconds': 0.1,
                'tokens_per_second': 50.0,
                'model': 'mock_model',
                'stop_reason': 'length'
            }
        }
    
    def stream_generate(self, prompt: str, **kwargs) -> Iterator[Dict[str, Any]]:
        """Generate mock streaming response."""
        response = self.generate(prompt, **kwargs)
        words = response['response'].split()
        
        for i, word in enumerate(words):
            yield {
                'token': word + (" " if i < len(words) - 1 else ""),
                'is_complete': i == len(words) - 1,
                'metadata': {
                    'tokens_generated': i + 1,
                    'elapsed_time': (i + 1) * 0.1,
                    'tokens_per_second': (i + 1) / ((i + 1) * 0.1)
                }
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model info."""
        return {
            'model_path': 'mock_model.gguf',
            'model_name': 'mock_model.gguf',
            'model_size_mb': 0,
            'context_size': 2048,
            'threads': 1,
            'gpu_layers': 0,
            'is_loaded': True
        }
    
    def validate_model(self) -> bool:
        """Mock validation always passes."""
        return True
    
    def unload_model(self) -> None:
        """Mock unload."""
        pass


def create_llm(model_path: Union[str, Path], **kwargs) -> Union[LocalLLM, MockLLM]:
    """Factory function to create appropriate LLM instance.
    
    Args:
        model_path: Path to model file
        **kwargs: Additional arguments for LLM initialization
        
    Returns:
        LocalLLM if llama-cpp-python is available and model exists, MockLLM otherwise
    """
    if not LLAMA_CPP_AVAILABLE:
        logger.warning("llama-cpp-python not available, using MockLLM")
        return MockLLM(model_path, **kwargs)
    
    model_path = Path(model_path)
    if not model_path.exists():
        logger.warning(f"Model file not found: {model_path}, using MockLLM")
        return MockLLM(model_path, **kwargs)
    
    return LocalLLM(model_path, **kwargs)
