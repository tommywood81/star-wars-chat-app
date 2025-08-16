"""
Model management utilities for downloading and setting up Phi-2 and other LLMs.

This module handles model downloads, conversion, and validation for the Star Wars RAG system.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import requests
from urllib.parse import urlparse
import hashlib

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logger.warning("huggingface_hub not available. Model downloading will be limited.")


class ModelManager:
    """Manage LLM model downloads and setup."""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize model manager.
        
        Args:
            models_dir: Directory to store model files
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            "phi-2": {
                "repo_id": "microsoft/phi-2",
                "gguf_repo": "TheBloke/phi-2-GGUF",  # Community GGUF version
                "filename": "phi-2.Q4_K_M.gguf",
                "size_mb": 1600,
                "context_size": 2048,
                "description": "Microsoft Phi-2 quantized for efficient CPU inference"
            },
            "phi-1_5": {
                "repo_id": "microsoft/phi-1_5", 
                "gguf_repo": "TheBloke/phi-1_5-GGUF",
                "filename": "phi-1_5.Q4_K_M.gguf",
                "size_mb": 900,
                "context_size": 2048,
                "description": "Microsoft Phi-1.5 quantized - smaller alternative"
            },
            "tinyllama": {
                "repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "gguf_repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", 
                "size_mb": 700,
                "context_size": 2048,
                "description": "TinyLlama 1.1B - very fast and lightweight option"
            }
        }
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """List available model configurations.
        
        Returns:
            Dictionary of model configurations
        """
        return self.model_configs.copy()
    
    def list_downloaded_models(self) -> List[Dict[str, Any]]:
        """List models that are already downloaded.
        
        Returns:
            List of downloaded model information
        """
        downloaded = []
        
        for model_file in self.models_dir.glob("*.gguf"):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            
            # Try to match with known configurations
            model_name = "unknown"
            for name, config in self.model_configs.items():
                if config["filename"] in model_file.name:
                    model_name = name
                    break
            
            downloaded.append({
                "name": model_name,
                "filename": model_file.name,
                "path": str(model_file),
                "size_mb": round(size_mb, 1),
                "exists": True
            })
        
        return downloaded
    
    def download_model(self, model_name: str, force: bool = False) -> Path:
        """Download a model by name.
        
        Args:
            model_name: Name of model to download
            force: Re-download even if file exists
            
        Returns:
            Path to downloaded model file
            
        Raises:
            ValueError: If model name is not recognized
            RuntimeError: If download fails
        """
        if model_name not in self.model_configs:
            available = ", ".join(self.model_configs.keys())
            raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
        
        config = self.model_configs[model_name]
        model_path = self.models_dir / config["filename"]
        
        # Check if already exists
        if model_path.exists() and not force:
            logger.info(f"Model {model_name} already exists at {model_path}")
            return model_path
        
        # Download model
        if not HF_HUB_AVAILABLE:
            raise RuntimeError("huggingface_hub required for model downloading")
        
        try:
            logger.info(f"Downloading {model_name} ({config['size_mb']}MB)...")
            
            downloaded_path = hf_hub_download(
                repo_id=config["gguf_repo"],
                filename=config["filename"],
                cache_dir=str(self.models_dir / ".cache"),
                local_dir=str(self.models_dir),
                local_dir_use_symlinks=False
            )
            
            # Ensure it's in the right place
            if Path(downloaded_path) != model_path:
                shutil.move(downloaded_path, model_path)
            
            logger.info(f"Successfully downloaded {model_name} to {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            raise RuntimeError(f"Model download failed: {e}")
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get path to model file if it exists.
        
        Args:
            model_name: Name of model
            
        Returns:
            Path to model file or None if not found
        """
        if model_name not in self.model_configs:
            return None
        
        config = self.model_configs[model_name]
        model_path = self.models_dir / config["filename"]
        
        return model_path if model_path.exists() else None
    
    def validate_model_file(self, model_path: Path) -> bool:
        """Validate that a model file is properly formatted.
        
        Args:
            model_path: Path to model file
            
        Returns:
            True if file appears valid
        """
        if not model_path.exists():
            return False
        
        # Basic checks
        if model_path.stat().st_size < 100 * 1024 * 1024:  # Less than 100MB seems suspicious
            logger.warning(f"Model file {model_path} seems unusually small")
            return False
        
        # Check file extension
        if not model_path.suffix.lower() == ".gguf":
            logger.warning(f"Model file {model_path} doesn't have .gguf extension")
            return False
        
        # Check magic bytes (GGUF files start with "GGUF")
        try:
            with open(model_path, 'rb') as f:
                magic = f.read(4)
                if magic != b'GGUF':
                    logger.warning(f"Model file {model_path} doesn't have GGUF magic bytes")
                    return False
        except Exception as e:
            logger.error(f"Error reading model file {model_path}: {e}")
            return False
        
        return True
    
    def setup_recommended_model(self) -> Path:
        """Download and setup the recommended model for the project.
        
        Returns:
            Path to the recommended model
        """
        # Phi-2 is the recommended model per instructions
        recommended = "phi-2"
        
        logger.info(f"Setting up recommended model: {recommended}")
        
        # Check if already downloaded
        existing_path = self.get_model_path(recommended)
        if existing_path and self.validate_model_file(existing_path):
            logger.info(f"Recommended model already available: {existing_path}")
            return existing_path
        
        # Download the model
        try:
            model_path = self.download_model(recommended)
            
            if self.validate_model_file(model_path):
                logger.info(f"Successfully set up recommended model: {model_path}")
                return model_path
            else:
                raise RuntimeError("Downloaded model failed validation")
                
        except Exception as e:
            logger.error(f"Failed to setup recommended model: {e}")
            
            # Fallback to smaller model
            logger.info("Attempting fallback to smaller model...")
            try:
                fallback_path = self.download_model("tinyllama")
                if self.validate_model_file(fallback_path):
                    logger.info(f"Fallback model ready: {fallback_path}")
                    return fallback_path
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {fallback_error}")
            
            raise RuntimeError(f"Could not setup any model. Original error: {e}")
    
    def get_model_info(self, model_path: Path) -> Dict[str, Any]:
        """Get information about a model file.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Dictionary with model information
        """
        if not model_path.exists():
            return {"exists": False}
        
        # Basic file info
        stat = model_path.stat()
        info = {
            "exists": True,
            "path": str(model_path),
            "filename": model_path.name,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 1),
            "modified": stat.st_mtime,
            "is_valid": self.validate_model_file(model_path)
        }
        
        # Try to match with known config
        for name, config in self.model_configs.items():
            if config["filename"] in model_path.name:
                info.update({
                    "model_name": name,
                    "description": config["description"],
                    "context_size": config["context_size"],
                    "expected_size_mb": config["size_mb"]
                })
                break
        
        return info
    
    def cleanup_cache(self) -> None:
        """Clean up download cache directory."""
        cache_dir = self.models_dir / ".cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            logger.info("Cleaned up model download cache")


def auto_setup_model(models_dir: str = "models") -> Optional[Path]:
    """Automatically setup the best available model.
    
    Args:
        models_dir: Directory for model storage
        
    Returns:
        Path to ready model or None if setup fails
    """
    manager = ModelManager(models_dir)
    
    try:
        # Check for existing models first
        downloaded = manager.list_downloaded_models()
        if downloaded:
            # Use first valid existing model
            for model in downloaded:
                if model["exists"]:
                    model_path = Path(model["path"])
                    if manager.validate_model_file(model_path):
                        logger.info(f"Using existing model: {model_path}")
                        return model_path
        
        # No existing models, download recommended
        logger.info("No existing models found, downloading recommended model...")
        return manager.setup_recommended_model()
        
    except Exception as e:
        logger.error(f"Auto model setup failed: {e}")
        return None
