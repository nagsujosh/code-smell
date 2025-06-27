"""
Configuration management for the code similarity detection system.
"""

import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for managing application settings."""
    
    # Model Configuration
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "codebert")
    MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./model_cache")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))
    
    # Application Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "1048576"))  # 1MB in bytes
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))
    
    # Performance Configuration
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
    DEVICE = os.getenv("DEVICE", "auto")  # auto, cpu, cuda, mps
    
    # Storage Configuration
    DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
    OUTPUT_DIR = DATA_DIR / "embeddings"
    RESULTS_DIR = DATA_DIR / "results"
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate that all required configuration is present.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        # For the simplified version, we don't need external database credentials
        return True
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.OUTPUT_DIR,
            cls.RESULTS_DIR,
            Path(cls.MODEL_CACHE_DIR)
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_config(cls, model_name: Optional[str] = None) -> dict:
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model (uses default if None)
            
        Returns:
            Model configuration dictionary
        """
        if model_name is None:
            model_name = cls.DEFAULT_MODEL
        
        return {
            "name": model_name,
            "cache_dir": cls.MODEL_CACHE_DIR,
            "device": cls.DEVICE,
            "batch_size": cls.BATCH_SIZE,
            "max_workers": cls.MAX_WORKERS
        }
    
    @classmethod
    def get_storage_config(cls) -> dict:
        """
        Get storage configuration.
        
        Returns:
            Storage configuration dictionary
        """
        return {
            "data_dir": str(cls.DATA_DIR),
            "output_dir": str(cls.OUTPUT_DIR),
            "results_dir": str(cls.RESULTS_DIR)
        }
    
    @classmethod
    def get_logging_config(cls) -> dict:
        """
        Get logging configuration.
        
        Returns:
            Logging configuration dictionary
        """
        return {
            "level": cls.LOG_LEVEL,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        } 