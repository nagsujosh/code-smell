"""
Model registry for CodeBERT embedding model.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class ModelType(Enum):
    """Types of embedding models."""
    BERT = "bert"


@dataclass
class ModelConfig:
    """Configuration for an embedding model."""
    name: str
    model_id: str
    type: ModelType
    languages: List[str]
    embedding_dim: int
    max_length: int
    description: str
    size_mb: Optional[int] = None
    requires_gpu: bool = False
    pooling_strategy: str = "mean"
    special_tokens: Optional[List[str]] = None


class ModelRegistry:
    """Registry of CodeBERT embedding model."""
    
    MODELS = {
        "codebert": ModelConfig(
            name="CodeBERT",
            model_id="microsoft/codebert-base",
            type=ModelType.BERT,
            languages=["python", "java", "javascript", "php", "ruby", "go"],
            embedding_dim=768,
            max_length=512,
            description="Robust BERT-based model by Microsoft for code embedding",
            size_mb=110,
            pooling_strategy="mean"
        )
    }
    
    @classmethod
    def get_model(cls, model_name: str) -> ModelConfig:
        """Get model configuration by name."""
        if model_name not in cls.MODELS:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(cls.MODELS.keys())}")
        return cls.MODELS[model_name]
    
    @classmethod
    def list_models(cls) -> Dict[str, ModelConfig]:
        """List all available models."""
        return cls.MODELS.copy()
    
    @classmethod
    def get_default_model(cls) -> str:
        """Get the default model name."""
        return "codebert" 