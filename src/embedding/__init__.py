"""
Semantic embedding generation for code entities.

Provides functionality for extracting semantic payloads from code
and generating vector embeddings using pre-trained models.
"""

from src.embedding.extractor import PayloadExtractor, SemanticPayload
from src.embedding.encoder import CodeEncoder
from src.embedding.embedder import EmbeddingStage

__all__ = [
    "PayloadExtractor",
    "SemanticPayload",
    "CodeEncoder",
    "EmbeddingStage",
]

