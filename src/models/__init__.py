"""
Embedding models for code similarity detection.
"""

from src.models.model_registry import ModelRegistry

__all__ = ["ModelRegistry"]

# Import embedder only when needed to avoid heavy dependencies
def get_embedder():
    """Get the CodeEmbedder class when needed."""
    from src.models.embedder import CodeEmbedder
    return CodeEmbedder 