"""
Code Similarity Detection with Embedding Models

A lightweight system for detecting code similarity using state-of-the-art embedding models
with local JSON storage.
"""

__version__ = "1.0.0"
__author__ = "CodeSmell Team"
__email__ = "team@codesmell.dev"

from .models.embedder import CodeEmbedder
from .similarity.calculator import SimilarityCalculator
from .utils.json_storage import JSONStorage

__all__ = ["CodeEmbedder", "SimilarityCalculator", "JSONStorage"] 