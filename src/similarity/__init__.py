"""
Similarity analysis module for comparing repositories.

Provides structural, semantic, and hybrid similarity computation
with explainable results.
"""

from src.similarity.structural import StructuralAnalyzer, StructuralSimilarity
from src.similarity.semantic import SemanticAnalyzer, SemanticSimilarity
from src.similarity.hybrid import HybridAnalyzer, SimilarityResult
from src.similarity.analyzer import SimilarityStage

__all__ = [
    "StructuralAnalyzer",
    "StructuralSimilarity",
    "SemanticAnalyzer",
    "SemanticSimilarity",
    "HybridAnalyzer",
    "SimilarityResult",
    "SimilarityStage",
]

