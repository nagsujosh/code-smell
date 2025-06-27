"""
Static analysis module with language detection and pluggable analyzers.

Provides language-agnostic analysis capabilities through a plugin-based
architecture where each language has its own analyzer implementation.
"""

from src.analysis.detector import LanguageDetector
from src.analysis.analyzer import StaticAnalyzer, AnalysisResult
from src.analysis.registry import AnalyzerRegistry
from src.analysis.entities import (
    CodeEntity,
    FileEntity,
    ModuleEntity,
    ClassEntity,
    FunctionEntity,
    ImportEntity,
    Relationship,
    RelationshipType,
)

__all__ = [
    "LanguageDetector",
    "StaticAnalyzer",
    "AnalysisResult",
    "AnalyzerRegistry",
    "CodeEntity",
    "FileEntity",
    "ModuleEntity",
    "ClassEntity",
    "FunctionEntity",
    "ImportEntity",
    "Relationship",
    "RelationshipType",
]

