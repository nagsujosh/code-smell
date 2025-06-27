"""
Analyzer registry for language-specific analyzers.

Provides a plugin-based architecture where language analyzers
can be registered and retrieved dynamically.
"""

import logging
from typing import Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class BaseLanguageAnalyzer:
    """
    Abstract base class for language-specific analyzers.
    
    Each language plugin must implement this interface to provide
    static analysis capabilities for that language.
    """

    LANGUAGE: str = "unknown"
    SUPPORTED_EXTENSIONS: List[str] = []

    def analyze_file(self, file_path: str, content: str) -> "AnalysisResult":
        """
        Analyze a single source file.
        
        Args:
            file_path: Path to the file being analyzed.
            content: Source code content.
            
        Returns:
            AnalysisResult containing extracted entities and relationships.
        """
        raise NotImplementedError("Subclasses must implement analyze_file")

    def extract_entities(self, file_path: str, content: str) -> List:
        """Extract code entities from source."""
        raise NotImplementedError("Subclasses must implement extract_entities")

    def extract_relationships(self, file_path: str, content: str, entities: List) -> List:
        """Extract relationships between entities."""
        raise NotImplementedError("Subclasses must implement extract_relationships")


class AnalyzerRegistry:
    """
    Central registry for language analyzers.
    
    Manages the registration and retrieval of language-specific
    analyzer plugins.
    """

    _analyzers: Dict[str, Type[BaseLanguageAnalyzer]] = {}
    _instances: Dict[str, BaseLanguageAnalyzer] = {}

    @classmethod
    def register(cls, analyzer_class: Type[BaseLanguageAnalyzer]) -> Type[BaseLanguageAnalyzer]:
        """
        Register a language analyzer.
        
        Can be used as a decorator:
            @AnalyzerRegistry.register
            class PythonAnalyzer(BaseLanguageAnalyzer):
                ...
        
        Args:
            analyzer_class: The analyzer class to register.
            
        Returns:
            The registered class (for decorator usage).
        """
        language = analyzer_class.LANGUAGE
        if language in cls._analyzers:
            logger.warning(
                f"Overwriting existing analyzer for {language}: "
                f"{cls._analyzers[language].__name__} -> {analyzer_class.__name__}"
            )

        cls._analyzers[language] = analyzer_class
        logger.debug(f"Registered analyzer for {language}: {analyzer_class.__name__}")
        return analyzer_class

    @classmethod
    def get_analyzer(cls, language: str) -> Optional[BaseLanguageAnalyzer]:
        """
        Get an analyzer instance for a language.
        
        Lazily instantiates analyzers on first request.
        
        Args:
            language: Language identifier.
            
        Returns:
            Analyzer instance or None if not available.
        """
        if language not in cls._analyzers:
            return None

        if language not in cls._instances:
            cls._instances[language] = cls._analyzers[language]()

        return cls._instances[language]

    @classmethod
    def has_analyzer(cls, language: str) -> bool:
        """Check if an analyzer exists for a language."""
        return language in cls._analyzers

    @classmethod
    def list_languages(cls) -> List[str]:
        """List all languages with registered analyzers."""
        return list(cls._analyzers.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered analyzers (mainly for testing)."""
        cls._analyzers.clear()
        cls._instances.clear()

    @classmethod
    def get_analyzer_class(cls, language: str) -> Optional[Type[BaseLanguageAnalyzer]]:
        """Get the analyzer class (not instance) for a language."""
        return cls._analyzers.get(language)

