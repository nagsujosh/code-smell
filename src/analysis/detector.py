"""
Language detection for source files.

Provides language detection based on file extensions with
confidence scoring and support for multi-language repositories.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.core.config import AnalysisConfig
from src.ingestion.repository import Repository, FileInfo

logger = logging.getLogger(__name__)


@dataclass
class LanguageDetectionResult:
    """Result of language detection for a file."""
    language: str
    confidence: float
    method: str  # "extension", "shebang", "content"


@dataclass
class RepositoryLanguageProfile:
    """Language profile for an entire repository."""
    primary_language: str
    language_distribution: Dict[str, int]
    language_percentages: Dict[str, float]
    file_mappings: Dict[str, str]  # file_path -> language


class LanguageDetector:
    """
    Detects programming languages from source files.
    
    Uses file extensions as the primary detection method with
    fallback to shebang lines and content analysis.
    """

    EXTENSION_MAP: Dict[str, str] = {
        ".py": "python",
        ".pyw": "python",
        ".pyi": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".mts": "typescript",
        ".cts": "typescript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".hpp": "cpp",
        ".hxx": "cpp",
        ".cs": "csharp",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".scala": "scala",
        ".sc": "scala",
        ".vue": "vue",
        ".svelte": "svelte",
        ".elm": "elm",
        ".ex": "elixir",
        ".exs": "elixir",
        ".erl": "erlang",
        ".hrl": "erlang",
        ".hs": "haskell",
        ".lhs": "haskell",
        ".ml": "ocaml",
        ".mli": "ocaml",
        ".clj": "clojure",
        ".cljs": "clojure",
        ".cljc": "clojure",
        ".r": "r",
        ".R": "r",
        ".jl": "julia",
        ".lua": "lua",
        ".pl": "perl",
        ".pm": "perl",
        ".sh": "shell",
        ".bash": "shell",
        ".zsh": "shell",
        ".fish": "shell",
        ".sql": "sql",
        ".m": "objective-c",
        ".mm": "objective-c",
        ".dart": "dart",
        ".groovy": "groovy",
        ".gradle": "groovy",
    }

    SHEBANG_MAP: Dict[str, str] = {
        "python": "python",
        "python3": "python",
        "python2": "python",
        "node": "javascript",
        "nodejs": "javascript",
        "ruby": "ruby",
        "perl": "perl",
        "bash": "shell",
        "sh": "shell",
        "zsh": "shell",
        "php": "php",
        "lua": "lua",
    }

    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
        self.extension_map = {
            **self.EXTENSION_MAP,
            **self.config.language_extensions,
        }

    def detect_file_language(
        self, file_path: Path, content: str = None
    ) -> LanguageDetectionResult:
        """
        Detect the programming language of a file.
        
        Args:
            file_path: Path to the file.
            content: Optional file content for shebang detection.
            
        Returns:
            LanguageDetectionResult with detected language and confidence.
        """
        extension = file_path.suffix.lower()
        if extension in self.extension_map:
            return LanguageDetectionResult(
                language=self.extension_map[extension],
                confidence=1.0,
                method="extension",
            )

        if content:
            shebang_result = self._detect_from_shebang(content)
            if shebang_result:
                return shebang_result

        return LanguageDetectionResult(
            language="unknown",
            confidence=0.0,
            method="none",
        )

    def _detect_from_shebang(
        self, content: str
    ) -> Optional[LanguageDetectionResult]:
        """Detect language from shebang line."""
        lines = content.split("\n", 1)
        if not lines:
            return None

        first_line = lines[0].strip()
        if not first_line.startswith("#!"):
            return None

        shebang = first_line[2:].strip()

        for pattern, language in self.SHEBANG_MAP.items():
            if pattern in shebang:
                return LanguageDetectionResult(
                    language=language,
                    confidence=0.9,
                    method="shebang",
                )

        return None

    def detect_repository_languages(
        self, repository: Repository
    ) -> RepositoryLanguageProfile:
        """
        Detect languages for all files in a repository.
        
        Args:
            repository: Repository to analyze.
            
        Returns:
            RepositoryLanguageProfile with complete language analysis.
        """
        file_mappings: Dict[str, str] = {}
        language_counts: Dict[str, int] = {}

        for file_info in repository.files:
            result = self.detect_file_language(file_info.path)

            if result.confidence >= self.config.language_confidence_threshold:
                file_mappings[file_info.relative_path] = result.language
                file_info.language = result.language

                language_counts[result.language] = (
                    language_counts.get(result.language, 0) + 1
                )

        total_files = len(file_mappings) or 1
        language_percentages = {
            lang: (count / total_files) * 100
            for lang, count in language_counts.items()
        }

        primary_language = max(
            language_counts.items(),
            key=lambda x: x[1],
            default=("unknown", 0)
        )[0]

        repository.update_language_distribution()

        logger.info(
            f"Detected {len(language_counts)} languages in repository, "
            f"primary: {primary_language}"
        )

        return RepositoryLanguageProfile(
            primary_language=primary_language,
            language_distribution=language_counts,
            language_percentages=language_percentages,
            file_mappings=file_mappings,
        )

    def get_supported_languages(self) -> List[str]:
        """Get list of all supported languages."""
        return sorted(set(self.extension_map.values()))

    def get_extensions_for_language(self, language: str) -> List[str]:
        """Get all file extensions associated with a language."""
        return [
            ext for ext, lang in self.extension_map.items()
            if lang == language
        ]

