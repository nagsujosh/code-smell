"""
Static analyzer orchestration.

Coordinates language detection and delegates to appropriate
language-specific analyzers for static analysis.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import PipelineConfig
from src.core.exceptions import AnalysisError, LanguageNotSupportedError
from src.core.pipeline import PipelineStage, PipelineState
from src.ingestion.repository import Repository
from src.analysis.detector import LanguageDetector, RepositoryLanguageProfile
from src.analysis.registry import AnalyzerRegistry, BaseLanguageAnalyzer
from src.analysis.entities import CodeEntity, Relationship

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result of static analysis for a single file or repository."""

    entities: List[CodeEntity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def merge(self, other: "AnalysisResult") -> None:
        """Merge another analysis result into this one."""
        self.entities.extend(other.entities)
        self.relationships.extend(other.relationships)
        self.errors.extend(other.errors)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "errors": self.errors,
            "metrics": self.metrics,
        }


class StaticAnalyzer(PipelineStage):
    """
    Pipeline stage for static analysis.
    
    Detects languages and delegates to appropriate language-specific
    analyzers for entity and relationship extraction.
    """

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.detector = LanguageDetector(config.analysis)
        self._load_language_plugins()

    @property
    def name(self) -> str:
        return "analysis"

    @property
    def dependencies(self) -> List[str]:
        return ["ingestion"]

    def _load_language_plugins(self) -> None:
        """Load all available language analyzer plugins (tree-sitter based)."""
        from src.analysis.languages import (
            python_analyzer,
            javascript_analyzer,
            java_analyzer,
            go_analyzer,
            rust_analyzer,
            cpp_analyzer,
            ruby_analyzer,
            php_analyzer,
        )
        logger.info(
            f"Loaded tree-sitter analyzers for: {AnalyzerRegistry.list_languages()}"
        )

    def execute(self, state: PipelineState) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Execute static analysis on ingested repositories.
        
        Args:
            state: Pipeline state containing ingested repositories.
            
        Returns:
            Tuple of (analysis_results, metrics).
        """
        ingestion_data = state.data.get("ingestion", {})
        output = {}
        metrics = {
            "source_entities": 0,
            "source_relationships": 0,
            "target_entities": 0,
            "target_relationships": 0,
            "languages_analyzed": [],
            "files_analyzed": 0,
            "files_skipped": 0,
        }

        if "source" in ingestion_data:
            source_repo = ingestion_data["source"]
            source_profile = self.detector.detect_repository_languages(source_repo)
            source_result = self._analyze_repository(source_repo, source_profile)

            output["source"] = {
                "result": source_result,
                "profile": source_profile,
            }
            metrics["source_entities"] = len(source_result.entities)
            metrics["source_relationships"] = len(source_result.relationships)

        if "target" in ingestion_data:
            target_repo = ingestion_data["target"]
            target_profile = self.detector.detect_repository_languages(target_repo)
            target_result = self._analyze_repository(target_repo, target_profile)

            output["target"] = {
                "result": target_result,
                "profile": target_profile,
            }
            metrics["target_entities"] = len(target_result.entities)
            metrics["target_relationships"] = len(target_result.relationships)

        all_languages = set()
        for key in ["source", "target"]:
            if key in output:
                all_languages.update(
                    output[key]["profile"].language_distribution.keys()
                )
        metrics["languages_analyzed"] = list(all_languages)

        return output, metrics

    def _analyze_repository(
        self,
        repository: Repository,
        profile: RepositoryLanguageProfile
    ) -> AnalysisResult:
        """
        Analyze all files in a repository.
        
        Args:
            repository: Repository to analyze.
            profile: Language profile for the repository.
            
        Returns:
            Combined AnalysisResult for all files.
        """
        combined_result = AnalysisResult()

        files_analyzed = 0
        files_skipped = 0

        for file_info in repository.iter_source_files():
            language = profile.file_mappings.get(file_info.relative_path)

            if not language or language == "unknown":
                files_skipped += 1
                continue

            analyzer = AnalyzerRegistry.get_analyzer(language)
            if not analyzer:
                self.logger.debug(
                    f"No analyzer for {language}, skipping {file_info.relative_path}"
                )
                files_skipped += 1
                continue

            try:
                content = self._read_file_content(file_info.path)
                if not content:
                    continue

                file_result = analyzer.analyze_file(
                    str(file_info.relative_path),
                    content
                )
                combined_result.merge(file_result)
                files_analyzed += 1

            except Exception as e:
                self.logger.warning(
                    f"Error analyzing {file_info.relative_path}: {e}"
                )
                combined_result.errors.append({
                    "file": file_info.relative_path,
                    "error": str(e),
                    "type": type(e).__name__,
                })

        combined_result.metrics = {
            "files_analyzed": files_analyzed,
            "files_skipped": files_skipped,
            "total_entities": len(combined_result.entities),
            "total_relationships": len(combined_result.relationships),
        }

        self.logger.info(
            f"Analyzed {files_analyzed} files, "
            f"extracted {len(combined_result.entities)} entities, "
            f"{len(combined_result.relationships)} relationships"
        )

        return combined_result

    def _read_file_content(self, file_path: Path) -> Optional[str]:
        """Safely read file content."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            max_lines = self.config.analysis.max_lines_per_file
            if max_lines > 0:
                lines = content.split("\n")
                if len(lines) > max_lines:
                    self.logger.debug(
                        f"Truncating {file_path}: {len(lines)} -> {max_lines} lines"
                    )
                    content = "\n".join(lines[:max_lines])

            return content

        except (IOError, OSError) as e:
            self.logger.warning(f"Failed to read {file_path}: {e}")
            return None

    def analyze_single_file(
        self,
        file_path: str,
        content: str,
        language: str = None
    ) -> AnalysisResult:
        """
        Analyze a single file directly.
        
        Args:
            file_path: Path to the file.
            content: File content.
            language: Language (auto-detected if not provided).
            
        Returns:
            AnalysisResult for the file.
        """
        if not language:
            result = self.detector.detect_file_language(Path(file_path), content)
            language = result.language

        analyzer = AnalyzerRegistry.get_analyzer(language)
        if not analyzer:
            raise LanguageNotSupportedError(language)

        return analyzer.analyze_file(file_path, content)

