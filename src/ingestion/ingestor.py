"""
Repository ingestion pipeline stage.

Handles the complete ingestion workflow including validation,
cloning (if remote), and file discovery.
"""

import fnmatch
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import PipelineConfig, IngestionConfig
from src.core.exceptions import IngestionError, RepositoryValidationError
from src.core.pipeline import PipelineStage, PipelineState
from src.ingestion.repository import Repository, RepositoryMetadata, FileInfo
from src.ingestion.git_handler import GitHandler

logger = logging.getLogger(__name__)


class RepositoryIngestor(PipelineStage):
    """
    Pipeline stage for repository ingestion.
    
    Validates repository input, clones remote repositories,
    discovers source files, and produces a canonical repository
    snapshot for downstream processing.
    """

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.ingestion_config = config.ingestion
        self.git_handler = GitHandler(self.ingestion_config)

    @property
    def name(self) -> str:
        return "ingestion"

    def execute(self, state: PipelineState) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Execute repository ingestion.
        
        Args:
            state: Current pipeline state containing source and target repo paths.
            
        Returns:
            Tuple of (output_data, metrics).
        """
        output = {}
        metrics = {
            "source_files_discovered": 0,
            "target_files_discovered": 0,
            "source_total_size_bytes": 0,
            "target_total_size_bytes": 0,
        }

        source_repo = self._ingest_repository(state.source_repo, "source")
        output["source"] = source_repo
        metrics["source_files_discovered"] = len(source_repo.files)
        metrics["source_total_size_bytes"] = source_repo.metadata.total_size_bytes

        if state.target_repo:
            target_repo = self._ingest_repository(state.target_repo, "target")
            output["target"] = target_repo
            metrics["target_files_discovered"] = len(target_repo.files)
            metrics["target_total_size_bytes"] = target_repo.metadata.total_size_bytes

        return output, metrics

    def _ingest_repository(self, source: str, label: str) -> Repository:
        """
        Ingest a single repository.
        
        Args:
            source: Path or URL to the repository.
            label: Label for logging (source/target).
            
        Returns:
            Repository object with discovered files.
        """
        self.logger.info(f"Ingesting {label} repository: {source}")

        repo_path, source_type, is_temp = self._resolve_repository_path(source)

        if not self._validate_directory(repo_path):
            raise RepositoryValidationError(
                str(repo_path),
                "Directory does not exist or is not accessible"
            )

        git_info = self.git_handler.get_repository_info(repo_path)

        repo_name = self._extract_repo_name(source, repo_path)
        metadata = RepositoryMetadata(
            name=repo_name,
            source=source,
            source_type=source_type,
            root_path=repo_path,
            commit_hash=git_info.get("commit_hash"),
            branch=git_info.get("branch"),
        )

        repository = Repository(metadata=metadata)

        discovered_files = self._discover_files(repo_path)
        for file_info in discovered_files:
            repository.add_file(file_info)

        repository.update_language_distribution()

        self.logger.info(
            f"{label.capitalize()} repository ingested: "
            f"{repository.metadata.total_files} files, "
            f"{repository.metadata.total_size_bytes / 1024:.1f} KB"
        )

        return repository

    def _resolve_repository_path(self, source: str) -> Tuple[Path, str, bool]:
        """
        Resolve the repository source to a local path.
        
        Args:
            source: Path or URL to the repository.
            
        Returns:
            Tuple of (resolved_path, source_type, is_temporary).
        """
        if self.git_handler.is_git_url(source):
            work_dir = Path(self.config.work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
            clone_path = self.git_handler.clone_repository(source, work_dir)
            return clone_path, "remote", True

        path = Path(source).resolve()
        if not path.exists():
            raise RepositoryValidationError(str(path), "Path does not exist")

        return path, "local", False

    def _validate_directory(self, path: Path) -> bool:
        """Validate that a path is an accessible directory."""
        return path.exists() and path.is_dir() and os.access(path, os.R_OK)

    def _extract_repo_name(self, source: str, path: Path) -> str:
        """Extract a repository name from source or path."""
        if self.git_handler.is_git_url(source):
            owner, name = self.git_handler.parse_github_url(source)
            if name:
                return name

        return path.name

    def _discover_files(self, repo_path: Path) -> List[FileInfo]:
        """
        Discover all relevant source files in the repository.
        
        Applies ignore patterns and size limits from configuration.
        
        Args:
            repo_path: Path to the repository root.
            
        Returns:
            List of FileInfo objects for discovered files.
        """
        files = []
        ignore_patterns = self.ingestion_config.ignore_patterns

        for root, dirs, filenames in os.walk(repo_path):
            current_path = Path(root)

            dirs[:] = [
                d for d in dirs
                if not self._should_ignore(current_path / d, repo_path, ignore_patterns)
            ]

            for filename in filenames:
                file_path = current_path / filename

                if self._should_ignore(file_path, repo_path, ignore_patterns):
                    continue

                try:
                    if not file_path.is_file():
                        continue

                    stat = file_path.stat()
                    if stat.st_size > self.ingestion_config.max_file_size:
                        self.logger.debug(
                            f"Skipping large file: {file_path} "
                            f"({stat.st_size / 1024:.1f} KB)"
                        )
                        continue

                    if stat.st_size == 0:
                        continue

                    file_info = FileInfo.from_path(file_path, repo_path)
                    files.append(file_info)

                except (OSError, IOError) as e:
                    self.logger.warning(f"Error accessing file {file_path}: {e}")
                    continue

        self.logger.debug(f"Discovered {len(files)} files in {repo_path}")
        return files

    def _should_ignore(
        self, path: Path, repo_root: Path, patterns: List[str]
    ) -> bool:
        """
        Check if a path should be ignored based on patterns.
        
        Args:
            path: Path to check.
            repo_root: Repository root path.
            patterns: List of ignore patterns.
            
        Returns:
            True if the path should be ignored.
        """
        relative = path.relative_to(repo_root)
        name = path.name

        for pattern in patterns:
            if fnmatch.fnmatch(name, pattern):
                return True

            if fnmatch.fnmatch(str(relative), pattern):
                return True

            for part in relative.parts:
                if fnmatch.fnmatch(part, pattern):
                    return True

        return False


def ingest_repository(source: str, config: PipelineConfig = None) -> Repository:
    """
    Convenience function to ingest a single repository.
    
    Args:
        source: Path or URL to the repository.
        config: Optional pipeline configuration.
        
    Returns:
        Ingested Repository object.
    """
    if config is None:
        from src.core.config import Config
        config = Config.get()

    ingestor = RepositoryIngestor(config)

    class MockState:
        source_repo = source
        target_repo = None
        data = {}

    result, _ = ingestor.execute(MockState())
    return result["source"]

