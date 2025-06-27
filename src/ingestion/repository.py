"""
Repository data structures and metadata management.

Provides canonical representations of repository snapshots with
associated metadata for downstream processing.
"""

import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class FileInfo:
    """Information about a single file in the repository."""

    path: Path
    relative_path: str
    size_bytes: int
    extension: str
    language: Optional[str] = None
    content_hash: Optional[str] = None

    @classmethod
    def from_path(cls, path: Path, repo_root: Path) -> "FileInfo":
        """Create FileInfo from a file path."""
        relative = path.relative_to(repo_root)
        stat = path.stat()

        return cls(
            path=path,
            relative_path=str(relative),
            size_bytes=stat.st_size,
            extension=path.suffix.lower(),
        )

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of file content."""
        if self.content_hash is not None:
            return self.content_hash

        sha256 = hashlib.sha256()
        try:
            with open(self.path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            self.content_hash = sha256.hexdigest()
        except (IOError, OSError):
            self.content_hash = ""

        return self.content_hash

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "path": str(self.path),
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "extension": self.extension,
            "language": self.language,
            "content_hash": self.content_hash,
        }


@dataclass
class RepositoryMetadata:
    """Metadata about a repository snapshot."""

    name: str
    source: str
    source_type: str  # "local" or "remote"
    root_path: Path
    snapshot_time: datetime = field(default_factory=datetime.now)
    commit_hash: Optional[str] = None
    branch: Optional[str] = None
    total_files: int = 0
    total_size_bytes: int = 0
    language_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "source": self.source,
            "source_type": self.source_type,
            "root_path": str(self.root_path),
            "snapshot_time": self.snapshot_time.isoformat(),
            "commit_hash": self.commit_hash,
            "branch": self.branch,
            "total_files": self.total_files,
            "total_size_bytes": self.total_size_bytes,
            "language_distribution": self.language_distribution,
        }


@dataclass
class Repository:
    """
    Canonical representation of a repository snapshot.
    
    Contains all files and metadata needed for analysis.
    """

    metadata: RepositoryMetadata
    files: List[FileInfo] = field(default_factory=list)
    _file_index: Dict[str, FileInfo] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Build file index after initialization."""
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild the internal file index."""
        self._file_index = {f.relative_path: f for f in self.files}

    def add_file(self, file_info: FileInfo) -> None:
        """Add a file to the repository."""
        self.files.append(file_info)
        self._file_index[file_info.relative_path] = file_info
        self.metadata.total_files = len(self.files)
        self.metadata.total_size_bytes += file_info.size_bytes

    def get_file(self, relative_path: str) -> Optional[FileInfo]:
        """Get a file by its relative path."""
        return self._file_index.get(relative_path)

    def get_files_by_extension(self, extension: str) -> List[FileInfo]:
        """Get all files with a specific extension."""
        ext = extension if extension.startswith(".") else f".{extension}"
        return [f for f in self.files if f.extension == ext]

    def get_files_by_language(self, language: str) -> List[FileInfo]:
        """Get all files for a specific language."""
        return [f for f in self.files if f.language == language]

    def get_extensions(self) -> Set[str]:
        """Get all unique file extensions in the repository."""
        return {f.extension for f in self.files if f.extension}

    def get_languages(self) -> Set[str]:
        """Get all detected languages in the repository."""
        return {f.language for f in self.files if f.language}

    def update_language_distribution(self) -> None:
        """Update the language distribution in metadata."""
        distribution: Dict[str, int] = {}
        for f in self.files:
            if f.language:
                distribution[f.language] = distribution.get(f.language, 0) + 1
        self.metadata.language_distribution = distribution

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "metadata": self.metadata.to_dict(),
            "files": [f.to_dict() for f in self.files],
        }

    def get_file_tree(self) -> Dict:
        """
        Build a hierarchical file tree representation.
        
        Returns:
            Nested dictionary representing the directory structure.
        """
        tree: Dict = {}

        for file_info in self.files:
            parts = Path(file_info.relative_path).parts
            current = tree

            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    current[part] = file_info
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

        return tree

    def iter_source_files(self) -> List[FileInfo]:
        """
        Iterate over source code files only.
        
        Excludes binary files and non-source extensions.
        """
        source_extensions = {
            ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go",
            ".rs", ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp",
            ".cs", ".rb", ".php", ".swift", ".kt", ".scala",
            ".vue", ".svelte", ".elm", ".ex", ".exs", ".erl",
            ".hs", ".ml", ".mli", ".clj", ".cljs", ".r", ".jl",
        }
        return [f for f in self.files if f.extension in source_extensions]

