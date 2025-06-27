"""
Repository ingestion module for local and remote repositories.

Handles validation, cloning, and normalization of repository structures.
"""

from src.ingestion.repository import Repository, RepositoryMetadata
from src.ingestion.ingestor import RepositoryIngestor
from src.ingestion.git_handler import GitHandler

__all__ = [
    "Repository",
    "RepositoryMetadata",
    "RepositoryIngestor",
    "GitHandler",
]

