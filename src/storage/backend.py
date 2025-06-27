"""
Storage backend implementations.

Defines the storage interface and provides concrete implementations
for different storage mechanisms.
"""

import gzip
import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.config import StorageConfig
from src.core.exceptions import StorageError
from src.graph.semantic_graph import SemanticGraph, GraphNode, GraphEdge

logger = logging.getLogger(__name__)


@dataclass
class StorageMetadata:
    """Metadata about a stored graph."""
    graph_name: str
    repository_source: str
    created_at: datetime
    node_count: int
    edge_count: int
    has_embeddings: bool
    embedding_dimension: Optional[int]
    storage_format: str
    file_path: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "graph_name": self.graph_name,
            "repository_source": self.repository_source,
            "created_at": self.created_at.isoformat(),
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "has_embeddings": self.has_embeddings,
            "embedding_dimension": self.embedding_dimension,
            "storage_format": self.storage_format,
            "file_path": self.file_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StorageMetadata":
        """Create from dictionary."""
        return cls(
            graph_name=data["graph_name"],
            repository_source=data["repository_source"],
            created_at=datetime.fromisoformat(data["created_at"]),
            node_count=data["node_count"],
            edge_count=data["edge_count"],
            has_embeddings=data["has_embeddings"],
            embedding_dimension=data.get("embedding_dimension"),
            storage_format=data["storage_format"],
            file_path=data["file_path"],
        )


class StorageBackend(ABC):
    """
    Abstract base class for storage backends.
    
    Defines the interface for storing and retrieving semantic graphs.
    """

    @abstractmethod
    def save(
        self,
        graph: SemanticGraph,
        name: str,
        metadata: Dict[str, Any] = None,
    ) -> StorageMetadata:
        """Save a graph to storage."""
        pass

    @abstractmethod
    def load(self, name: str) -> SemanticGraph:
        """Load a graph from storage."""
        pass

    @abstractmethod
    def delete(self, name: str) -> bool:
        """Delete a graph from storage."""
        pass

    @abstractmethod
    def exists(self, name: str) -> bool:
        """Check if a graph exists in storage."""
        pass

    @abstractmethod
    def list_graphs(self) -> List[StorageMetadata]:
        """List all stored graphs."""
        pass


class JSONStorageBackend(StorageBackend):
    """
    JSON-based storage backend.
    
    Stores graphs as JSON files with optional gzip compression
    and separate numpy arrays for embeddings.
    """

    def __init__(self, config: StorageConfig = None):
        self.config = config or StorageConfig()
        self.storage_dir = Path(self.config.storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._metadata_cache: Dict[str, StorageMetadata] = {}
        self._load_metadata_cache()

    def _load_metadata_cache(self) -> None:
        """Load metadata cache from index file."""
        index_path = self.storage_dir / "index.json"
        if index_path.exists():
            try:
                with open(index_path, "r") as f:
                    data = json.load(f)
                for name, meta_dict in data.items():
                    self._metadata_cache[name] = StorageMetadata.from_dict(meta_dict)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load metadata cache: {e}")

    def _save_metadata_cache(self) -> None:
        """Save metadata cache to index file."""
        index_path = self.storage_dir / "index.json"
        data = {
            name: meta.to_dict()
            for name, meta in self._metadata_cache.items()
        }
        with open(index_path, "w") as f:
            json.dump(data, f, indent=2)

    def save(
        self,
        graph: SemanticGraph,
        name: str,
        metadata: Dict[str, Any] = None,
    ) -> StorageMetadata:
        """
        Save a graph to JSON storage.
        
        Args:
            graph: Graph to save.
            name: Unique name for the graph.
            metadata: Additional metadata.
            
        Returns:
            StorageMetadata for the saved graph.
        """
        metadata = metadata or {}
        graph_dir = self.storage_dir / name
        graph_dir.mkdir(parents=True, exist_ok=True)

        has_embeddings = False
        embedding_dim = None
        embeddings_data = {}

        for node in graph.iter_nodes():
            if node.embedding is not None:
                has_embeddings = True
                embedding_dim = len(node.embedding)
                embeddings_data[node.id] = node.embedding.tolist()

        graph_data = graph.to_dict(include_embeddings=False)
        graph_data["metadata"] = metadata

        graph_file = graph_dir / "graph.json"
        if self.config.enable_compression:
            graph_file = graph_file.with_suffix(".json.gz")
            with gzip.open(graph_file, "wt", encoding="utf-8") as f:
                json.dump(graph_data, f, indent=2)
        else:
            with open(graph_file, "w") as f:
                json.dump(graph_data, f, indent=2)

        if embeddings_data:
            embeddings_file = graph_dir / "embeddings.json"
            if self.config.enable_compression:
                embeddings_file = embeddings_file.with_suffix(".json.gz")
                with gzip.open(embeddings_file, "wt", encoding="utf-8") as f:
                    json.dump(embeddings_data, f)
            else:
                with open(embeddings_file, "w") as f:
                    json.dump(embeddings_data, f)

        storage_meta = StorageMetadata(
            graph_name=name,
            repository_source=metadata.get("source", ""),
            created_at=datetime.now(),
            node_count=graph.node_count,
            edge_count=graph.edge_count,
            has_embeddings=has_embeddings,
            embedding_dimension=embedding_dim,
            storage_format="json.gz" if self.config.enable_compression else "json",
            file_path=str(graph_dir),
        )

        self._metadata_cache[name] = storage_meta
        self._save_metadata_cache()

        logger.info(f"Saved graph '{name}' to {graph_dir}")
        return storage_meta

    def load(self, name: str) -> SemanticGraph:
        """
        Load a graph from JSON storage.
        
        Args:
            name: Name of the graph to load.
            
        Returns:
            Loaded SemanticGraph.
            
        Raises:
            StorageError: If graph not found.
        """
        graph_dir = self.storage_dir / name
        if not graph_dir.exists():
            raise StorageError(f"Graph not found: {name}")

        graph_file = graph_dir / "graph.json.gz"
        if not graph_file.exists():
            graph_file = graph_dir / "graph.json"

        if not graph_file.exists():
            raise StorageError(f"Graph file not found: {graph_file}")

        if graph_file.suffix == ".gz":
            with gzip.open(graph_file, "rt", encoding="utf-8") as f:
                graph_data = json.load(f)
        else:
            with open(graph_file, "r") as f:
                graph_data = json.load(f)

        graph = SemanticGraph(name=graph_data.get("name", name))

        for node_data in graph_data.get("nodes", []):
            node = GraphNode(
                id=node_data["id"],
                node_type=node_data["node_type"],
                name=node_data["name"],
                qualified_name=node_data["qualified_name"],
                language=node_data["language"],
                attributes=node_data.get("attributes", {}),
                code_snippet=node_data.get("code_snippet"),
            )
            graph.add_node(node)

        for edge_data in graph_data.get("edges", []):
            edge = GraphEdge(
                source_id=edge_data["source_id"],
                target_id=edge_data["target_id"],
                edge_type=edge_data["edge_type"],
                attributes=edge_data.get("attributes", {}),
            )
            graph.add_edge(edge)

        embeddings_file = graph_dir / "embeddings.json.gz"
        if not embeddings_file.exists():
            embeddings_file = graph_dir / "embeddings.json"

        if embeddings_file.exists():
            if embeddings_file.suffix == ".gz":
                with gzip.open(embeddings_file, "rt", encoding="utf-8") as f:
                    embeddings_data = json.load(f)
            else:
                with open(embeddings_file, "r") as f:
                    embeddings_data = json.load(f)

            for node_id, embedding_list in embeddings_data.items():
                node = graph.get_node(node_id)
                if node:
                    node.embedding = np.array(embedding_list)

        logger.info(f"Loaded graph '{name}' from {graph_dir}")
        return graph

    def delete(self, name: str) -> bool:
        """
        Delete a graph from storage.
        
        Args:
            name: Name of the graph to delete.
            
        Returns:
            True if deleted, False if not found.
        """
        graph_dir = self.storage_dir / name
        if not graph_dir.exists():
            return False

        try:
            shutil.rmtree(graph_dir)
            self._metadata_cache.pop(name, None)
            self._save_metadata_cache()
            logger.info(f"Deleted graph '{name}'")
            return True
        except OSError as e:
            logger.error(f"Failed to delete graph '{name}': {e}")
            return False

    def exists(self, name: str) -> bool:
        """Check if a graph exists in storage."""
        graph_dir = self.storage_dir / name
        return graph_dir.exists()

    def list_graphs(self) -> List[StorageMetadata]:
        """List all stored graphs."""
        return list(self._metadata_cache.values())

    def get_metadata(self, name: str) -> Optional[StorageMetadata]:
        """Get metadata for a specific graph."""
        return self._metadata_cache.get(name)

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_size = 0
        for path in self.storage_dir.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size

        return {
            "graph_count": len(self._metadata_cache),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "storage_dir": str(self.storage_dir),
        }

