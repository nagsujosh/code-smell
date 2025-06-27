"""
Storage manager for coordinating graph persistence.

Provides a high-level interface for storing and managing
semantic graphs with support for versioning and caching.
"""

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import PipelineConfig, StorageConfig
from src.core.pipeline import PipelineStage, PipelineState
from src.core.exceptions import StorageError
from src.graph.semantic_graph import SemanticGraph
from src.storage.backend import StorageBackend, JSONStorageBackend, StorageMetadata

logger = logging.getLogger(__name__)


class StorageManager(PipelineStage):
    """
    Pipeline stage for graph storage management.
    
    Coordinates saving and loading of semantic graphs with
    support for caching and metadata management.
    """

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.backend = JSONStorageBackend(config.storage)
        self._graph_cache: Dict[str, SemanticGraph] = {}

    @property
    def name(self) -> str:
        return "storage"

    @property
    def dependencies(self) -> List[str]:
        return ["embedding"]

    def execute(self, state: PipelineState) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Store graphs from the embedding stage.
        
        Args:
            state: Pipeline state containing embedded graphs.
            
        Returns:
            Tuple of (storage_results, metrics).
        """
        embedding_data = state.data.get("embedding", {})
        ingestion_data = state.data.get("ingestion", {})

        output = {}
        metrics = {
            "graphs_stored": 0,
            "total_nodes": 0,
            "total_edges": 0,
        }

        if "source" in embedding_data:
            source_graph = embedding_data["source"]["graph"]
            source_repo = ingestion_data.get("source")
            source_name = self._generate_graph_name(
                source_repo.metadata.source if source_repo else "source"
            )

            source_meta = self.store_graph(
                source_graph,
                source_name,
                {
                    "source": source_repo.metadata.source if source_repo else "",
                    "type": "source",
                    "pipeline_id": state.pipeline_id,
                }
            )

            output["source"] = {
                "name": source_name,
                "metadata": source_meta,
                "graph": source_graph,
            }
            metrics["graphs_stored"] += 1
            metrics["total_nodes"] += source_graph.node_count
            metrics["total_edges"] += source_graph.edge_count

        if "target" in embedding_data:
            target_graph = embedding_data["target"]["graph"]
            target_repo = ingestion_data.get("target")
            target_name = self._generate_graph_name(
                target_repo.metadata.source if target_repo else "target"
            )

            target_meta = self.store_graph(
                target_graph,
                target_name,
                {
                    "source": target_repo.metadata.source if target_repo else "",
                    "type": "target",
                    "pipeline_id": state.pipeline_id,
                }
            )

            output["target"] = {
                "name": target_name,
                "metadata": target_meta,
                "graph": target_graph,
            }
            metrics["graphs_stored"] += 1
            metrics["total_nodes"] += target_graph.node_count
            metrics["total_edges"] += target_graph.edge_count

        return output, metrics

    def _generate_graph_name(self, source: str) -> str:
        """Generate a unique name for a graph based on source."""
        hash_input = f"{source}:{datetime.now().isoformat()}"
        hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:8]

        source_name = source.split("/")[-1] if "/" in source else source
        source_name = source_name.replace(".git", "").replace(" ", "_")[:20]

        return f"{source_name}_{hash_value}"

    def store_graph(
        self,
        graph: SemanticGraph,
        name: str = None,
        metadata: Dict[str, Any] = None,
    ) -> StorageMetadata:
        """
        Store a graph with optional metadata.
        
        Args:
            graph: Graph to store.
            name: Optional name (auto-generated if not provided).
            metadata: Additional metadata to store.
            
        Returns:
            StorageMetadata for the stored graph.
        """
        if name is None:
            name = self._generate_graph_name(graph.name or "graph")

        storage_meta = self.backend.save(graph, name, metadata)

        self._graph_cache[name] = graph

        return storage_meta

    def load_graph(self, name: str, use_cache: bool = True) -> SemanticGraph:
        """
        Load a graph by name.
        
        Args:
            name: Name of the graph to load.
            use_cache: Whether to use cached version if available.
            
        Returns:
            Loaded SemanticGraph.
        """
        if use_cache and name in self._graph_cache:
            return self._graph_cache[name]

        graph = self.backend.load(name)
        self._graph_cache[name] = graph

        return graph

    def delete_graph(self, name: str) -> bool:
        """
        Delete a graph from storage.
        
        Args:
            name: Name of the graph to delete.
            
        Returns:
            True if deleted successfully.
        """
        self._graph_cache.pop(name, None)
        return self.backend.delete(name)

    def list_graphs(self) -> List[StorageMetadata]:
        """List all stored graphs."""
        return self.backend.list_graphs()

    def graph_exists(self, name: str) -> bool:
        """Check if a graph exists."""
        return self.backend.exists(name)

    def get_graph_metadata(self, name: str) -> Optional[StorageMetadata]:
        """Get metadata for a specific graph."""
        return self.backend.get_metadata(name)

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return self.backend.get_storage_stats()

    def clear_cache(self) -> None:
        """Clear the in-memory graph cache."""
        self._graph_cache.clear()

    def compare_graphs(
        self,
        name1: str,
        name2: str,
    ) -> Dict[str, Any]:
        """
        Load two graphs for comparison.
        
        Args:
            name1: Name of first graph.
            name2: Name of second graph.
            
        Returns:
            Dictionary with both graphs loaded.
        """
        graph1 = self.load_graph(name1)
        graph2 = self.load_graph(name2)

        return {
            "graph1": {
                "name": name1,
                "graph": graph1,
                "node_count": graph1.node_count,
                "edge_count": graph1.edge_count,
            },
            "graph2": {
                "name": name2,
                "graph": graph2,
                "node_count": graph2.node_count,
                "edge_count": graph2.edge_count,
            },
        }

    def cleanup_old_graphs(self, days: int = 30) -> int:
        """
        Remove graphs older than specified days.
        
        Args:
            days: Age threshold in days.
            
        Returns:
            Number of graphs deleted.
        """
        from datetime import timedelta

        threshold = datetime.now() - timedelta(days=days)
        deleted_count = 0

        for meta in self.list_graphs():
            if meta.created_at < threshold:
                if self.delete_graph(meta.graph_name):
                    deleted_count += 1
                    logger.info(f"Deleted old graph: {meta.graph_name}")

        return deleted_count

