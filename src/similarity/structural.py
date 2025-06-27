"""
Structural similarity analysis for semantic graphs.

Computes similarity based on graph topology, node type distributions,
and dependency structures.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from src.graph.semantic_graph import SemanticGraph

logger = logging.getLogger(__name__)


@dataclass
class StructuralSimilarity:
    """Result of structural similarity analysis."""

    score: float
    node_type_similarity: float
    edge_type_similarity: float
    dependency_overlap: float
    topology_similarity: float
    size_ratio: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "node_type_similarity": self.node_type_similarity,
            "edge_type_similarity": self.edge_type_similarity,
            "dependency_overlap": self.dependency_overlap,
            "topology_similarity": self.topology_similarity,
            "size_ratio": self.size_ratio,
            "details": self.details,
        }


class StructuralAnalyzer:
    """
    Analyzes structural similarity between semantic graphs.
    
    Computes similarity based on:
    - Node type distributions
    - Edge type distributions
    - Dependency overlap
    - Graph topology metrics
    """

    def __init__(self):
        self._weights = {
            "node_type": 0.25,
            "edge_type": 0.20,
            "dependency": 0.30,
            "topology": 0.25,
        }

    def compute_similarity(
        self,
        graph1: SemanticGraph,
        graph2: SemanticGraph,
    ) -> StructuralSimilarity:
        """
        Compute structural similarity between two graphs.
        
        Args:
            graph1: First semantic graph.
            graph2: Second semantic graph.
            
        Returns:
            StructuralSimilarity result.
        """
        node_type_sim = self._compute_node_type_similarity(graph1, graph2)

        edge_type_sim = self._compute_edge_type_similarity(graph1, graph2)

        dependency_overlap = self._compute_dependency_overlap(graph1, graph2)

        topology_sim = self._compute_topology_similarity(graph1, graph2)

        size_ratio = self._compute_size_ratio(graph1, graph2)

        overall_score = (
            self._weights["node_type"] * node_type_sim +
            self._weights["edge_type"] * edge_type_sim +
            self._weights["dependency"] * dependency_overlap +
            self._weights["topology"] * topology_sim
        )

        details = self._collect_details(graph1, graph2)

        return StructuralSimilarity(
            score=overall_score,
            node_type_similarity=node_type_sim,
            edge_type_similarity=edge_type_sim,
            dependency_overlap=dependency_overlap,
            topology_similarity=topology_sim,
            size_ratio=size_ratio,
            details=details,
        )

    def _compute_node_type_similarity(
        self,
        graph1: SemanticGraph,
        graph2: SemanticGraph,
    ) -> float:
        """Compute similarity based on node type distribution."""
        dist1 = graph1.get_type_distribution()["nodes"]
        dist2 = graph2.get_type_distribution()["nodes"]

        return self._cosine_similarity_dicts(dist1, dist2)

    def _compute_edge_type_similarity(
        self,
        graph1: SemanticGraph,
        graph2: SemanticGraph,
    ) -> float:
        """Compute similarity based on edge type distribution."""
        dist1 = graph1.get_type_distribution()["edges"]
        dist2 = graph2.get_type_distribution()["edges"]

        return self._cosine_similarity_dicts(dist1, dist2)

    def _compute_dependency_overlap(
        self,
        graph1: SemanticGraph,
        graph2: SemanticGraph,
    ) -> float:
        """Compute overlap in external dependencies."""
        deps1 = self._extract_dependencies(graph1)
        deps2 = self._extract_dependencies(graph2)

        if not deps1 and not deps2:
            return 1.0
        if not deps1 or not deps2:
            return 0.0

        intersection = deps1 & deps2
        union = deps1 | deps2

        return len(intersection) / len(union) if union else 0.0

    def _extract_dependencies(self, graph: SemanticGraph) -> Set[str]:
        """Extract external dependency names from a graph."""
        deps = set()
        for node in graph.get_nodes_by_type("external_dependency"):
            name = node.attributes.get("module_path", node.name)
            root_module = name.split(".")[0]
            deps.add(root_module)
        return deps

    def _compute_topology_similarity(
        self,
        graph1: SemanticGraph,
        graph2: SemanticGraph,
    ) -> float:
        """Compute similarity based on graph topology metrics."""
        metrics1 = self._compute_topology_metrics(graph1)
        metrics2 = self._compute_topology_metrics(graph2)

        similarities = []

        for key in metrics1:
            if key in metrics2:
                v1, v2 = metrics1[key], metrics2[key]
                if v1 == 0 and v2 == 0:
                    sim = 1.0
                else:
                    sim = 1.0 - abs(v1 - v2) / max(abs(v1), abs(v2), 1e-9)
                similarities.append(max(0.0, sim))

        return np.mean(similarities) if similarities else 0.0

    def _compute_topology_metrics(self, graph: SemanticGraph) -> Dict[str, float]:
        """Compute topology metrics for a graph."""
        g = graph.get_networkx_graph()

        if graph.node_count == 0:
            return {
                "density": 0.0,
                "avg_degree": 0.0,
                "avg_clustering": 0.0,
                "depth_ratio": 0.0,
            }

        try:
            density = nx.density(g)
        except Exception:
            density = 0.0

        try:
            avg_degree = sum(d for _, d in g.degree()) / max(graph.node_count, 1)
        except Exception:
            avg_degree = 0.0

        try:
            avg_clustering = nx.average_clustering(g.to_undirected())
        except Exception:
            avg_clustering = 0.0

        depth = self._compute_max_depth(graph)
        depth_ratio = depth / max(graph.node_count, 1)

        return {
            "density": density,
            "avg_degree": avg_degree,
            "avg_clustering": avg_clustering,
            "depth_ratio": depth_ratio,
        }

    def _compute_max_depth(self, graph: SemanticGraph) -> int:
        """Compute maximum depth from root nodes."""
        g = graph.get_networkx_graph()

        roots = [n for n in g.nodes() if g.in_degree(n) == 0]
        if not roots:
            roots = list(g.nodes())[:1]

        max_depth = 0
        for root in roots:
            try:
                lengths = nx.single_source_shortest_path_length(g, root)
                if lengths:
                    max_depth = max(max_depth, max(lengths.values()))
            except nx.NetworkXError:
                continue

        return max_depth

    def _compute_size_ratio(
        self,
        graph1: SemanticGraph,
        graph2: SemanticGraph,
    ) -> float:
        """Compute size ratio between graphs."""
        size1 = graph1.node_count
        size2 = graph2.node_count

        if size1 == 0 and size2 == 0:
            return 1.0
        if size1 == 0 or size2 == 0:
            return 0.0

        return min(size1, size2) / max(size1, size2)

    def _cosine_similarity_dicts(
        self,
        dict1: Dict[str, int],
        dict2: Dict[str, int],
    ) -> float:
        """Compute cosine similarity between two count dictionaries."""
        all_keys = set(dict1.keys()) | set(dict2.keys())
        if not all_keys:
            return 1.0

        vec1 = np.array([dict1.get(k, 0) for k in all_keys])
        vec2 = np.array([dict2.get(k, 0) for k in all_keys])

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 and norm2 == 0:
            return 1.0
        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _collect_details(
        self,
        graph1: SemanticGraph,
        graph2: SemanticGraph,
    ) -> Dict[str, Any]:
        """Collect detailed comparison information."""
        return {
            "graph1_stats": graph1.get_statistics(),
            "graph2_stats": graph2.get_statistics(),
            "common_node_types": list(
                graph1.get_node_types() & graph2.get_node_types()
            ),
            "common_edge_types": list(
                graph1.get_edge_types() & graph2.get_edge_types()
            ),
            "common_dependencies": list(
                self._extract_dependencies(graph1) &
                self._extract_dependencies(graph2)
            ),
        }

