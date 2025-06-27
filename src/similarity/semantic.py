"""
Semantic similarity analysis using embeddings.

Computes similarity based on vector embeddings of code entities
with hierarchical aggregation.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.graph.semantic_graph import SemanticGraph, GraphNode

logger = logging.getLogger(__name__)


@dataclass
class SemanticSimilarity:
    """Result of semantic similarity analysis."""

    score: float
    function_similarity: float
    class_similarity: float
    file_similarity: float
    matched_pairs: List[Dict[str, Any]] = field(default_factory=list)
    aggregation_method: str = "weighted_average"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "function_similarity": self.function_similarity,
            "class_similarity": self.class_similarity,
            "file_similarity": self.file_similarity,
            "matched_pairs": self.matched_pairs[:10],  # Top 10 matches
            "aggregation_method": self.aggregation_method,
            "details": self.details,
        }


class SemanticAnalyzer:
    """
    Analyzes semantic similarity between graphs using embeddings.
    
    Computes similarity by comparing embeddings at different
    granularity levels (function, class, file) and aggregating
    hierarchically.
    """

    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self._weights = {
            "function": 0.50,
            "class": 0.30,
            "file": 0.20,
        }

    def compute_similarity(
        self,
        graph1: SemanticGraph,
        graph2: SemanticGraph,
    ) -> SemanticSimilarity:
        """
        Compute semantic similarity between two graphs.
        
        Args:
            graph1: First semantic graph with embeddings.
            graph2: Second semantic graph with embeddings.
            
        Returns:
            SemanticSimilarity result.
        """
        func_sim, func_pairs = self._compute_level_similarity(
            graph1, graph2, "function"
        )

        class_sim, class_pairs = self._compute_level_similarity(
            graph1, graph2, "class"
        )

        file_sim, file_pairs = self._compute_level_similarity(
            graph1, graph2, "file"
        )

        overall_score = (
            self._weights["function"] * func_sim +
            self._weights["class"] * class_sim +
            self._weights["file"] * file_sim
        )

        all_pairs = func_pairs + class_pairs + file_pairs
        all_pairs.sort(key=lambda x: x["similarity"], reverse=True)

        details = {
            "function_matches": len(func_pairs),
            "class_matches": len(class_pairs),
            "file_matches": len(file_pairs),
            "total_functions_1": len(graph1.get_nodes_by_type("function")),
            "total_functions_2": len(graph2.get_nodes_by_type("function")),
            "total_classes_1": len(graph1.get_nodes_by_type("class")),
            "total_classes_2": len(graph2.get_nodes_by_type("class")),
        }

        return SemanticSimilarity(
            score=overall_score,
            function_similarity=func_sim,
            class_similarity=class_sim,
            file_similarity=file_sim,
            matched_pairs=all_pairs,
            details=details,
        )

    def _compute_level_similarity(
        self,
        graph1: SemanticGraph,
        graph2: SemanticGraph,
        node_type: str,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Compute similarity at a specific node type level.
        
        Uses optimal matching to find best pairings between
        nodes of the same type.
        """
        nodes1 = self._get_nodes_with_embeddings(graph1, node_type)
        nodes2 = self._get_nodes_with_embeddings(graph2, node_type)

        if not nodes1 or not nodes2:
            return 0.0, []

        similarity_matrix = self._compute_similarity_matrix(nodes1, nodes2)

        matched_pairs = self._find_best_matches(
            nodes1, nodes2, similarity_matrix
        )

        if matched_pairs:
            avg_similarity = np.mean([p["similarity"] for p in matched_pairs])
        else:
            avg_similarity = 0.0

        coverage = len(matched_pairs) / max(len(nodes1), len(nodes2))
        adjusted_similarity = avg_similarity * min(1.0, coverage + 0.5)

        return adjusted_similarity, matched_pairs

    def _get_nodes_with_embeddings(
        self,
        graph: SemanticGraph,
        node_type: str,
    ) -> List[GraphNode]:
        """Get nodes of a type that have embeddings."""
        nodes = graph.get_nodes_by_type(node_type)
        return [n for n in nodes if n.embedding is not None]

    def _compute_similarity_matrix(
        self,
        nodes1: List[GraphNode],
        nodes2: List[GraphNode],
    ) -> np.ndarray:
        """Compute pairwise cosine similarity matrix."""
        embeddings1 = np.array([n.embedding for n in nodes1])
        embeddings2 = np.array([n.embedding for n in nodes2])

        norms1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norms2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)

        norms1 = np.where(norms1 == 0, 1, norms1)
        norms2 = np.where(norms2 == 0, 1, norms2)

        embeddings1_normalized = embeddings1 / norms1
        embeddings2_normalized = embeddings2 / norms2

        return np.dot(embeddings1_normalized, embeddings2_normalized.T)

    def _find_best_matches(
        self,
        nodes1: List[GraphNode],
        nodes2: List[GraphNode],
        similarity_matrix: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Find best matching pairs using greedy matching.
        
        Matches nodes greedily by highest similarity score,
        ensuring each node is matched at most once.
        """
        matched_pairs = []
        used1 = set()
        used2 = set()

        flat_indices = np.argsort(similarity_matrix.flatten())[::-1]

        for flat_idx in flat_indices:
            i = flat_idx // similarity_matrix.shape[1]
            j = flat_idx % similarity_matrix.shape[1]

            if i in used1 or j in used2:
                continue

            similarity = similarity_matrix[i, j]

            if similarity < self.similarity_threshold:
                break

            matched_pairs.append({
                "node1_id": nodes1[i].id,
                "node1_name": nodes1[i].name,
                "node1_type": nodes1[i].node_type,
                "node2_id": nodes2[j].id,
                "node2_name": nodes2[j].name,
                "node2_type": nodes2[j].node_type,
                "similarity": float(similarity),
            })

            used1.add(i)
            used2.add(j)

        return matched_pairs

    def compute_node_similarity(
        self,
        node1: GraphNode,
        node2: GraphNode,
    ) -> float:
        """
        Compute similarity between two specific nodes.
        
        Args:
            node1: First node.
            node2: Second node.
            
        Returns:
            Cosine similarity score.
        """
        if node1.embedding is None or node2.embedding is None:
            return 0.0

        emb1 = node1.embedding
        emb2 = node2.embedding

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def find_similar_nodes(
        self,
        node: GraphNode,
        graph: SemanticGraph,
        top_k: int = 5,
    ) -> List[Tuple[GraphNode, float]]:
        """
        Find the most similar nodes in a graph to a given node.
        
        Args:
            node: Query node.
            graph: Graph to search.
            top_k: Number of results to return.
            
        Returns:
            List of (node, similarity) tuples.
        """
        if node.embedding is None:
            return []

        candidates = []
        for candidate in graph.get_nodes_with_embeddings():
            if candidate.id == node.id:
                continue

            similarity = self.compute_node_similarity(node, candidate)
            candidates.append((candidate, similarity))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

