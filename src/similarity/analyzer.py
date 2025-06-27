"""
Similarity analysis pipeline stage.

Orchestrates the complete similarity analysis between
two repositories.
"""

import logging
from typing import Any, Dict, List, Tuple

from src.core.config import PipelineConfig
from src.core.pipeline import PipelineStage, PipelineState
from src.graph.semantic_graph import SemanticGraph
from src.similarity.hybrid import HybridAnalyzer, SimilarityResult

logger = logging.getLogger(__name__)


class SimilarityStage(PipelineStage):
    """
    Pipeline stage for computing repository similarity.
    
    Requires both source and target repositories to have been
    processed through embedding.
    """

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.analyzer = HybridAnalyzer(config.similarity)

    @property
    def name(self) -> str:
        return "similarity"

    @property
    def dependencies(self) -> List[str]:
        return ["storage"]

    def execute(self, state: PipelineState) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Compute similarity between source and target repositories.
        
        Args:
            state: Pipeline state containing stored graphs.
            
        Returns:
            Tuple of (similarity_result, metrics).
        """
        storage_data = state.data.get("storage", {})

        if "source" not in storage_data:
            raise ValueError("Source repository not found in pipeline state")

        source_graph = storage_data["source"]["graph"]

        if "target" not in storage_data:
            return self._analyze_single_repository(source_graph)

        target_graph = storage_data["target"]["graph"]

        result = self.analyzer.compute_similarity(source_graph, target_graph)

        metrics = {
            "overall_similarity": result.overall_score,
            "structural_similarity": result.structural.score,
            "semantic_similarity": result.semantic.score,
            "matched_pairs_count": len(result.semantic.matched_pairs),
        }

        output = {
            "result": result,
            "source_name": source_graph.name,
            "target_name": target_graph.name,
        }

        return output, metrics

    def _analyze_single_repository(
        self,
        graph: SemanticGraph,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Analyze a single repository without comparison.
        
        Provides structural analysis and internal similarity insights.
        """
        self.logger.info("Analyzing single repository (no target provided)")

        statistics = graph.get_statistics()
        type_distribution = graph.get_type_distribution()

        internal_similarities = self._compute_internal_similarities(graph)

        output = {
            "mode": "single_repository",
            "graph_name": graph.name,
            "statistics": statistics,
            "type_distribution": type_distribution,
            "internal_similarities": internal_similarities,
        }

        metrics = {
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "entities_with_embeddings": len(graph.get_nodes_with_embeddings()),
        }

        return output, metrics

    def _compute_internal_similarities(
        self,
        graph: SemanticGraph,
    ) -> Dict[str, Any]:
        """
        Compute similarity between entities within a single repository.
        
        Useful for detecting code duplication or similar patterns.
        """
        from src.similarity.semantic import SemanticAnalyzer

        analyzer = SemanticAnalyzer(
            similarity_threshold=self.config.similarity.similarity_threshold
        )

        similar_pairs = []

        nodes_with_embeddings = graph.get_nodes_with_embeddings()

        for i, node1 in enumerate(nodes_with_embeddings):
            for node2 in nodes_with_embeddings[i + 1:]:
                if node1.node_type != node2.node_type:
                    continue

                similarity = analyzer.compute_node_similarity(node1, node2)

                if similarity >= self.config.similarity.similarity_threshold:
                    similar_pairs.append({
                        "node1": node1.name,
                        "node2": node2.name,
                        "type": node1.node_type,
                        "similarity": similarity,
                    })

        similar_pairs.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "similar_pairs_count": len(similar_pairs),
            "top_similar_pairs": similar_pairs[:10],
        }


def compute_similarity(
    graph1: SemanticGraph,
    graph2: SemanticGraph,
    config: PipelineConfig = None,
) -> SimilarityResult:
    """
    Convenience function to compute similarity between two graphs.
    
    Args:
        graph1: First semantic graph.
        graph2: Second semantic graph.
        config: Optional configuration.
        
    Returns:
        SimilarityResult.
    """
    if config is None:
        from src.core.config import Config
        config = Config.get()

    analyzer = HybridAnalyzer(config.similarity)
    return analyzer.compute_similarity(graph1, graph2)

