"""
Hybrid similarity analysis combining structural and semantic approaches.

Provides weighted combination of similarity signals with
configurable weights and detailed breakdown.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.config import SimilarityConfig
from src.graph.semantic_graph import SemanticGraph
from src.similarity.structural import StructuralAnalyzer, StructuralSimilarity
from src.similarity.semantic import SemanticAnalyzer, SemanticSimilarity

logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """
    Complete similarity analysis result.
    
    Contains overall score, component scores, and detailed
    breakdown for explainability.
    """

    overall_score: float
    structural: StructuralSimilarity
    semantic: SemanticSimilarity
    structural_weight: float
    semantic_weight: float
    computed_at: datetime = field(default_factory=datetime.now)
    source_name: str = ""
    target_name: str = ""
    explanation: str = ""
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "structural_score": self.structural.score,
            "semantic_score": self.semantic.score,
            "structural_weight": self.structural_weight,
            "semantic_weight": self.semantic_weight,
            "computed_at": self.computed_at.isoformat(),
            "source_name": self.source_name,
            "target_name": self.target_name,
            "explanation": self.explanation,
            "limitations": self.limitations,
            "breakdown": {
                "structural": self.structural.to_dict(),
                "semantic": self.semantic.to_dict(),
            },
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the similarity result."""
        return {
            "overall_score": round(self.overall_score, 4),
            "structural_score": round(self.structural.score, 4),
            "semantic_score": round(self.semantic.score, 4),
            "interpretation": self._interpret_score(),
            "top_matches": self.semantic.matched_pairs[:5],
            "common_dependencies": self.structural.details.get(
                "common_dependencies", []
            )[:10],
        }

    def _interpret_score(self) -> str:
        """Provide human-readable interpretation of the score."""
        score = self.overall_score

        if score >= 0.9:
            return "Very High Similarity - Codebases are nearly identical"
        elif score >= 0.75:
            return "High Similarity - Significant structural and semantic overlap"
        elif score >= 0.5:
            return "Moderate Similarity - Some common patterns and components"
        elif score >= 0.25:
            return "Low Similarity - Few common elements"
        else:
            return "Very Low Similarity - Codebases are largely different"


class HybridAnalyzer:
    """
    Combines structural and semantic similarity analysis.
    
    Provides weighted combination of signals with configurable
    weights and comprehensive result breakdown.
    """

    def __init__(self, config: SimilarityConfig = None):
        self.config = config or SimilarityConfig()
        self.structural_analyzer = StructuralAnalyzer()
        self.semantic_analyzer = SemanticAnalyzer(
            similarity_threshold=self.config.similarity_threshold
        )

    def compute_similarity(
        self,
        graph1: SemanticGraph,
        graph2: SemanticGraph,
    ) -> SimilarityResult:
        """
        Compute hybrid similarity between two graphs.
        
        Args:
            graph1: First semantic graph.
            graph2: Second semantic graph.
            
        Returns:
            SimilarityResult with complete analysis.
        """
        logger.info(
            f"Computing similarity between '{graph1.name}' and '{graph2.name}'"
        )

        structural_result = self.structural_analyzer.compute_similarity(
            graph1, graph2
        )

        semantic_result = self.semantic_analyzer.compute_similarity(
            graph1, graph2
        )

        overall_score = (
            self.config.structural_weight * structural_result.score +
            self.config.semantic_weight * semantic_result.score
        )

        explanation = self._generate_explanation(
            graph1, graph2, structural_result, semantic_result, overall_score
        )

        limitations = self._identify_limitations(graph1, graph2)

        result = SimilarityResult(
            overall_score=overall_score,
            structural=structural_result,
            semantic=semantic_result,
            structural_weight=self.config.structural_weight,
            semantic_weight=self.config.semantic_weight,
            source_name=graph1.name,
            target_name=graph2.name,
            explanation=explanation,
            limitations=limitations,
        )

        logger.info(
            f"Similarity computed: overall={overall_score:.4f}, "
            f"structural={structural_result.score:.4f}, "
            f"semantic={semantic_result.score:.4f}"
        )

        return result

    def _generate_explanation(
        self,
        graph1: SemanticGraph,
        graph2: SemanticGraph,
        structural: StructuralSimilarity,
        semantic: SemanticSimilarity,
        overall: float,
    ) -> str:
        """Generate human-readable explanation of similarity."""
        parts = []

        parts.append(
            f"The overall similarity score of {overall:.2%} is computed from "
            f"structural similarity ({structural.score:.2%}) weighted at "
            f"{self.config.structural_weight:.0%} and semantic similarity "
            f"({semantic.score:.2%}) weighted at {self.config.semantic_weight:.0%}."
        )

        if structural.size_ratio < 0.5:
            parts.append(
                f"Note: The repositories differ significantly in size "
                f"(ratio: {structural.size_ratio:.2f}), which may affect comparability."
            )

        if structural.dependency_overlap > 0.7:
            parts.append(
                f"High dependency overlap ({structural.dependency_overlap:.2%}) "
                f"suggests similar technology stacks."
            )
        elif structural.dependency_overlap < 0.3:
            parts.append(
                f"Low dependency overlap ({structural.dependency_overlap:.2%}) "
                f"indicates different technology choices."
            )

        if semantic.matched_pairs:
            top_match = semantic.matched_pairs[0]
            parts.append(
                f"Highest semantic match: '{top_match['node1_name']}' and "
                f"'{top_match['node2_name']}' ({top_match['similarity']:.2%})."
            )

        return " ".join(parts)

    def _identify_limitations(
        self,
        graph1: SemanticGraph,
        graph2: SemanticGraph,
    ) -> List[str]:
        """Identify limitations in the analysis."""
        limitations = []

        nodes_with_embeddings_1 = len(graph1.get_nodes_with_embeddings())
        nodes_with_embeddings_2 = len(graph2.get_nodes_with_embeddings())

        if nodes_with_embeddings_1 == 0 or nodes_with_embeddings_2 == 0:
            limitations.append(
                "One or both graphs have no embeddings, limiting semantic analysis."
            )
        elif nodes_with_embeddings_1 < 10 or nodes_with_embeddings_2 < 10:
            limitations.append(
                "Few entities have embeddings, semantic similarity may be less reliable."
            )

        languages1 = {n.language for n in graph1.iter_nodes()}
        languages2 = {n.language for n in graph2.iter_nodes()}
        common_languages = languages1 & languages2

        if not common_languages:
            limitations.append(
                "Repositories use different programming languages, "
                "limiting direct comparison."
            )

        if graph1.node_count < 5 or graph2.node_count < 5:
            limitations.append(
                "Small repository size limits the reliability of structural analysis."
            )

        limitations.append(
            "This analysis is based on static structure and semantics only; "
            "runtime behavior is not considered."
        )

        return limitations

    def set_weights(
        self,
        structural_weight: float = None,
        semantic_weight: float = None,
    ) -> None:
        """
        Update similarity weights.
        
        Args:
            structural_weight: New structural weight.
            semantic_weight: New semantic weight.
        """
        if structural_weight is not None:
            self.config.structural_weight = structural_weight
        if semantic_weight is not None:
            self.config.semantic_weight = semantic_weight

        total = self.config.structural_weight + self.config.semantic_weight
        self.config.structural_weight /= total
        self.config.semantic_weight /= total

        logger.info(
            f"Updated weights: structural={self.config.structural_weight:.2f}, "
            f"semantic={self.config.semantic_weight:.2f}"
        )

