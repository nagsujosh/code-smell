"""
Report generation from pipeline state.

Transforms pipeline results into structured reports with
comprehensive breakdowns and explanations.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import PipelineConfig
from src.core.pipeline import PipelineStage, PipelineState
from src.similarity.hybrid import SimilarityResult
from src.reporting.report import Report, ReportSection

logger = logging.getLogger(__name__)


class ReportGenerator(PipelineStage):
    """
    Pipeline stage for generating similarity reports.
    
    Creates comprehensive reports from analysis results with
    detailed breakdowns and explainability features.
    """

    def __init__(self, config: PipelineConfig):
        super().__init__(config)

    @property
    def name(self) -> str:
        return "reporting"

    @property
    def dependencies(self) -> List[str]:
        return ["similarity"]

    def execute(self, state: PipelineState) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate report from pipeline state.
        
        Args:
            state: Complete pipeline state.
            
        Returns:
            Tuple of (report, metrics).
        """
        similarity_data = state.data.get("similarity", {})
        ingestion_data = state.data.get("ingestion", {})

        if "result" in similarity_data:
            report = self._generate_comparison_report(state)
        else:
            report = self._generate_single_repo_report(state)

        output = {"report": report}
        metrics = {"sections_generated": len(report.sections)}

        return output, metrics

    def _generate_comparison_report(self, state: PipelineState) -> Report:
        """Generate an extensive comparison report for two repositories."""
        similarity_data = state.data["similarity"]
        ingestion_data = state.data.get("ingestion", {})
        analysis_data = state.data.get("analysis", {})
        storage_data = state.data.get("storage", {})

        result: SimilarityResult = similarity_data["result"]

        source_repo = ingestion_data.get("source")
        target_repo = ingestion_data.get("target")

        report = Report(
            title="Repository Similarity Analysis Report",
            pipeline_id=state.pipeline_id,
            source_repository=source_repo.metadata.source if source_repo else "",
            target_repository=target_repo.metadata.source if target_repo else "",
        )

        report.summary = {
            "overall_similarity": result.overall_score,
            "structural_similarity": result.structural.score,
            "semantic_similarity": result.semantic.score,
            "interpretation": result._interpret_score(),
        }

        report.add_section(self._create_overview_section(result))
        report.add_section(self._create_structural_section(result))
        report.add_section(self._create_semantic_section(result))
        report.add_section(self._create_repository_stats_section(
            source_repo, target_repo, storage_data
        ))
        report.add_section(self._create_graph_analysis_section(
            storage_data, result
        ))
        report.add_section(self._create_dependency_analysis_section(result))
        report.add_section(self._create_code_patterns_section(result))
        report.add_section(self._create_methodology_section())
        report.add_section(self._create_limitations_section(result))
        report.add_section(self._create_recommendations_section(result))

        report.metadata = {
            "pipeline_stages": list(state.stage_results.keys()),
            "config": {
                "structural_weight": result.structural_weight,
                "semantic_weight": result.semantic_weight,
            },
            "analysis_timestamp": datetime.now().isoformat(),
            "engine_version": "1.0.0",
        }

        return report

    def _generate_single_repo_report(self, state: PipelineState) -> Report:
        """Generate a report for a single repository analysis."""
        similarity_data = state.data.get("similarity", {})
        ingestion_data = state.data.get("ingestion", {})
        storage_data = state.data.get("storage", {})

        source_repo = ingestion_data.get("source")

        report = Report(
            title="Repository Analysis Report",
            pipeline_id=state.pipeline_id,
            source_repository=source_repo.metadata.source if source_repo else "",
        )

        report.summary = {
            "mode": "single_repository",
            "node_count": similarity_data.get("statistics", {}).get("node_count", 0),
            "edge_count": similarity_data.get("statistics", {}).get("edge_count", 0),
        }

        report.add_section(ReportSection(
            title="Repository Overview",
            content={
                "name": source_repo.metadata.name if source_repo else "",
                "source": source_repo.metadata.source if source_repo else "",
                "total_files": source_repo.metadata.total_files if source_repo else 0,
                "languages": source_repo.metadata.language_distribution if source_repo else {},
            }
        ))

        internal_sim = similarity_data.get("internal_similarities", {})
        if internal_sim:
            report.add_section(ReportSection(
                title="Internal Similarity Analysis",
                content={
                    "similar_pairs_count": internal_sim.get("similar_pairs_count", 0),
                    "top_similar_pairs": internal_sim.get("top_similar_pairs", []),
                },
                notes=[
                    "Internal similarity helps identify potential code duplication.",
                ]
            ))

        return report

    def _create_overview_section(self, result: SimilarityResult) -> ReportSection:
        """Create the overview section with comprehensive scoring details."""
        return ReportSection(
            title="Overview",
            content={
                "overall_score": result.overall_score,
                "interpretation": result._interpret_score(),
                "computed_at": result.computed_at.isoformat(),
                "source_repository": result.source_name,
                "target_repository": result.target_name,
                "weights": {
                    "structural": result.structural_weight,
                    "semantic": result.semantic_weight,
                },
                "score_formula": f"({result.structural_weight:.2f} * structural) + ({result.semantic_weight:.2f} * semantic)",
                "computed_score": f"({result.structural_weight:.2f} * {result.structural.score:.4f}) + ({result.semantic_weight:.2f} * {result.semantic.score:.4f}) = {result.overall_score:.4f}",
            },
            notes=[
                result.explanation,
            ]
        )

    def _create_structural_section(self, result: SimilarityResult) -> ReportSection:
        """Create the structural similarity section with full breakdown."""
        structural = result.structural

        graph1_stats = structural.details.get("graph1_stats", {})
        graph2_stats = structural.details.get("graph2_stats", {})

        return ReportSection(
            title="Structural Similarity",
            content={
                "score": structural.score,
                "components": {
                    "node_type_similarity": structural.node_type_similarity,
                    "edge_type_similarity": structural.edge_type_similarity,
                    "dependency_overlap": structural.dependency_overlap,
                    "topology_similarity": structural.topology_similarity,
                },
                "component_weights": {
                    "node_type": 0.25,
                    "edge_type": 0.20,
                    "dependency": 0.30,
                    "topology": 0.25,
                },
                "size_ratio": structural.size_ratio,
                "graph_comparison": {
                    "source_nodes": graph1_stats.get("node_count", 0),
                    "source_edges": graph1_stats.get("edge_count", 0),
                    "target_nodes": graph2_stats.get("node_count", 0),
                    "target_edges": graph2_stats.get("edge_count", 0),
                },
            },
            subsections=[
                ReportSection(
                    title="Common Dependencies",
                    content={
                        "dependencies": structural.details.get(
                            "common_dependencies", []
                        ),
                        "count": len(structural.details.get("common_dependencies", [])),
                    }
                ),
                ReportSection(
                    title="Node Type Distribution",
                    content={
                        "common_types": structural.details.get(
                            "common_node_types", []
                        ),
                        "source_distribution": graph1_stats.get("type_distribution", {}).get("nodes", {}),
                        "target_distribution": graph2_stats.get("type_distribution", {}).get("nodes", {}),
                    }
                ),
                ReportSection(
                    title="Edge Type Distribution",
                    content={
                        "common_types": structural.details.get(
                            "common_edge_types", []
                        ),
                        "source_distribution": graph1_stats.get("type_distribution", {}).get("edges", {}),
                        "target_distribution": graph2_stats.get("type_distribution", {}).get("edges", {}),
                    }
                ),
            ],
            notes=[
                "Structural similarity measures how alike the codebases are "
                "in terms of organization, dependencies, and graph topology.",
                f"Size ratio of {structural.size_ratio:.2%} indicates "
                + ("similar repository sizes." if structural.size_ratio > 0.5 else "significantly different repository sizes."),
            ]
        )

    def _create_semantic_section(self, result: SimilarityResult) -> ReportSection:
        """Create the semantic similarity section with full match details."""
        semantic = result.semantic

        function_matches = [m for m in semantic.matched_pairs if m["node1_type"] == "function"]
        class_matches = [m for m in semantic.matched_pairs if m["node1_type"] == "class"]
        file_matches = [m for m in semantic.matched_pairs if m["node1_type"] == "file"]

        return ReportSection(
            title="Semantic Similarity",
            content={
                "score": semantic.score,
                "components": {
                    "function_similarity": semantic.function_similarity,
                    "class_similarity": semantic.class_similarity,
                    "file_similarity": semantic.file_similarity,
                },
                "component_weights": {
                    "function": 0.50,
                    "class": 0.30,
                    "file": 0.20,
                },
                "total_matches": len(semantic.matched_pairs),
                "matches_by_type": {
                    "function_matches": len(function_matches),
                    "class_matches": len(class_matches),
                    "file_matches": len(file_matches),
                },
                "entity_counts": {
                    "source_functions": semantic.details.get("total_functions_1", 0),
                    "target_functions": semantic.details.get("total_functions_2", 0),
                    "source_classes": semantic.details.get("total_classes_1", 0),
                    "target_classes": semantic.details.get("total_classes_2", 0),
                },
            },
            subsections=[
                ReportSection(
                    title="Top Function Matches",
                    content={
                        "matches": [
                            {
                                "source": m["node1_name"],
                                "target": m["node2_name"],
                                "similarity": m["similarity"],
                            }
                            for m in function_matches[:15]
                        ],
                    }
                ),
                ReportSection(
                    title="Top Class Matches",
                    content={
                        "matches": [
                            {
                                "source": m["node1_name"],
                                "target": m["node2_name"],
                                "similarity": m["similarity"],
                            }
                            for m in class_matches[:15]
                        ],
                    }
                ),
                ReportSection(
                    title="Top File Matches",
                    content={
                        "matches": [
                            {
                                "source": m["node1_name"],
                                "target": m["node2_name"],
                                "similarity": m["similarity"],
                            }
                            for m in file_matches[:15]
                        ],
                    }
                ),
                ReportSection(
                    title="All Top Semantic Matches",
                    content={
                        "matches": [
                            {
                                "source": m["node1_name"],
                                "target": m["node2_name"],
                                "similarity": m["similarity"],
                                "type": m["node1_type"],
                            }
                            for m in semantic.matched_pairs[:25]
                        ],
                    }
                ),
            ],
            notes=[
                "Semantic similarity measures conceptual similarity based on "
                "code meaning using neural embeddings (CodeBERT).",
                f"Aggregation method: {semantic.aggregation_method}",
                f"Total matched pairs above threshold: {len(semantic.matched_pairs)}",
            ]
        )

    def _create_repository_stats_section(
        self,
        source_repo,
        target_repo,
        storage_data: Dict,
    ) -> ReportSection:
        """Create the repository statistics section with full details."""
        source_stats = {}
        target_stats = {}

        if source_repo:
            source_stats = {
                "name": source_repo.metadata.name,
                "files": source_repo.metadata.total_files,
                "size_bytes": source_repo.metadata.total_size_bytes,
                "size_mb": round(source_repo.metadata.total_size_bytes / (1024 * 1024), 2),
                "languages": source_repo.metadata.language_distribution,
                "primary_language": max(
                    source_repo.metadata.language_distribution.items(),
                    key=lambda x: x[1],
                    default=("unknown", 0)
                )[0] if source_repo.metadata.language_distribution else "unknown",
            }

        if target_repo:
            target_stats = {
                "name": target_repo.metadata.name,
                "files": target_repo.metadata.total_files,
                "size_bytes": target_repo.metadata.total_size_bytes,
                "size_mb": round(target_repo.metadata.total_size_bytes / (1024 * 1024), 2),
                "languages": target_repo.metadata.language_distribution,
                "primary_language": max(
                    target_repo.metadata.language_distribution.items(),
                    key=lambda x: x[1],
                    default=("unknown", 0)
                )[0] if target_repo.metadata.language_distribution else "unknown",
            }

        if "source" in storage_data:
            source_graph = storage_data["source"]["graph"]
            source_stats["nodes"] = source_graph.node_count
            source_stats["edges"] = source_graph.edge_count
            source_stats["node_types"] = list(source_graph.get_node_types())
            source_stats["edge_types"] = list(source_graph.get_edge_types())

        if "target" in storage_data:
            target_graph = storage_data["target"]["graph"]
            target_stats["nodes"] = target_graph.node_count
            target_stats["edges"] = target_graph.edge_count
            target_stats["node_types"] = list(target_graph.get_node_types())
            target_stats["edge_types"] = list(target_graph.get_edge_types())

        return ReportSection(
            title="Repository Statistics",
            content={
                "source": source_stats,
                "target": target_stats,
                "comparison": {
                    "file_ratio": (
                        min(source_stats.get("files", 1), target_stats.get("files", 1)) /
                        max(source_stats.get("files", 1), target_stats.get("files", 1))
                    ) if source_stats.get("files") and target_stats.get("files") else 0,
                    "node_ratio": (
                        min(source_stats.get("nodes", 1), target_stats.get("nodes", 1)) /
                        max(source_stats.get("nodes", 1), target_stats.get("nodes", 1))
                    ) if source_stats.get("nodes") and target_stats.get("nodes") else 0,
                },
            }
        )

    def _create_graph_analysis_section(
        self,
        storage_data: Dict,
        result: SimilarityResult,
    ) -> ReportSection:
        """Create graph analysis section with topology details."""
        graph1_stats = result.structural.details.get("graph1_stats", {})
        graph2_stats = result.structural.details.get("graph2_stats", {})

        return ReportSection(
            title="Graph Topology Analysis",
            content={
                "source_graph": {
                    "node_count": graph1_stats.get("node_count", 0),
                    "edge_count": graph1_stats.get("edge_count", 0),
                    "density": graph1_stats.get("density", 0),
                    "type_distribution": graph1_stats.get("type_distribution", {}),
                },
                "target_graph": {
                    "node_count": graph2_stats.get("node_count", 0),
                    "edge_count": graph2_stats.get("edge_count", 0),
                    "density": graph2_stats.get("density", 0),
                    "type_distribution": graph2_stats.get("type_distribution", {}),
                },
                "topology_similarity": result.structural.topology_similarity,
            },
            notes=[
                "Graph topology reveals the structural organization of the codebase.",
                "Higher density indicates more interconnected code components.",
            ]
        )

    def _create_dependency_analysis_section(
        self,
        result: SimilarityResult,
    ) -> ReportSection:
        """Create dependency analysis section."""
        common_deps = result.structural.details.get("common_dependencies", [])
        
        return ReportSection(
            title="Dependency Analysis",
            content={
                "dependency_overlap": result.structural.dependency_overlap,
                "common_dependencies": common_deps,
                "common_dependency_count": len(common_deps),
            },
            notes=[
                f"Dependency overlap of {result.structural.dependency_overlap:.2%} indicates "
                + ("similar technology stacks." if result.structural.dependency_overlap > 0.5 
                   else "different technology choices."),
                "Common dependencies suggest shared functionality or framework usage.",
            ]
        )

    def _create_code_patterns_section(
        self,
        result: SimilarityResult,
    ) -> ReportSection:
        """Create code patterns analysis section."""
        semantic = result.semantic
        
        high_similarity_matches = [
            m for m in semantic.matched_pairs 
            if m["similarity"] >= 0.9
        ]
        
        moderate_matches = [
            m for m in semantic.matched_pairs 
            if 0.8 <= m["similarity"] < 0.9
        ]
        
        return ReportSection(
            title="Code Pattern Analysis",
            content={
                "high_similarity_patterns": {
                    "count": len(high_similarity_matches),
                    "threshold": 0.9,
                    "matches": [
                        {
                            "source": m["node1_name"],
                            "target": m["node2_name"],
                            "similarity": m["similarity"],
                            "type": m["node1_type"],
                        }
                        for m in high_similarity_matches[:10]
                    ],
                },
                "moderate_similarity_patterns": {
                    "count": len(moderate_matches),
                    "threshold_range": [0.8, 0.9],
                    "matches": [
                        {
                            "source": m["node1_name"],
                            "target": m["node2_name"],
                            "similarity": m["similarity"],
                            "type": m["node1_type"],
                        }
                        for m in moderate_matches[:10]
                    ],
                },
            },
            notes=[
                f"Found {len(high_similarity_matches)} code elements with very high similarity (>=90%).",
                f"Found {len(moderate_matches)} code elements with high similarity (80-90%).",
                "High similarity patterns may indicate shared code, common idioms, or similar implementations.",
            ]
        )

    def _create_methodology_section(self) -> ReportSection:
        """Create the methodology section with detailed explanation."""
        return ReportSection(
            title="Methodology",
            content={
                "approach": "Hybrid structural and semantic analysis",
                "pipeline_stages": [
                    "Repository Ingestion",
                    "Language Detection",
                    "Static Structure Extraction",
                    "Semantic Graph Construction",
                    "Embedding Generation",
                    "Similarity Analysis",
                    "Report Generation",
                ],
                "structural_analysis": {
                    "description": "Compares graph topology, node/edge distributions, "
                                   "and dependency overlap",
                    "metrics": [
                        "Node type distribution similarity (cosine)",
                        "Edge type distribution similarity (cosine)",
                        "Dependency overlap (Jaccard index)",
                        "Topology metrics (density, clustering, depth)",
                    ],
                    "weights": {
                        "node_type": 0.25,
                        "edge_type": 0.20,
                        "dependency": 0.30,
                        "topology": 0.25,
                    },
                },
                "semantic_analysis": {
                    "description": "Compares code meaning using neural embeddings",
                    "encoder": "microsoft/codebert-base",
                    "embedding_dimension": 768,
                    "granularity": ["function", "class", "file"],
                    "matching": "Greedy optimal matching with threshold",
                    "similarity_threshold": 0.7,
                    "weights": {
                        "function": 0.50,
                        "class": 0.30,
                        "file": 0.20,
                    },
                },
                "hybrid_combination": {
                    "description": "Weighted combination of structural and semantic scores",
                    "default_structural_weight": 0.4,
                    "default_semantic_weight": 0.6,
                },
            }
        )

    def _create_limitations_section(self, result: SimilarityResult) -> ReportSection:
        """Create the limitations section with explicit assumptions."""
        return ReportSection(
            title="Limitations and Assumptions",
            content={
                "analysis_limitations": result.limitations,
                "general_limitations": [
                    "Static analysis only - runtime behavior not considered",
                    "Variable-level data flow not tracked",
                    "Similarity is computed, not semantic equivalence",
                    "Cross-language comparison limited by embedding model training",
                    "Large repositories may have sampling limitations",
                ],
                "assumptions": [
                    "Code follows standard conventions for the language",
                    "File extensions correctly indicate language",
                    "Repository structure is representative of codebase organization",
                    "Function/class naming reflects purpose",
                    "Import statements reflect actual dependencies",
                ],
                "score_interpretation": {
                    "0.9_to_1.0": "Very High - Nearly identical codebases",
                    "0.75_to_0.9": "High - Significant overlap, possibly related projects",
                    "0.5_to_0.75": "Moderate - Some common patterns",
                    "0.25_to_0.5": "Low - Few common elements",
                    "0.0_to_0.25": "Very Low - Largely different codebases",
                },
            }
        )

    def _create_recommendations_section(self, result: SimilarityResult) -> ReportSection:
        """Create recommendations based on analysis results."""
        recommendations = []
        
        if result.overall_score >= 0.8:
            recommendations.extend([
                "Consider investigating potential code sharing or common ancestry",
                "Review licensing if repositories are from different sources",
                "Document the relationship between these codebases",
            ])
        elif result.overall_score >= 0.5:
            recommendations.extend([
                "Explore the common patterns for potential code reuse",
                "Consider creating shared libraries for common functionality",
                "Document architectural similarities for knowledge transfer",
            ])
        else:
            recommendations.extend([
                "These codebases appear distinct and serve different purposes",
                "Limited opportunity for direct code sharing",
                "May benefit from different architectural approaches",
            ])
        
        if result.structural.dependency_overlap > 0.7:
            recommendations.append(
                "High dependency overlap - consider standardizing dependency versions"
            )
        
        if abs(result.structural.score - result.semantic.score) > 0.2:
            recommendations.append(
                "Significant difference between structural and semantic similarity - "
                "investigate organizational vs. implementation differences"
            )
        
        return ReportSection(
            title="Recommendations",
            content={
                "recommendations": recommendations,
                "based_on": {
                    "overall_score": result.overall_score,
                    "structural_score": result.structural.score,
                    "semantic_score": result.semantic.score,
                },
            }
        )


def generate_report(
    state: PipelineState,
    config: PipelineConfig = None,
) -> Report:
    """
    Convenience function to generate a report from pipeline state.
    
    Args:
        state: Complete pipeline state.
        config: Optional configuration.
        
    Returns:
        Generated Report.
    """
    if config is None:
        from src.core.config import Config
        config = Config.get()

    generator = ReportGenerator(config)
    output, _ = generator.execute(state)
    return output["report"]
