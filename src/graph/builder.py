"""
Graph builder for constructing semantic graphs from analysis results.

Transforms extracted code entities and relationships into a
queryable semantic graph representation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import PipelineConfig, GraphConfig
from src.core.pipeline import PipelineStage, PipelineState
from src.analysis.entities import (
    CodeEntity,
    Relationship,
    EntityType,
    RelationshipType,
)
from src.analysis.analyzer import AnalysisResult
from src.graph.semantic_graph import SemanticGraph, GraphNode, GraphEdge

logger = logging.getLogger(__name__)


class GraphBuilder(PipelineStage):
    """
    Pipeline stage for building semantic graphs.
    
    Converts analysis results into structured semantic graphs
    with typed nodes and edges.
    """

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.graph_config = config.graph

    @property
    def name(self) -> str:
        return "graph_construction"

    @property
    def dependencies(self) -> List[str]:
        return ["analysis"]

    def execute(self, state: PipelineState) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Build semantic graphs from analysis results.
        
        Args:
            state: Pipeline state containing analysis results.
            
        Returns:
            Tuple of (graphs_dict, metrics).
        """
        analysis_data = state.data.get("analysis", {})
        output = {}
        metrics = {
            "source_nodes": 0,
            "source_edges": 0,
            "target_nodes": 0,
            "target_edges": 0,
        }

        if "source" in analysis_data:
            ingestion_data = state.data.get("ingestion", {})
            source_repo = ingestion_data.get("source")
            repo_name = source_repo.metadata.name if source_repo else "source"

            source_result = analysis_data["source"]["result"]
            source_graph = self._build_graph(source_result, repo_name)
            output["source"] = source_graph
            metrics["source_nodes"] = source_graph.node_count
            metrics["source_edges"] = source_graph.edge_count

        if "target" in analysis_data:
            ingestion_data = state.data.get("ingestion", {})
            target_repo = ingestion_data.get("target")
            repo_name = target_repo.metadata.name if target_repo else "target"

            target_result = analysis_data["target"]["result"]
            target_graph = self._build_graph(target_result, repo_name)
            output["target"] = target_graph
            metrics["target_nodes"] = target_graph.node_count
            metrics["target_edges"] = target_graph.edge_count

        return output, metrics

    def _build_graph(
        self, analysis_result: AnalysisResult, name: str
    ) -> SemanticGraph:
        """
        Build a semantic graph from analysis results.
        
        Args:
            analysis_result: Result from static analysis.
            name: Name for the graph.
            
        Returns:
            Constructed SemanticGraph.
        """
        graph = SemanticGraph(name=name)

        repo_node = GraphNode(
            id=f"repo:{name}",
            node_type="repository",
            name=name,
            qualified_name=name,
            language="mixed",
            attributes={"is_root": True},
        )
        graph.add_node(repo_node)

        entity_id_map: Dict[str, str] = {}

        for entity in analysis_result.entities:
            if not self._should_include_entity(entity):
                continue

            node = self._entity_to_node(entity)
            graph.add_node(node)
            entity_id_map[entity.id] = node.id

            if entity.entity_type == EntityType.FILE:
                graph.add_edge(GraphEdge(
                    source_id=repo_node.id,
                    target_id=node.id,
                    edge_type="contains",
                ))

        for relationship in analysis_result.relationships:
            if not self._should_include_relationship(relationship):
                continue

            source_id = entity_id_map.get(relationship.source_id)
            target_id = entity_id_map.get(relationship.target_id)

            if source_id and target_id:
                edge = self._relationship_to_edge(
                    relationship, source_id, target_id
                )
                graph.add_edge(edge)

        stats = graph.get_statistics()
        self.logger.info(
            f"Built graph '{name}': {stats['node_count']} nodes, "
            f"{stats['edge_count']} edges"
        )

        return graph

    def _should_include_entity(self, entity: CodeEntity) -> bool:
        """Check if an entity should be included in the graph."""
        entity_type_str = entity.entity_type.value

        type_mapping = {
            "file": "file",
            "module": "module",
            "package": "module",
            "class": "class",
            "interface": "interface",
            "struct": "class",
            "function": "function",
            "method": "function",
            "import": "external_dependency",
            "external_dependency": "external_dependency",
        }

        mapped_type = type_mapping.get(entity_type_str, entity_type_str)
        return mapped_type in self.graph_config.node_types

    def _should_include_relationship(self, relationship: Relationship) -> bool:
        """Check if a relationship should be included in the graph."""
        rel_type = relationship.relationship_type.value
        return rel_type in self.graph_config.edge_types

    def _entity_to_node(self, entity: CodeEntity) -> GraphNode:
        """Convert a code entity to a graph node."""
        entity_type_str = entity.entity_type.value

        type_mapping = {
            "method": "function",
            "package": "module",
            "struct": "class",
            "import": "external_dependency",
        }
        node_type = type_mapping.get(entity_type_str, entity_type_str)

        attributes = dict(entity.attributes)
        if hasattr(entity, "docstring") and entity.docstring:
            attributes["docstring"] = entity.docstring[:500]

        if hasattr(entity, "location") and entity.location:
            attributes["location"] = entity.location.to_dict()

        if hasattr(entity, "decorators"):
            attributes["decorators"] = entity.decorators
        if hasattr(entity, "parameters"):
            attributes["parameters"] = entity.parameters
        if hasattr(entity, "return_type"):
            attributes["return_type"] = entity.return_type
        if hasattr(entity, "base_classes"):
            attributes["base_classes"] = entity.base_classes
        if hasattr(entity, "complexity"):
            attributes["complexity"] = entity.complexity

        return GraphNode(
            id=entity.id,
            node_type=node_type,
            name=entity.name,
            qualified_name=entity.qualified_name,
            language=entity.language,
            attributes=attributes,
        )

    def _relationship_to_edge(
        self,
        relationship: Relationship,
        source_id: str,
        target_id: str,
    ) -> GraphEdge:
        """Convert a relationship to a graph edge."""
        return GraphEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=relationship.relationship_type.value,
            attributes=relationship.attributes,
        )


def build_graph(
    analysis_result: AnalysisResult,
    name: str = "codebase",
    config: PipelineConfig = None,
) -> SemanticGraph:
    """
    Convenience function to build a graph from analysis results.
    
    Args:
        analysis_result: Result from static analysis.
        name: Name for the graph.
        config: Optional pipeline configuration.
        
    Returns:
        Constructed SemanticGraph.
    """
    if config is None:
        from src.core.config import Config
        config = Config.get()

    builder = GraphBuilder(config)
    return builder._build_graph(analysis_result, name)

