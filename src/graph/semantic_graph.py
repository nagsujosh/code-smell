"""
Semantic graph data structures.

Defines the core graph representation for codebases with
typed nodes, edges, and associated metadata.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """
    Node in the semantic graph.
    
    Represents a code entity with associated metadata and
    optional embedding vector.
    """

    id: str
    node_type: str
    name: str
    qualified_name: str
    language: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    code_snippet: Optional[str] = field(default=None, repr=False)

    def to_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "node_type": self.node_type,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "language": self.language,
            "attributes": self.attributes,
        }
        if include_embedding and self.embedding is not None:
            data["embedding"] = self.embedding.tolist()
        if self.code_snippet:
            data["code_snippet"] = self.code_snippet
        return data


@dataclass
class GraphEdge:
    """
    Edge in the semantic graph.
    
    Represents a relationship between two code entities.
    """

    source_id: str
    target_id: str
    edge_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "attributes": self.attributes,
        }


class SemanticGraph:
    """
    Semantic graph representation of a codebase.
    
    Wraps a NetworkX directed graph with additional functionality
    for code-specific operations and embedding management.
    """

    def __init__(self, name: str = ""):
        self.name = name
        self._graph = nx.DiGraph()
        self._nodes: Dict[str, GraphNode] = {}
        self._node_type_index: Dict[str, Set[str]] = {}
        self._edge_type_index: Dict[str, List[Tuple[str, str]]] = {}

    @property
    def node_count(self) -> int:
        """Number of nodes in the graph."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Number of edges in the graph."""
        return self._graph.number_of_edges()

    def add_node(self, node: GraphNode) -> None:
        """
        Add a node to the graph.
        
        Args:
            node: GraphNode to add.
        """
        self._nodes[node.id] = node
        self._graph.add_node(
            node.id,
            node_type=node.node_type,
            name=node.name,
            qualified_name=node.qualified_name,
            language=node.language,
            **node.attributes,
        )

        if node.node_type not in self._node_type_index:
            self._node_type_index[node.node_type] = set()
        self._node_type_index[node.node_type].add(node.id)

    def add_edge(self, edge: GraphEdge) -> None:
        """
        Add an edge to the graph.
        
        Args:
            edge: GraphEdge to add.
        """
        if edge.source_id not in self._nodes:
            logger.warning(f"Source node not found: {edge.source_id}")
            return
        if edge.target_id not in self._nodes:
            logger.warning(f"Target node not found: {edge.target_id}")
            return

        self._graph.add_edge(
            edge.source_id,
            edge.target_id,
            edge_type=edge.edge_type,
            **edge.attributes,
        )

        if edge.edge_type not in self._edge_type_index:
            self._edge_type_index[edge.edge_type] = []
        self._edge_type_index[edge.edge_type].append(
            (edge.source_id, edge.target_id)
        )

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_nodes_by_type(self, node_type: str) -> List[GraphNode]:
        """Get all nodes of a specific type."""
        node_ids = self._node_type_index.get(node_type, set())
        return [self._nodes[nid] for nid in node_ids]

    def get_edges_by_type(self, edge_type: str) -> List[GraphEdge]:
        """Get all edges of a specific type."""
        edge_pairs = self._edge_type_index.get(edge_type, [])
        edges = []
        for source_id, target_id in edge_pairs:
            edge_data = self._graph.get_edge_data(source_id, target_id)
            if edge_data:
                edges.append(GraphEdge(
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=edge_data.get("edge_type", edge_type),
                    attributes={
                        k: v for k, v in edge_data.items()
                        if k != "edge_type"
                    },
                ))
        return edges

    def get_neighbors(
        self, node_id: str, direction: str = "out"
    ) -> List[GraphNode]:
        """
        Get neighboring nodes.
        
        Args:
            node_id: ID of the source node.
            direction: "in", "out", or "both".
            
        Returns:
            List of neighboring nodes.
        """
        neighbors = set()
        if direction in ("out", "both"):
            neighbors.update(self._graph.successors(node_id))
        if direction in ("in", "both"):
            neighbors.update(self._graph.predecessors(node_id))
        return [self._nodes[nid] for nid in neighbors if nid in self._nodes]

    def get_subgraph(self, node_ids: Set[str]) -> "SemanticGraph":
        """
        Extract a subgraph containing only specified nodes.
        
        Args:
            node_ids: Set of node IDs to include.
            
        Returns:
            New SemanticGraph containing only the specified nodes.
        """
        subgraph = SemanticGraph(name=f"{self.name}_subgraph")

        for node_id in node_ids:
            if node_id in self._nodes:
                subgraph.add_node(self._nodes[node_id])

        for source_id, target_id in self._graph.edges():
            if source_id in node_ids and target_id in node_ids:
                edge_data = self._graph.get_edge_data(source_id, target_id)
                subgraph.add_edge(GraphEdge(
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=edge_data.get("edge_type", "unknown"),
                    attributes={
                        k: v for k, v in edge_data.items()
                        if k != "edge_type"
                    },
                ))

        return subgraph

    def iter_nodes(self) -> Iterator[GraphNode]:
        """Iterate over all nodes."""
        yield from self._nodes.values()

    def iter_edges(self) -> Iterator[GraphEdge]:
        """Iterate over all edges."""
        for source_id, target_id, data in self._graph.edges(data=True):
            yield GraphEdge(
                source_id=source_id,
                target_id=target_id,
                edge_type=data.get("edge_type", "unknown"),
                attributes={k: v for k, v in data.items() if k != "edge_type"},
            )

    def get_node_types(self) -> Set[str]:
        """Get all node types present in the graph."""
        return set(self._node_type_index.keys())

    def get_edge_types(self) -> Set[str]:
        """Get all edge types present in the graph."""
        return set(self._edge_type_index.keys())

    def get_type_distribution(self) -> Dict[str, Dict[str, int]]:
        """Get distribution of node and edge types."""
        return {
            "nodes": {
                node_type: len(node_ids)
                for node_type, node_ids in self._node_type_index.items()
            },
            "edges": {
                edge_type: len(edges)
                for edge_type, edges in self._edge_type_index.items()
            },
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        if self.node_count == 0:
            return {
                "node_count": 0,
                "edge_count": 0,
                "type_distribution": {"nodes": {}, "edges": {}},
            }

        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "type_distribution": self.get_type_distribution(),
            "density": nx.density(self._graph),
            "avg_degree": sum(d for _, d in self._graph.degree()) / self.node_count,
            "connected_components": (
                nx.number_weakly_connected_components(self._graph)
            ),
        }

    def to_dict(self, include_embeddings: bool = False) -> Dict[str, Any]:
        """Convert graph to dictionary for serialization."""
        return {
            "name": self.name,
            "nodes": [
                node.to_dict(include_embedding=include_embeddings)
                for node in self._nodes.values()
            ],
            "edges": [edge.to_dict() for edge in self.iter_edges()],
            "statistics": self.get_statistics(),
        }

    def save(self, path: Path, include_embeddings: bool = True) -> None:
        """
        Save graph to file.
        
        Args:
            path: Path to save the graph.
            include_embeddings: Whether to include embedding vectors.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict(include_embeddings=include_embeddings)

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Graph saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "SemanticGraph":
        """
        Load graph from file.
        
        Args:
            path: Path to the graph file.
            
        Returns:
            Loaded SemanticGraph.
        """
        with open(path, "r") as f:
            data = json.load(f)

        graph = cls(name=data.get("name", ""))

        for node_data in data.get("nodes", []):
            embedding = None
            if "embedding" in node_data:
                embedding = np.array(node_data["embedding"])

            node = GraphNode(
                id=node_data["id"],
                node_type=node_data["node_type"],
                name=node_data["name"],
                qualified_name=node_data["qualified_name"],
                language=node_data["language"],
                attributes=node_data.get("attributes", {}),
                embedding=embedding,
                code_snippet=node_data.get("code_snippet"),
            )
            graph.add_node(node)

        for edge_data in data.get("edges", []):
            edge = GraphEdge(
                source_id=edge_data["source_id"],
                target_id=edge_data["target_id"],
                edge_type=edge_data["edge_type"],
                attributes=edge_data.get("attributes", {}),
            )
            graph.add_edge(edge)

        logger.info(f"Graph loaded from {path}")
        return graph

    def get_networkx_graph(self) -> nx.DiGraph:
        """Get the underlying NetworkX graph."""
        return self._graph

    def compute_pagerank(self) -> Dict[str, float]:
        """Compute PageRank scores for all nodes."""
        if self.node_count == 0:
            return {}
        return nx.pagerank(self._graph)

    def find_shortest_path(
        self, source_id: str, target_id: str
    ) -> Optional[List[str]]:
        """Find shortest path between two nodes."""
        try:
            return nx.shortest_path(self._graph, source_id, target_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_nodes_with_embeddings(self) -> List[GraphNode]:
        """Get all nodes that have embedding vectors."""
        return [
            node for node in self._nodes.values()
            if node.embedding is not None
        ]

