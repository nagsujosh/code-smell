"""
Embedding pipeline stage for semantic encoding.

Coordinates payload extraction and embedding generation
for graph nodes.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import PipelineConfig
from src.core.pipeline import PipelineStage, PipelineState
from src.graph.semantic_graph import SemanticGraph
from src.embedding.extractor import PayloadExtractor, SemanticPayload
from src.embedding.encoder import CodeEncoder

logger = logging.getLogger(__name__)


class EmbeddingStage(PipelineStage):
    """
    Pipeline stage for generating semantic embeddings.
    
    Extracts semantic payloads from graph nodes and generates
    vector embeddings using pre-trained models.
    """

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.extractor = PayloadExtractor(config.embedding)
        self.encoder = None

    @property
    def name(self) -> str:
        return "embedding"

    @property
    def dependencies(self) -> List[str]:
        return ["graph_construction"]

    def _ensure_encoder(self) -> None:
        """Ensure the encoder is loaded."""
        if self.encoder is None:
            self.encoder = CodeEncoder(self.config.embedding)
            self.encoder.load_model()

    def execute(self, state: PipelineState) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate embeddings for graph nodes.
        
        Args:
            state: Pipeline state containing graphs.
            
        Returns:
            Tuple of (output_data, metrics).
        """
        graph_data = state.data.get("graph_construction", {})
        ingestion_data = state.data.get("ingestion", {})

        output = {}
        metrics = {
            "source_embeddings_generated": 0,
            "target_embeddings_generated": 0,
            "embedding_dimension": 0,
        }

        self._ensure_encoder()
        metrics["embedding_dimension"] = self.encoder.embedding_dim

        if "source" in graph_data:
            source_graph = graph_data["source"]
            source_files = self._get_source_files(ingestion_data.get("source"))

            source_result = self._embed_graph(source_graph, source_files)
            output["source"] = source_result
            metrics["source_embeddings_generated"] = source_result["count"]

        if "target" in graph_data:
            target_graph = graph_data["target"]
            target_files = self._get_source_files(ingestion_data.get("target"))

            target_result = self._embed_graph(target_graph, target_files)
            output["target"] = target_result
            metrics["target_embeddings_generated"] = target_result["count"]

        return output, metrics

    def _get_source_files(self, repository) -> Dict[str, str]:
        """Load source file contents from repository."""
        if repository is None:
            return {}

        source_files = {}
        for file_info in repository.files:
            try:
                with open(file_info.path, "r", encoding="utf-8", errors="replace") as f:
                    source_files[file_info.relative_path] = f.read()
            except (IOError, OSError) as e:
                self.logger.warning(f"Failed to read {file_info.path}: {e}")

        return source_files

    def _embed_graph(
        self,
        graph: SemanticGraph,
        source_files: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Generate embeddings for all applicable nodes in a graph.
        
        Args:
            graph: Semantic graph to process.
            source_files: Mapping of file paths to content.
            
        Returns:
            Dictionary with embedding results.
        """
        payloads = self.extractor.extract_from_graph(graph, source_files)

        embeddings = self.encoder.encode_payloads(payloads)

        for node_id, embedding in embeddings.items():
            node = graph.get_node(node_id)
            if node:
                node.embedding = embedding

        self.logger.info(
            f"Generated {len(embeddings)} embeddings for graph '{graph.name}'"
        )

        return {
            "graph": graph,
            "payloads": payloads,
            "count": len(embeddings),
            "dimension": self.encoder.embedding_dim,
        }

    def embed_code(self, code: str, entity_type: str = "function") -> Dict[str, Any]:
        """
        Embed a code snippet directly.
        
        Args:
            code: Source code to embed.
            entity_type: Type of code entity.
            
        Returns:
            Dictionary with embedding and payload.
        """
        self._ensure_encoder()

        payload = self.extractor.extract_from_code(code, entity_type)
        embedding = self.encoder.encode_payload(payload)

        return {
            "embedding": embedding,
            "payload": payload,
            "dimension": self.encoder.embedding_dim,
        }


def embed_graph(
    graph: SemanticGraph,
    source_files: Dict[str, str] = None,
    config: PipelineConfig = None,
) -> SemanticGraph:
    """
    Convenience function to embed a graph.
    
    Args:
        graph: Graph to embed.
        source_files: Optional source file contents.
        config: Optional configuration.
        
    Returns:
        Graph with embeddings attached to nodes.
    """
    if config is None:
        from src.core.config import Config
        config = Config.get()

    extractor = PayloadExtractor(config.embedding)
    encoder = CodeEncoder(config.embedding)
    encoder.load_model()

    payloads = extractor.extract_from_graph(graph, source_files or {})
    embeddings = encoder.encode_payloads(payloads)

    for node_id, embedding in embeddings.items():
        node = graph.get_node(node_id)
        if node:
            node.embedding = embedding

    return graph


def embed_code(code: str, config: PipelineConfig = None) -> Tuple[Any, SemanticPayload]:
    """
    Convenience function to embed code directly.
    
    Args:
        code: Source code to embed.
        config: Optional configuration.
        
    Returns:
        Tuple of (embedding, payload).
    """
    if config is None:
        from src.core.config import Config
        config = Config.get()

    extractor = PayloadExtractor(config.embedding)
    encoder = CodeEncoder(config.embedding)
    encoder.load_model()

    payload = extractor.extract_from_code(code)
    embedding = encoder.encode_payload(payload)

    return embedding, payload

