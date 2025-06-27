"""
Semantic payload extraction from code entities.

Extracts significant code content for embedding generation,
including function signatures, docstrings, and key code lines.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.config import EmbeddingConfig
from src.graph.semantic_graph import SemanticGraph, GraphNode

logger = logging.getLogger(__name__)


@dataclass
class SemanticPayload:
    """
    Semantic content extracted from a code entity.
    
    Contains the significant code content that will be
    embedded for semantic similarity computation.
    """

    node_id: str
    node_type: str
    signature: str
    docstring: Optional[str] = None
    code_lines: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_text(self, max_length: int = 512) -> str:
        """
        Convert payload to text for embedding.
        
        Args:
            max_length: Maximum character length.
            
        Returns:
            Text representation of the payload.
        """
        parts = []

        parts.append(self.signature)

        if self.docstring:
            doc_lines = self.docstring.split("\n")[:3]
            parts.append(" ".join(doc_lines).strip())

        if self.code_lines:
            code_text = " ".join(self.code_lines[:10])
            parts.append(code_text)

        text = " ".join(parts)

        if len(text) > max_length:
            text = text[:max_length - 3] + "..."

        return text

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "signature": self.signature,
            "docstring": self.docstring,
            "code_lines": self.code_lines,
            "imports": self.imports,
            "context": self.context,
        }


class PayloadExtractor:
    """
    Extracts semantic payloads from code entities.
    
    Processes graph nodes and source files to extract
    significant code content for embedding.
    """

    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()

    def extract_from_graph(
        self,
        graph: SemanticGraph,
        source_files: Dict[str, str] = None,
    ) -> Dict[str, SemanticPayload]:
        """
        Extract payloads for all nodes in a graph.
        
        Args:
            graph: Semantic graph to process.
            source_files: Mapping of file paths to content.
            
        Returns:
            Dictionary mapping node IDs to payloads.
        """
        payloads = {}
        source_files = source_files or {}

        for node in graph.iter_nodes():
            if node.node_type in ("function", "method", "class", "file"):
                payload = self._extract_node_payload(node, source_files)
                if payload:
                    payloads[node.id] = payload

        logger.info(f"Extracted {len(payloads)} semantic payloads")
        return payloads

    def _extract_node_payload(
        self,
        node: GraphNode,
        source_files: Dict[str, str],
    ) -> Optional[SemanticPayload]:
        """Extract payload for a single node."""
        if node.node_type == "function":
            return self._extract_function_payload(node, source_files)
        elif node.node_type == "class":
            return self._extract_class_payload(node, source_files)
        elif node.node_type == "file":
            return self._extract_file_payload(node, source_files)
        return None

    def _extract_function_payload(
        self,
        node: GraphNode,
        source_files: Dict[str, str],
    ) -> SemanticPayload:
        """Extract payload for a function node."""
        params = node.attributes.get("parameters", [])
        param_str = ", ".join(
            p.get("name", "") + (f": {p.get('type')}" if p.get("type") else "")
            for p in params
        )

        return_type = node.attributes.get("return_type", "")
        return_str = f" -> {return_type}" if return_type else ""

        is_async = node.attributes.get("is_async", False)
        prefix = "async def" if is_async else "def"

        signature = f"{prefix} {node.name}({param_str}){return_str}"

        docstring = node.attributes.get("docstring")

        code_lines = self._extract_code_lines(node, source_files)

        return SemanticPayload(
            node_id=node.id,
            node_type=node.node_type,
            signature=signature,
            docstring=docstring,
            code_lines=code_lines,
            context={
                "decorators": node.attributes.get("decorators", []),
                "complexity": node.attributes.get("complexity", 1),
                "language": node.language,
            },
        )

    def _extract_class_payload(
        self,
        node: GraphNode,
        source_files: Dict[str, str],
    ) -> SemanticPayload:
        """Extract payload for a class node."""
        base_classes = node.attributes.get("base_classes", [])
        bases_str = f"({', '.join(base_classes)})" if base_classes else ""

        signature = f"class {node.name}{bases_str}"

        docstring = node.attributes.get("docstring")

        code_lines = self._extract_code_lines(node, source_files)

        return SemanticPayload(
            node_id=node.id,
            node_type=node.node_type,
            signature=signature,
            docstring=docstring,
            code_lines=code_lines,
            context={
                "decorators": node.attributes.get("decorators", []),
                "is_abstract": node.attributes.get("is_abstract", False),
                "language": node.language,
            },
        )

    def _extract_file_payload(
        self,
        node: GraphNode,
        source_files: Dict[str, str],
    ) -> SemanticPayload:
        """Extract payload for a file node."""
        signature = f"file: {node.name}"

        file_path = node.qualified_name
        content = source_files.get(file_path, "")

        code_lines = []
        imports = []

        if content:
            lines = content.split("\n")
            for line in lines[:self.config.significant_lines_count]:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    code_lines.append(stripped)
                    if stripped.startswith(("import ", "from ")):
                        imports.append(stripped)

        return SemanticPayload(
            node_id=node.id,
            node_type=node.node_type,
            signature=signature,
            code_lines=code_lines[:20],
            imports=imports,
            context={
                "language": node.language,
            },
        )

    def _extract_code_lines(
        self,
        node: GraphNode,
        source_files: Dict[str, str],
    ) -> List[str]:
        """Extract significant code lines for a node."""
        location = node.attributes.get("location")
        if not location:
            return []

        file_path = location.get("file_path")
        if not file_path or file_path not in source_files:
            return []

        content = source_files[file_path]
        lines = content.split("\n")

        start_line = location.get("line_start", 1) - 1
        end_line = location.get("line_end", start_line + 10)

        max_lines = self.config.significant_lines_count
        if end_line - start_line > max_lines:
            end_line = start_line + max_lines

        code_lines = []
        for line in lines[start_line:end_line]:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                cleaned = self._clean_code_line(stripped)
                if cleaned:
                    code_lines.append(cleaned)

        return code_lines

    def _clean_code_line(self, line: str) -> str:
        """Clean a code line for embedding."""
        line = re.sub(r'#.*$', '', line).strip()

        line = re.sub(r'""".*?"""', '', line)
        line = re.sub(r"'''.*?'''", '', line)

        line = re.sub(r'"[^"]*"', '""', line)
        line = re.sub(r"'[^']*'", "''", line)

        line = re.sub(r'\d+\.\d+', 'NUM', line)
        line = re.sub(r'\d+', 'NUM', line)

        return line.strip()

    def extract_from_code(
        self,
        code: str,
        entity_type: str = "function",
        name: str = "unknown",
    ) -> SemanticPayload:
        """
        Extract payload directly from code content.
        
        Args:
            code: Source code content.
            entity_type: Type of the entity.
            name: Name of the entity.
            
        Returns:
            Extracted SemanticPayload.
        """
        lines = code.strip().split("\n")

        signature = lines[0] if lines else ""

        docstring = None
        code_start = 1
        if len(lines) > 1:
            for i, line in enumerate(lines[1:], 1):
                stripped = line.strip()
                if stripped.startswith(('"""', "'''")):
                    doc_lines = []
                    for j in range(i, len(lines)):
                        doc_lines.append(lines[j])
                        if lines[j].strip().endswith(('"""', "'''")):
                            code_start = j + 1
                            break
                    docstring = "\n".join(doc_lines)
                    break
                elif stripped and not stripped.startswith("#"):
                    break

        code_lines = []
        for line in lines[code_start:code_start + self.config.significant_lines_count]:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                code_lines.append(self._clean_code_line(stripped))

        return SemanticPayload(
            node_id=f"{entity_type}:{name}",
            node_type=entity_type,
            signature=signature,
            docstring=docstring,
            code_lines=code_lines,
        )

