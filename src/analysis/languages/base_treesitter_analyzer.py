"""
Base Tree-sitter Analyzer Module.

Provides abstract base class for language analyzers using tree-sitter
for accurate AST-based parsing. Tree-sitter offers consistent, fast,
and reliable parsing across multiple programming languages.
"""

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from src.analysis.registry import BaseLanguageAnalyzer
from src.analysis.analyzer import AnalysisResult
from src.analysis.entities import (
    CodeEntity,
    FileEntity,
    ClassEntity,
    FunctionEntity,
    ImportEntity,
    ModuleEntity,
    Relationship,
    RelationshipType,
    Location,
    EntityType,
)

logger = logging.getLogger(__name__)


class BaseTreeSitterAnalyzer(BaseLanguageAnalyzer):
    """
    Abstract base class for tree-sitter based language analyzers.
    
    Tree-sitter provides incremental parsing with a consistent API
    across languages, enabling accurate extraction of code entities
    and relationships from source files.
    
    Subclasses must implement:
        - _get_parser(): Return configured tree-sitter parser
        - _get_query_patterns(): Return tree-sitter query patterns
        - _process_captures(): Process query captures into entities
    """

    LANGUAGE: str = "unknown"
    SUPPORTED_EXTENSIONS: List[str] = []

    # Tree-sitter node types for common constructs (override in subclasses)
    CLASS_NODE_TYPES: List[str] = []
    FUNCTION_NODE_TYPES: List[str] = []
    IMPORT_NODE_TYPES: List[str] = []
    COMMENT_NODE_TYPES: List[str] = ["comment", "line_comment", "block_comment"]

    def __init__(self):
        self._parser = None
        self._language = None
        self._current_file = None
        self._current_module = None

    @abstractmethod
    def _initialize_parser(self) -> bool:
        """
        Initialize the tree-sitter parser with language grammar.
        
        Returns:
            True if initialization successful, False otherwise.
        """
        pass

    @abstractmethod
    def _extract_classes(self, tree, content: bytes) -> List[ClassEntity]:
        """Extract class/struct/interface entities from parse tree."""
        pass

    @abstractmethod
    def _extract_functions(self, tree, content: bytes) -> List[FunctionEntity]:
        """Extract function/method entities from parse tree."""
        pass

    @abstractmethod
    def _extract_imports(self, tree, content: bytes) -> List[ImportEntity]:
        """Extract import/require statements from parse tree."""
        pass

    def analyze_file(self, file_path: str, content: str) -> AnalysisResult:
        """
        Analyze a source file using tree-sitter parsing.
        
        Args:
            file_path: Path to the file being analyzed.
            content: Source code content as string.
            
        Returns:
            AnalysisResult containing extracted entities and relationships.
        """
        self._current_file = file_path
        self._current_module = self._path_to_module(file_path)

        result = AnalysisResult()

        # Initialize parser if needed
        if not self._parser and not self._initialize_parser():
            logger.warning(
                f"Tree-sitter parser unavailable for {self.LANGUAGE}, "
                f"skipping {file_path}"
            )
            return result

        # Create file entity
        file_entity = FileEntity(
            name=file_path.split("/")[-1],
            file_path=file_path,
            language=self.LANGUAGE,
        )
        result.entities.append(file_entity)

        # Parse content
        try:
            content_bytes = content.encode("utf-8")
            tree = self._parser.parse(content_bytes)
        except Exception as e:
            logger.error(f"Parse error in {file_path}: {e}")
            result.errors.append({
                "file": file_path,
                "error": str(e),
                "type": "ParseError",
            })
            return result

        # Extract entities
        classes = self._extract_classes(tree, content_bytes)
        functions = self._extract_functions(tree, content_bytes)
        imports = self._extract_imports(tree, content_bytes)

        result.entities.extend(classes)
        result.entities.extend(functions)
        result.entities.extend(imports)

        # Build relationships
        relationships = self._build_relationships(
            file_entity, classes, functions, imports, tree, content_bytes
        )
        result.relationships.extend(relationships)

        # Compute metrics
        result.metrics = {
            "total_lines": len(content.split("\n")),
            "classes": len(classes),
            "functions": len(functions),
            "imports": len(imports),
            "parse_errors": len(result.errors),
        }

        return result

    def _build_relationships(
        self,
        file_entity: FileEntity,
        classes: List[ClassEntity],
        functions: List[FunctionEntity],
        imports: List[ImportEntity],
        tree,
        content_bytes: bytes,
    ) -> List[Relationship]:
        """
        Build relationships between extracted entities.
        
        Args:
            file_entity: The file containing all entities.
            classes: Extracted class entities.
            functions: Extracted function entities.
            imports: Extracted import entities.
            tree: Parsed tree-sitter tree.
            content_bytes: Source content as bytes.
            
        Returns:
            List of relationships between entities.
        """
        relationships = []

        # File contains classes
        for cls in classes:
            relationships.append(Relationship(
                source_id=file_entity.id,
                target_id=cls.id,
                relationship_type=RelationshipType.CONTAINS,
            ))

        # File contains top-level functions
        for func in functions:
            if not func.is_method:
                relationships.append(Relationship(
                    source_id=file_entity.id,
                    target_id=func.id,
                    relationship_type=RelationshipType.CONTAINS,
                ))

        # File imports dependencies
        for imp in imports:
            relationships.append(Relationship(
                source_id=file_entity.id,
                target_id=imp.id,
                relationship_type=RelationshipType.IMPORTS,
            ))

        # Class defines methods
        relationships.extend(
            self._build_class_method_relationships(classes, functions)
        )

        # Function calls (extracted from tree analysis)
        relationships.extend(
            self._extract_call_relationships(functions, tree, content_bytes)
        )

        return relationships

    def _build_class_method_relationships(
        self,
        classes: List[ClassEntity],
        functions: List[FunctionEntity],
    ) -> List[Relationship]:
        """Build relationships between classes and their methods."""
        relationships = []
        
        for func in functions:
            if func.is_method and func.parent_class:
                for cls in classes:
                    if cls.name == func.parent_class:
                        relationships.append(Relationship(
                            source_id=cls.id,
                            target_id=func.id,
                            relationship_type=RelationshipType.DEFINES,
                        ))
                        break

        return relationships

    def _extract_call_relationships(
        self,
        functions: List[FunctionEntity],
        tree,
        content_bytes: bytes,
    ) -> List[Relationship]:
        """
        Extract function call relationships from the parse tree.
        
        This is a simplified implementation; subclasses can override
        for more accurate language-specific call detection.
        """
        relationships = []
        function_names = {f.name: f for f in functions}
        
        # Walk tree looking for call expressions
        cursor = tree.walk()
        
        def visit_node(node):
            if node.type in ("call_expression", "function_call", "method_call"):
                # Try to extract called function name
                callee = self._get_call_target(node, content_bytes)
                if callee and callee in function_names:
                    # Find the containing function
                    container = self._find_containing_function(
                        node, functions, content_bytes
                    )
                    if container and container.name != callee:
                        relationships.append(Relationship(
                            source_id=container.id,
                            target_id=function_names[callee].id,
                            relationship_type=RelationshipType.CALLS,
                        ))
            
            for child in node.children:
                visit_node(child)
        
        visit_node(tree.root_node)
        return relationships

    def _get_call_target(self, node, content_bytes: bytes) -> Optional[str]:
        """Extract the target function name from a call expression node."""
        # Default implementation - subclasses should override for accuracy
        for child in node.children:
            if child.type == "identifier":
                return content_bytes[child.start_byte:child.end_byte].decode("utf-8")
        return None

    def _find_containing_function(
        self,
        node,
        functions: List[FunctionEntity],
        content_bytes: bytes,
    ) -> Optional[FunctionEntity]:
        """Find the function entity that contains a given node."""
        node_line = node.start_point[0] + 1  # Convert to 1-based
        
        for func in functions:
            if func.location:
                if (func.location.line_start <= node_line <= 
                    (func.location.line_end or func.location.line_start + 1000)):
                    return func
        return None

    def _path_to_module(self, file_path: str) -> str:
        """Convert file path to module-like identifier."""
        module = file_path.replace("/", ".").replace("\\", ".")
        for ext in self.SUPPORTED_EXTENSIONS:
            if module.endswith(ext):
                module = module[:-len(ext)]
                break
        return module.strip(".")

    def _node_to_location(self, node) -> Location:
        """Convert tree-sitter node position to Location object."""
        return Location(
            file_path=self._current_file,
            line_start=node.start_point[0] + 1,  # Convert to 1-based
            column_start=node.start_point[1],
            line_end=node.end_point[0] + 1,
            column_end=node.end_point[1],
        )

    def _get_node_text(self, node, content_bytes: bytes) -> str:
        """Extract text content from a tree-sitter node."""
        return content_bytes[node.start_byte:node.end_byte].decode("utf-8")

    def _find_child_by_type(self, node, node_type: str):
        """Find first child node of specified type."""
        for child in node.children:
            if child.type == node_type:
                return child
        return None

    def _find_children_by_type(self, node, node_type: str) -> List:
        """Find all child nodes of specified type."""
        return [child for child in node.children if child.type == node_type]

    def _is_external_import(self, module_path: str) -> bool:
        """
        Determine if import path refers to external dependency.
        
        Subclasses should override with language-specific logic.
        """
        return True

