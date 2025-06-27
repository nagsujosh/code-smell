"""
Java Analyzer Module.

Provides tree-sitter based static analysis for Java source files.
Extracts classes, interfaces, methods, and import statements with
accurate AST-based parsing.
"""

import logging
from typing import List, Optional

from src.analysis.registry import AnalyzerRegistry
from src.analysis.languages.base_treesitter_analyzer import BaseTreeSitterAnalyzer
from src.analysis.entities import (
    ClassEntity,
    FunctionEntity,
    ImportEntity,
    Location,
)

logger = logging.getLogger(__name__)

# Tree-sitter imports with fallback handling
try:
    import tree_sitter_java as ts_java
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning("tree-sitter-java not available, using regex fallback")


@AnalyzerRegistry.register
class JavaAnalyzer(BaseTreeSitterAnalyzer):
    """
    Tree-sitter based analyzer for Java source code.
    
    Extracts:
        - Classes, interfaces, enums, and records
        - Methods and constructors
        - Import statements (standard and static)
        - Inheritance and implementation relationships
    """

    LANGUAGE = "java"
    SUPPORTED_EXTENSIONS = [".java"]

    CLASS_NODE_TYPES = [
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "record_declaration",
    ]
    FUNCTION_NODE_TYPES = ["method_declaration", "constructor_declaration"]
    IMPORT_NODE_TYPES = ["import_declaration"]

    def __init__(self):
        super().__init__()
        self._ts_available = TREE_SITTER_AVAILABLE

    def _initialize_parser(self) -> bool:
        """Initialize tree-sitter parser with Java grammar."""
        if not self._ts_available:
            return False

        try:
            self._language = Language(ts_java.language())
            self._parser = Parser(self._language)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Java parser: {e}")
            return False

    def _extract_classes(self, tree, content_bytes: bytes) -> List[ClassEntity]:
        """Extract class, interface, enum, and record declarations."""
        classes = []
        
        def visit_node(node, parent_class: Optional[str] = None):
            if node.type in self.CLASS_NODE_TYPES:
                name_node = self._find_child_by_type(node, "identifier")
                if name_node:
                    name = self._get_node_text(name_node, content_bytes)
                    
                    # Determine kind
                    kind = "class"
                    if node.type == "interface_declaration":
                        kind = "interface"
                    elif node.type == "enum_declaration":
                        kind = "enum"
                    elif node.type == "record_declaration":
                        kind = "record"

                    # Extract parent class (extends)
                    parent_classes = []
                    superclass = self._find_child_by_type(node, "superclass")
                    if superclass:
                        type_id = self._find_child_by_type(superclass, "type_identifier")
                        if type_id:
                            parent_classes.append(
                                self._get_node_text(type_id, content_bytes)
                            )

                    # Extract implemented interfaces
                    interfaces = self._find_child_by_type(node, "super_interfaces")
                    if interfaces:
                        type_list = self._find_child_by_type(interfaces, "type_list")
                        if type_list:
                            for child in type_list.children:
                                if child.type == "type_identifier":
                                    parent_classes.append(
                                        self._get_node_text(child, content_bytes)
                                    )

                    # Check modifiers
                    is_abstract = self._has_modifier(node, "abstract", content_bytes)
                    is_static = self._has_modifier(node, "static", content_bytes)

                    qualified = f"{self._current_module}.{name}"
                    if parent_class:
                        qualified = f"{self._current_module}.{parent_class}.{name}"

                    classes.append(ClassEntity(
                        name=name,
                        qualified_name=qualified,
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        parent_classes=parent_classes,
                        is_abstract=is_abstract,
                        attributes={"kind": kind, "is_static": is_static},
                    ))

                    # Process nested classes
                    for child in node.children:
                        visit_node(child, name)
                    return

            for child in node.children:
                visit_node(child, parent_class)

        visit_node(tree.root_node)
        return classes

    def _extract_functions(self, tree, content_bytes: bytes) -> List[FunctionEntity]:
        """Extract methods and constructors from Java source."""
        functions = []
        
        def visit_node(node, parent_class: Optional[str] = None):
            # Track class context
            current_class = parent_class
            if node.type in self.CLASS_NODE_TYPES:
                name_node = self._find_child_by_type(node, "identifier")
                if name_node:
                    current_class = self._get_node_text(name_node, content_bytes)

            # Method declaration
            if node.type == "method_declaration" and current_class:
                name_node = self._find_child_by_type(node, "identifier")
                if name_node:
                    name = self._get_node_text(name_node, content_bytes)
                    
                    # Extract modifiers
                    is_static = self._has_modifier(node, "static", content_bytes)
                    is_abstract = self._has_modifier(node, "abstract", content_bytes)
                    
                    # Extract return type
                    return_type = None
                    for child in node.children:
                        if child.type in ("type_identifier", "void_type", 
                                          "integral_type", "floating_point_type",
                                          "boolean_type", "generic_type"):
                            return_type = self._get_node_text(child, content_bytes)
                            break

                    # Extract parameters
                    params = self._extract_parameters(node, content_bytes)

                    functions.append(FunctionEntity(
                        name=name,
                        qualified_name=f"{self._current_module}.{current_class}.{name}",
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        is_method=True,
                        is_static=is_static,
                        parent_class=current_class,
                        return_type=return_type,
                        parameters=params,
                        attributes={"is_abstract": is_abstract},
                    ))

            # Constructor declaration
            elif node.type == "constructor_declaration" and current_class:
                name_node = self._find_child_by_type(node, "identifier")
                if name_node:
                    name = self._get_node_text(name_node, content_bytes)
                    params = self._extract_parameters(node, content_bytes)

                    functions.append(FunctionEntity(
                        name=name,
                        qualified_name=f"{self._current_module}.{current_class}.{name}",
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        is_method=True,
                        parent_class=current_class,
                        parameters=params,
                        attributes={"is_constructor": True},
                    ))

            for child in node.children:
                visit_node(child, current_class)

        visit_node(tree.root_node)
        return functions

    def _extract_imports(self, tree, content_bytes: bytes) -> List[ImportEntity]:
        """Extract import declarations from Java source."""
        imports = []
        
        def visit_node(node):
            if node.type == "import_declaration":
                # Check for static import
                is_static = any(
                    child.type == "static" for child in node.children
                )
                
                # Get the scoped identifier
                scoped = self._find_child_by_type(node, "scoped_identifier")
                if scoped:
                    module_path = self._get_node_text(scoped, content_bytes)
                else:
                    # Try identifier for single imports
                    ident = self._find_child_by_type(node, "identifier")
                    if ident:
                        module_path = self._get_node_text(ident, content_bytes)
                    else:
                        module_path = ""

                # Check for wildcard
                is_wildcard = any(
                    child.type == "asterisk" for child in node.children
                )
                if is_wildcard:
                    module_path += ".*"

                if module_path:
                    imports.append(ImportEntity(
                        name=module_path.split(".")[-1].replace("*", ""),
                        qualified_name=f"{self._current_module}:import:{module_path}",
                        language=self.LANGUAGE,
                        module_path=module_path,
                        is_external=self._is_external_import(module_path),
                        location=self._node_to_location(node),
                        attributes={"is_static": is_static, "is_wildcard": is_wildcard},
                    ))

            for child in node.children:
                visit_node(child)

        visit_node(tree.root_node)
        return imports

    def _has_modifier(self, node, modifier: str, content_bytes: bytes) -> bool:
        """Check if node has a specific modifier."""
        modifiers = self._find_child_by_type(node, "modifiers")
        if modifiers:
            for child in modifiers.children:
                if self._get_node_text(child, content_bytes) == modifier:
                    return True
        return False

    def _extract_parameters(self, node, content_bytes: bytes) -> List[str]:
        """Extract parameter list from method/constructor."""
        params = []
        formal_params = self._find_child_by_type(node, "formal_parameters")
        if formal_params:
            for child in formal_params.children:
                if child.type == "formal_parameter":
                    # Get parameter name
                    name_node = self._find_child_by_type(child, "identifier")
                    if name_node:
                        params.append(self._get_node_text(name_node, content_bytes))
        return params

    def _is_external_import(self, module_path: str) -> bool:
        """Determine if import refers to external package."""
        # Java standard library packages
        stdlib_prefixes = (
            "java.", "javax.", "sun.", "com.sun.", "org.w3c.", 
            "org.xml.", "org.ietf.", "jdk.",
        )
        return not module_path.startswith(stdlib_prefixes)

    def _get_call_target(self, node, content_bytes: bytes) -> Optional[str]:
        """Extract method name from method invocation."""
        if node.type == "method_invocation":
            name_node = self._find_child_by_type(node, "identifier")
            if name_node:
                return self._get_node_text(name_node, content_bytes)
        return None
