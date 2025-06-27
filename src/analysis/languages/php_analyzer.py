"""
PHP Analyzer Module.

Provides tree-sitter based static analysis for PHP source files.
Extracts classes, interfaces, traits, functions, and use statements
with accurate AST-based parsing.
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
    import tree_sitter_php as ts_php
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning("tree-sitter-php not available, using regex fallback")


@AnalyzerRegistry.register
class PHPAnalyzer(BaseTreeSitterAnalyzer):
    """
    Tree-sitter based analyzer for PHP source code.
    
    Extracts:
        - Classes, interfaces, traits, and enums
        - Functions and methods
        - use statements and namespace declarations
        - require/include statements
    """

    LANGUAGE = "php"
    SUPPORTED_EXTENSIONS = [".php", ".phtml", ".php5", ".php7"]

    CLASS_NODE_TYPES = [
        "class_declaration",
        "interface_declaration",
        "trait_declaration",
        "enum_declaration",
    ]
    FUNCTION_NODE_TYPES = ["function_definition", "method_declaration"]
    IMPORT_NODE_TYPES = ["use_declaration", "namespace_use_declaration"]

    def __init__(self):
        super().__init__()
        self._ts_available = TREE_SITTER_AVAILABLE
        self._current_namespace = ""

    def _initialize_parser(self) -> bool:
        """Initialize tree-sitter parser with PHP grammar."""
        if not self._ts_available:
            return False

        try:
            # PHP grammar provides language_php for PHP code
            self._language = Language(ts_php.language_php())
            self._parser = Parser(self._language)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize PHP parser: {e}")
            return False

    def _extract_classes(self, tree, content_bytes: bytes) -> List[ClassEntity]:
        """Extract class, interface, trait, and enum declarations."""
        classes = []
        
        def visit_node(node, namespace: str = ""):
            current_ns = namespace
            
            # Track namespace
            if node.type == "namespace_definition":
                name_node = self._find_child_by_type(node, "namespace_name")
                if name_node:
                    current_ns = self._get_node_text(name_node, content_bytes)

            if node.type == "class_declaration":
                name_node = self._find_child_by_type(node, "name")
                if name_node:
                    name = self._get_node_text(name_node, content_bytes)
                    
                    # Extract parent class
                    parent_classes = []
                    base_clause = self._find_child_by_type(node, "base_clause")
                    if base_clause:
                        parent_name = self._find_child_by_type(base_clause, "name")
                        if parent_name:
                            parent_classes.append(
                                self._get_node_text(parent_name, content_bytes)
                            )
                    
                    # Extract implemented interfaces
                    impl_clause = self._find_child_by_type(node, "class_interface_clause")
                    if impl_clause:
                        for child in impl_clause.children:
                            if child.type == "name":
                                parent_classes.append(
                                    self._get_node_text(child, content_bytes)
                                )

                    # Check modifiers
                    is_abstract = self._has_modifier(node, "abstract", content_bytes)
                    is_final = self._has_modifier(node, "final", content_bytes)

                    qualified = f"{current_ns}\\{name}" if current_ns else f"{self._current_module}\\{name}"

                    classes.append(ClassEntity(
                        name=name,
                        qualified_name=qualified,
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        parent_classes=parent_classes,
                        is_abstract=is_abstract,
                        attributes={"kind": "class", "is_final": is_final, "namespace": current_ns},
                    ))

            elif node.type == "interface_declaration":
                name_node = self._find_child_by_type(node, "name")
                if name_node:
                    name = self._get_node_text(name_node, content_bytes)
                    qualified = f"{current_ns}\\{name}" if current_ns else f"{self._current_module}\\{name}"

                    classes.append(ClassEntity(
                        name=name,
                        qualified_name=qualified,
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        attributes={"kind": "interface", "namespace": current_ns},
                    ))

            elif node.type == "trait_declaration":
                name_node = self._find_child_by_type(node, "name")
                if name_node:
                    name = self._get_node_text(name_node, content_bytes)
                    qualified = f"{current_ns}\\{name}" if current_ns else f"{self._current_module}\\{name}"

                    classes.append(ClassEntity(
                        name=name,
                        qualified_name=qualified,
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        attributes={"kind": "trait", "namespace": current_ns},
                    ))

            elif node.type == "enum_declaration":
                name_node = self._find_child_by_type(node, "name")
                if name_node:
                    name = self._get_node_text(name_node, content_bytes)
                    qualified = f"{current_ns}\\{name}" if current_ns else f"{self._current_module}\\{name}"

                    classes.append(ClassEntity(
                        name=name,
                        qualified_name=qualified,
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        attributes={"kind": "enum", "namespace": current_ns},
                    ))

            for child in node.children:
                visit_node(child, current_ns)

        visit_node(tree.root_node)
        return classes

    def _extract_functions(self, tree, content_bytes: bytes) -> List[FunctionEntity]:
        """Extract function and method definitions."""
        functions = []
        
        def visit_node(node, parent_class: Optional[str] = None, namespace: str = ""):
            current_class = parent_class
            current_ns = namespace
            
            # Track namespace
            if node.type == "namespace_definition":
                name_node = self._find_child_by_type(node, "namespace_name")
                if name_node:
                    current_ns = self._get_node_text(name_node, content_bytes)

            # Track class context
            if node.type in self.CLASS_NODE_TYPES:
                name_node = self._find_child_by_type(node, "name")
                if name_node:
                    current_class = self._get_node_text(name_node, content_bytes)

            # Standalone function
            if node.type == "function_definition":
                name_node = self._find_child_by_type(node, "name")
                if name_node:
                    name = self._get_node_text(name_node, content_bytes)
                    params = self._extract_php_parameters(node, content_bytes)
                    return_type = self._extract_return_type(node, content_bytes)

                    qualified = f"{current_ns}\\{name}" if current_ns else f"{self._current_module}\\{name}"

                    functions.append(FunctionEntity(
                        name=name,
                        qualified_name=qualified,
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        parameters=params,
                        return_type=return_type,
                        attributes={"namespace": current_ns},
                    ))

            # Class method
            elif node.type == "method_declaration" and current_class:
                name_node = self._find_child_by_type(node, "name")
                if name_node:
                    name = self._get_node_text(name_node, content_bytes)
                    
                    # Check modifiers
                    is_static = self._has_modifier(node, "static", content_bytes)
                    is_abstract = self._has_modifier(node, "abstract", content_bytes)
                    visibility = self._get_visibility(node, content_bytes)
                    
                    params = self._extract_php_parameters(node, content_bytes)
                    return_type = self._extract_return_type(node, content_bytes)

                    qualified = f"{current_ns}\\{current_class}::{name}" if current_ns else f"{self._current_module}\\{current_class}::{name}"

                    functions.append(FunctionEntity(
                        name=name,
                        qualified_name=qualified,
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        is_method=True,
                        is_static=is_static,
                        parent_class=current_class,
                        parameters=params,
                        return_type=return_type,
                        attributes={
                            "visibility": visibility,
                            "is_abstract": is_abstract,
                            "namespace": current_ns,
                        },
                    ))

            for child in node.children:
                visit_node(child, current_class, current_ns)

        visit_node(tree.root_node)
        return functions

    def _extract_imports(self, tree, content_bytes: bytes) -> List[ImportEntity]:
        """Extract use statements and require/include."""
        imports = []
        
        def visit_node(node):
            # Use declarations
            if node.type == "namespace_use_declaration":
                for child in node.children:
                    if child.type == "namespace_use_clause":
                        name_node = self._find_child_by_type(child, "qualified_name")
                        if name_node:
                            path = self._get_node_text(name_node, content_bytes)
                            
                            # Check for alias
                            alias = None
                            alias_node = self._find_child_by_type(
                                child, "namespace_aliasing_clause"
                            )
                            if alias_node:
                                alias_name = self._find_child_by_type(alias_node, "name")
                                if alias_name:
                                    alias = self._get_node_text(alias_name, content_bytes)

                            imports.append(ImportEntity(
                                name=path.split("\\")[-1],
                                qualified_name=f"{self._current_module}:use:{path}",
                                language=self.LANGUAGE,
                                module_path=path,
                                is_external=self._is_external_import(path),
                                location=self._node_to_location(node),
                                attributes={"alias": alias} if alias else {},
                            ))

            # Require/include statements
            elif node.type in ("include_expression", "include_once_expression",
                              "require_expression", "require_once_expression"):
                # Get the file path argument
                for child in node.children:
                    if child.type == "string":
                        string_content = self._find_child_by_type(child, "string_content")
                        if string_content:
                            path = self._get_node_text(string_content, content_bytes)
                        else:
                            path = self._get_node_text(child, content_bytes).strip("'\"")
                        
                        imports.append(ImportEntity(
                            name=path.split("/")[-1],
                            qualified_name=f"{self._current_module}:require:{path}",
                            language=self.LANGUAGE,
                            module_path=path,
                            is_external=False,  # File includes are typically internal
                            location=self._node_to_location(node),
                            attributes={"kind": node.type},
                        ))
                        break

            for child in node.children:
                visit_node(child)

        visit_node(tree.root_node)
        return imports

    def _extract_php_parameters(self, node, content_bytes: bytes) -> List[str]:
        """Extract parameters from function/method definition."""
        params = []
        param_list = self._find_child_by_type(node, "formal_parameters")
        if param_list:
            for child in param_list.children:
                if child.type in ("simple_parameter", "variadic_parameter",
                                  "property_promotion_parameter"):
                    var_node = self._find_child_by_type(child, "variable_name")
                    if var_node:
                        params.append(self._get_node_text(var_node, content_bytes))
        return params

    def _extract_return_type(self, node, content_bytes: bytes) -> Optional[str]:
        """Extract return type from function signature."""
        return_type = self._find_child_by_type(node, "return_type")
        if return_type:
            # Get the actual type within return_type node
            for child in return_type.children:
                if child.type in ("name", "primitive_type", "nullable_type",
                                  "union_type", "intersection_type"):
                    return self._get_node_text(child, content_bytes)
        return None

    def _has_modifier(self, node, modifier: str, content_bytes: bytes) -> bool:
        """Check if node has a specific modifier."""
        for child in node.children:
            if child.type == "visibility_modifier" or child.type == modifier:
                text = self._get_node_text(child, content_bytes)
                if text == modifier:
                    return True
        return False

    def _get_visibility(self, node, content_bytes: bytes) -> str:
        """Get visibility modifier (public, protected, private)."""
        for child in node.children:
            if child.type == "visibility_modifier":
                return self._get_node_text(child, content_bytes)
        return "public"  # Default in PHP

    def _is_external_import(self, path: str) -> bool:
        """Determine if use statement refers to external package."""
        # Common framework namespaces indicate external packages
        external_prefixes = (
            "Symfony\\", "Laravel\\", "Illuminate\\", "Doctrine\\",
            "Guzzle\\", "GuzzleHttp\\", "Monolog\\", "PHPUnit\\",
            "Psr\\", "Twig\\", "Carbon\\", "League\\",
        )
        
        return path.startswith(external_prefixes)

    def _get_call_target(self, node, content_bytes: bytes) -> Optional[str]:
        """Extract function/method name from call expression."""
        if node.type == "function_call_expression":
            func = node.children[0] if node.children else None
            if func:
                if func.type == "name":
                    return self._get_node_text(func, content_bytes)
        elif node.type == "method_call_expression":
            name_node = self._find_child_by_type(node, "name")
            if name_node:
                return self._get_node_text(name_node, content_bytes)
        return None

