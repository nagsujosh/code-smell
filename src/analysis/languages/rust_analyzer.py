"""
Rust Analyzer Module.

Provides tree-sitter based static analysis for Rust source files.
Extracts structs, enums, traits, functions, impl blocks, and use
statements with accurate AST-based parsing.
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
    import tree_sitter_rust as ts_rust
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning("tree-sitter-rust not available, using regex fallback")


@AnalyzerRegistry.register
class RustAnalyzer(BaseTreeSitterAnalyzer):
    """
    Tree-sitter based analyzer for Rust source code.
    
    Extracts:
        - Structs, enums, and traits (as class entities)
        - Functions and methods (including impl blocks)
        - Use statements and extern crates
        - Module declarations
    """

    LANGUAGE = "rust"
    SUPPORTED_EXTENSIONS = [".rs"]

    CLASS_NODE_TYPES = [
        "struct_item", 
        "enum_item", 
        "trait_item",
        "type_item",
    ]
    FUNCTION_NODE_TYPES = ["function_item"]
    IMPORT_NODE_TYPES = ["use_declaration", "extern_crate_declaration"]

    def __init__(self):
        super().__init__()
        self._ts_available = TREE_SITTER_AVAILABLE

    def _initialize_parser(self) -> bool:
        """Initialize tree-sitter parser with Rust grammar."""
        if not self._ts_available:
            return False

        try:
            self._language = Language(ts_rust.language())
            self._parser = Parser(self._language)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Rust parser: {e}")
            return False

    def _extract_classes(self, tree, content_bytes: bytes) -> List[ClassEntity]:
        """Extract struct, enum, and trait definitions."""
        classes = []
        
        def visit_node(node):
            if node.type == "struct_item":
                name_node = self._find_child_by_type(node, "type_identifier")
                if name_node:
                    name = self._get_node_text(name_node, content_bytes)
                    is_pub = self._is_public(node, content_bytes)
                    
                    classes.append(ClassEntity(
                        name=name,
                        qualified_name=f"{self._current_module}::{name}",
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        attributes={"kind": "struct", "is_public": is_pub},
                    ))

            elif node.type == "enum_item":
                name_node = self._find_child_by_type(node, "type_identifier")
                if name_node:
                    name = self._get_node_text(name_node, content_bytes)
                    is_pub = self._is_public(node, content_bytes)
                    
                    classes.append(ClassEntity(
                        name=name,
                        qualified_name=f"{self._current_module}::{name}",
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        attributes={"kind": "enum", "is_public": is_pub},
                    ))

            elif node.type == "trait_item":
                name_node = self._find_child_by_type(node, "type_identifier")
                if name_node:
                    name = self._get_node_text(name_node, content_bytes)
                    is_pub = self._is_public(node, content_bytes)
                    
                    classes.append(ClassEntity(
                        name=name,
                        qualified_name=f"{self._current_module}::{name}",
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        attributes={"kind": "trait", "is_public": is_pub},
                    ))

            elif node.type == "type_item":
                name_node = self._find_child_by_type(node, "type_identifier")
                if name_node:
                    name = self._get_node_text(name_node, content_bytes)
                    is_pub = self._is_public(node, content_bytes)
                    
                    classes.append(ClassEntity(
                        name=name,
                        qualified_name=f"{self._current_module}::{name}",
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        attributes={"kind": "type_alias", "is_public": is_pub},
                    ))

            for child in node.children:
                visit_node(child)

        visit_node(tree.root_node)
        return classes

    def _extract_functions(self, tree, content_bytes: bytes) -> List[FunctionEntity]:
        """Extract functions and methods from Rust source."""
        functions = []
        
        def visit_node(node, impl_type: Optional[str] = None):
            # Track impl block context
            current_impl = impl_type
            if node.type == "impl_item":
                # Get the type being implemented
                type_node = self._find_child_by_type(node, "type_identifier")
                if type_node:
                    current_impl = self._get_node_text(type_node, content_bytes)
                else:
                    # Could be generic type
                    generic = self._find_child_by_type(node, "generic_type")
                    if generic:
                        type_id = self._find_child_by_type(generic, "type_identifier")
                        if type_id:
                            current_impl = self._get_node_text(type_id, content_bytes)

            if node.type == "function_item":
                name_node = self._find_child_by_type(node, "identifier")
                if name_node:
                    name = self._get_node_text(name_node, content_bytes)
                    
                    # Check modifiers
                    is_pub = self._is_public(node, content_bytes)
                    is_async = self._has_keyword(node, "async", content_bytes)
                    is_unsafe = self._has_keyword(node, "unsafe", content_bytes)
                    is_const = self._has_keyword(node, "const", content_bytes)
                    
                    # Extract parameters
                    params = self._extract_rust_parameters(node, content_bytes)
                    
                    # Check if it's a method (has self parameter)
                    is_method = current_impl is not None
                    
                    # Extract return type
                    return_type = self._extract_return_type(node, content_bytes)

                    qualified = f"{self._current_module}::{name}"
                    if current_impl:
                        qualified = f"{self._current_module}::{current_impl}::{name}"

                    functions.append(FunctionEntity(
                        name=name,
                        qualified_name=qualified,
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        is_method=is_method,
                        is_async=is_async,
                        parent_class=current_impl,
                        parameters=params,
                        return_type=return_type,
                        attributes={
                            "is_public": is_pub,
                            "is_unsafe": is_unsafe,
                            "is_const": is_const,
                        },
                    ))

            for child in node.children:
                visit_node(child, current_impl)

        visit_node(tree.root_node)
        return functions

    def _extract_imports(self, tree, content_bytes: bytes) -> List[ImportEntity]:
        """Extract use declarations and extern crates."""
        imports = []
        
        def visit_node(node):
            if node.type == "use_declaration":
                # Get the use path
                use_tree = None
                for child in node.children:
                    if child.type in ("use_tree", "scoped_identifier", "identifier"):
                        use_tree = child
                        break
                
                if use_tree:
                    paths = self._extract_use_paths(use_tree, content_bytes)
                    for path in paths:
                        imports.append(ImportEntity(
                            name=path.split("::")[-1],
                            qualified_name=f"{self._current_module}:use:{path}",
                            language=self.LANGUAGE,
                            module_path=path,
                            is_external=self._is_external_import(path),
                            location=self._node_to_location(node),
                        ))

            elif node.type == "extern_crate_declaration":
                crate_node = self._find_child_by_type(node, "identifier")
                if crate_node:
                    crate_name = self._get_node_text(crate_node, content_bytes)
                    imports.append(ImportEntity(
                        name=crate_name,
                        qualified_name=f"{self._current_module}:extern:{crate_name}",
                        language=self.LANGUAGE,
                        module_path=crate_name,
                        is_external=True,
                        location=self._node_to_location(node),
                        attributes={"is_extern_crate": True},
                    ))

            for child in node.children:
                visit_node(child)

        visit_node(tree.root_node)
        return imports

    def _extract_use_paths(self, node, content_bytes: bytes) -> List[str]:
        """Recursively extract all paths from use tree."""
        paths = []
        
        if node.type == "identifier":
            return [self._get_node_text(node, content_bytes)]
        
        if node.type == "scoped_identifier":
            return [self._get_node_text(node, content_bytes).replace(" ", "")]
        
        if node.type == "use_tree":
            # Check for use list (e.g., use foo::{bar, baz})
            use_list = self._find_child_by_type(node, "use_tree_list")
            if use_list:
                # Get prefix path
                prefix = ""
                scoped = self._find_child_by_type(node, "scoped_identifier")
                if scoped:
                    prefix = self._get_node_text(scoped, content_bytes) + "::"
                
                for child in use_list.children:
                    if child.type == "use_tree":
                        sub_paths = self._extract_use_paths(child, content_bytes)
                        for sp in sub_paths:
                            paths.append(prefix + sp)
                    elif child.type == "identifier":
                        paths.append(prefix + self._get_node_text(child, content_bytes))
            else:
                # Single path
                scoped = self._find_child_by_type(node, "scoped_identifier")
                if scoped:
                    paths.append(self._get_node_text(scoped, content_bytes))
                else:
                    ident = self._find_child_by_type(node, "identifier")
                    if ident:
                        paths.append(self._get_node_text(ident, content_bytes))
        
        return paths

    def _extract_rust_parameters(self, node, content_bytes: bytes) -> List[str]:
        """Extract parameters from function definition."""
        params = []
        param_list = self._find_child_by_type(node, "parameters")
        if param_list:
            for child in param_list.children:
                if child.type == "parameter":
                    # Get parameter name (pattern)
                    pattern = self._find_child_by_type(child, "identifier")
                    if pattern:
                        params.append(self._get_node_text(pattern, content_bytes))
                elif child.type in ("self_parameter", "self"):
                    params.append("self")
        return params

    def _extract_return_type(self, node, content_bytes: bytes) -> Optional[str]:
        """Extract return type from function signature."""
        # Look for -> type
        for i, child in enumerate(node.children):
            if child.type == "->":
                # Next child should be the return type
                if i + 1 < len(node.children):
                    return_type = node.children[i + 1]
                    return self._get_node_text(return_type, content_bytes)
        return None

    def _is_public(self, node, content_bytes: bytes) -> bool:
        """Check if item has pub visibility."""
        vis = self._find_child_by_type(node, "visibility_modifier")
        if vis:
            text = self._get_node_text(vis, content_bytes)
            return text.startswith("pub")
        return False

    def _has_keyword(self, node, keyword: str, content_bytes: bytes) -> bool:
        """Check if node contains a specific keyword."""
        for child in node.children:
            if child.type == keyword:
                return True
            if self._get_node_text(child, content_bytes) == keyword:
                return True
        return False

    def _is_external_import(self, module_path: str) -> bool:
        """Determine if use path refers to external crate."""
        # Standard library paths
        stdlib_prefixes = ("std", "core", "alloc", "proc_macro", "test")
        
        first_part = module_path.split("::")[0]
        
        # Internal paths
        if first_part in ("self", "super", "crate"):
            return False
        
        return first_part not in stdlib_prefixes
