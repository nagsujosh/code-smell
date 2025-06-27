"""
Go Analyzer Module.

Provides tree-sitter based static analysis for Go source files.
Extracts structs, interfaces, functions, methods, and imports
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
    import tree_sitter_go as ts_go
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning("tree-sitter-go not available, using regex fallback")


@AnalyzerRegistry.register
class GoAnalyzer(BaseTreeSitterAnalyzer):
    """
    Tree-sitter based analyzer for Go source code.
    
    Extracts:
        - Structs and interfaces (as class entities)
        - Functions and methods
        - Import declarations
        - Type definitions
    """

    LANGUAGE = "go"
    SUPPORTED_EXTENSIONS = [".go"]

    CLASS_NODE_TYPES = ["type_declaration"]
    FUNCTION_NODE_TYPES = ["function_declaration", "method_declaration"]
    IMPORT_NODE_TYPES = ["import_declaration"]

    def __init__(self):
        super().__init__()
        self._ts_available = TREE_SITTER_AVAILABLE

    def _initialize_parser(self) -> bool:
        """Initialize tree-sitter parser with Go grammar."""
        if not self._ts_available:
            return False

        try:
            self._language = Language(ts_go.language())
            self._parser = Parser(self._language)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Go parser: {e}")
            return False

    def _extract_classes(self, tree, content_bytes: bytes) -> List[ClassEntity]:
        """Extract struct and interface type declarations."""
        classes = []
        
        def visit_node(node):
            if node.type == "type_declaration":
                # Get type specs within the declaration
                for child in node.children:
                    if child.type == "type_spec":
                        name_node = self._find_child_by_type(child, "type_identifier")
                        if not name_node:
                            continue
                        
                        name = self._get_node_text(name_node, content_bytes)
                        
                        # Determine type kind (struct, interface, or alias)
                        kind = "type_alias"
                        for type_child in child.children:
                            if type_child.type == "struct_type":
                                kind = "struct"
                                break
                            elif type_child.type == "interface_type":
                                kind = "interface"
                                break

                        classes.append(ClassEntity(
                            name=name,
                            qualified_name=f"{self._current_module}.{name}",
                            language=self.LANGUAGE,
                            location=self._node_to_location(child),
                            attributes={"kind": kind},
                        ))

            for child in node.children:
                visit_node(child)

        visit_node(tree.root_node)
        return classes

    def _extract_functions(self, tree, content_bytes: bytes) -> List[FunctionEntity]:
        """Extract functions and methods from Go source."""
        functions = []
        
        def visit_node(node):
            # Function declaration
            if node.type == "function_declaration":
                name_node = self._find_child_by_type(node, "identifier")
                if name_node:
                    name = self._get_node_text(name_node, content_bytes)
                    
                    # Extract parameters
                    params = self._extract_go_parameters(node, content_bytes)
                    
                    # Extract return type
                    return_type = self._extract_return_type(node, content_bytes)

                    functions.append(FunctionEntity(
                        name=name,
                        qualified_name=f"{self._current_module}.{name}",
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        parameters=params,
                        return_type=return_type,
                    ))

            # Method declaration (function with receiver)
            elif node.type == "method_declaration":
                name_node = self._find_child_by_type(node, "field_identifier")
                if name_node:
                    name = self._get_node_text(name_node, content_bytes)
                    
                    # Extract receiver type (the struct this method belongs to)
                    receiver_type = None
                    params_node = self._find_child_by_type(node, "parameter_list")
                    if params_node:
                        # First parameter list is the receiver
                        for child in params_node.children:
                            if child.type == "parameter_declaration":
                                type_node = self._find_child_by_type(
                                    child, "type_identifier"
                                )
                                if type_node:
                                    receiver_type = self._get_node_text(
                                        type_node, content_bytes
                                    )
                                else:
                                    # Could be pointer receiver
                                    pointer = self._find_child_by_type(
                                        child, "pointer_type"
                                    )
                                    if pointer:
                                        type_id = self._find_child_by_type(
                                            pointer, "type_identifier"
                                        )
                                        if type_id:
                                            receiver_type = self._get_node_text(
                                                type_id, content_bytes
                                            )
                                break

                    # Extract parameters (skip receiver)
                    params = self._extract_go_parameters(node, content_bytes, skip_first=True)
                    return_type = self._extract_return_type(node, content_bytes)

                    qualified = f"{self._current_module}.{name}"
                    if receiver_type:
                        qualified = f"{self._current_module}.{receiver_type}.{name}"

                    functions.append(FunctionEntity(
                        name=name,
                        qualified_name=qualified,
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        is_method=True,
                        parent_class=receiver_type,
                        parameters=params,
                        return_type=return_type,
                    ))

            for child in node.children:
                visit_node(child)

        visit_node(tree.root_node)
        return functions

    def _extract_imports(self, tree, content_bytes: bytes) -> List[ImportEntity]:
        """Extract import declarations from Go source."""
        imports = []
        
        def visit_node(node):
            if node.type == "import_declaration":
                # Handle import spec list or single import
                for child in node.children:
                    if child.type == "import_spec_list":
                        for spec in child.children:
                            if spec.type == "import_spec":
                                self._process_import_spec(spec, content_bytes, imports)
                    elif child.type == "import_spec":
                        self._process_import_spec(child, content_bytes, imports)

            for child in node.children:
                visit_node(child)

        visit_node(tree.root_node)
        return imports

    def _process_import_spec(
        self, 
        spec_node, 
        content_bytes: bytes, 
        imports: List[ImportEntity]
    ) -> None:
        """Process a single import spec node."""
        # Get the path (interpreted string literal)
        path_node = self._find_child_by_type(spec_node, "interpreted_string_literal")
        if path_node:
            module_path = self._get_node_text(path_node, content_bytes)
            module_path = module_path.strip('"')
            
            # Get alias if present
            alias = None
            name_node = self._find_child_by_type(spec_node, "package_identifier")
            if name_node:
                alias = self._get_node_text(name_node, content_bytes)
            
            # Determine package name from path
            pkg_name = module_path.split("/")[-1]
            if alias:
                pkg_name = alias

            imports.append(ImportEntity(
                name=pkg_name,
                qualified_name=f"{self._current_module}:import:{module_path}",
                language=self.LANGUAGE,
                module_path=module_path,
                is_external=self._is_external_import(module_path),
                location=self._node_to_location(spec_node),
                attributes={"alias": alias} if alias else {},
            ))

    def _extract_go_parameters(
        self, 
        node, 
        content_bytes: bytes, 
        skip_first: bool = False
    ) -> List[str]:
        """Extract parameters from function/method declaration."""
        params = []
        param_lists = self._find_children_by_type(node, "parameter_list")
        
        # For methods, first param list is receiver; function params are second
        start_idx = 1 if skip_first and len(param_lists) > 1 else 0
        
        for i, param_list in enumerate(param_lists):
            if i < start_idx:
                continue
            for child in param_list.children:
                if child.type == "parameter_declaration":
                    # Get parameter name(s)
                    for name_child in child.children:
                        if name_child.type == "identifier":
                            params.append(
                                self._get_node_text(name_child, content_bytes)
                            )
        return params

    def _extract_return_type(self, node, content_bytes: bytes) -> Optional[str]:
        """Extract return type from function/method."""
        # Look for result type after parameter list
        result = self._find_child_by_type(node, "type_identifier")
        if result:
            return self._get_node_text(result, content_bytes)
        
        # Check for parameter list as return (multiple returns)
        param_lists = self._find_children_by_type(node, "parameter_list")
        if len(param_lists) >= 2:
            # Last param list might be return types
            last = param_lists[-1]
            types = []
            for child in last.children:
                if child.type in ("type_identifier", "pointer_type", "slice_type"):
                    types.append(self._get_node_text(child, content_bytes))
            if types:
                return ", ".join(types)
        
        return None

    def _is_external_import(self, module_path: str) -> bool:
        """Determine if import refers to external package."""
        # Standard library packages are typically single-word or common prefixes
        stdlib_prefixes = (
            "archive", "bufio", "bytes", "compress", "container", "context",
            "crypto", "database", "debug", "embed", "encoding", "errors",
            "expvar", "flag", "fmt", "go", "hash", "html", "image", "index",
            "io", "log", "maps", "math", "mime", "net", "os", "path",
            "plugin", "reflect", "regexp", "runtime", "slices", "sort",
            "strconv", "strings", "sync", "syscall", "testing", "text",
            "time", "unicode", "unsafe",
        )
        
        first_part = module_path.split("/")[0]
        return first_part not in stdlib_prefixes

    def _get_call_target(self, node, content_bytes: bytes) -> Optional[str]:
        """Extract function name from call expression."""
        if node.type == "call_expression":
            func = node.children[0] if node.children else None
            if func:
                if func.type == "identifier":
                    return self._get_node_text(func, content_bytes)
                elif func.type == "selector_expression":
                    # Get the field (method name)
                    field = self._find_child_by_type(func, "field_identifier")
                    if field:
                        return self._get_node_text(field, content_bytes)
        return None
