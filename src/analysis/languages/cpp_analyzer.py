"""
C and C++ Analyzer Module.

Provides tree-sitter based static analysis for C and C++ source files.
Extracts classes, structs, functions, and includes with accurate
AST-based parsing.
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
    import tree_sitter_cpp as ts_cpp
    import tree_sitter_c as ts_c
    from tree_sitter import Language, Parser
    TREE_SITTER_CPP_AVAILABLE = True
    TREE_SITTER_C_AVAILABLE = True
except ImportError:
    TREE_SITTER_CPP_AVAILABLE = False
    TREE_SITTER_C_AVAILABLE = False
    logger.warning("tree-sitter-cpp/c not available, using regex fallback")


@AnalyzerRegistry.register
class CppAnalyzer(BaseTreeSitterAnalyzer):
    """
    Tree-sitter based analyzer for C++ source code.
    
    Extracts:
        - Classes, structs, and namespaces
        - Functions and methods
        - Include directives
        - Template declarations
    """

    LANGUAGE = "cpp"
    SUPPORTED_EXTENSIONS = [".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".h++"]

    CLASS_NODE_TYPES = ["class_specifier", "struct_specifier"]
    FUNCTION_NODE_TYPES = ["function_definition", "function_declaration"]
    IMPORT_NODE_TYPES = ["preproc_include"]

    def __init__(self):
        super().__init__()
        self._ts_available = TREE_SITTER_CPP_AVAILABLE

    def _initialize_parser(self) -> bool:
        """Initialize tree-sitter parser with C++ grammar."""
        if not self._ts_available:
            return False

        try:
            self._language = Language(ts_cpp.language())
            self._parser = Parser(self._language)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize C++ parser: {e}")
            return False

    def _extract_classes(self, tree, content_bytes: bytes) -> List[ClassEntity]:
        """Extract class and struct definitions."""
        classes = []
        
        def visit_node(node, namespace: str = ""):
            current_ns = namespace
            
            # Track namespace context
            if node.type == "namespace_definition":
                name_node = self._find_child_by_type(node, "namespace_identifier")
                if name_node:
                    ns_name = self._get_node_text(name_node, content_bytes)
                    current_ns = f"{namespace}::{ns_name}" if namespace else ns_name

            if node.type in ("class_specifier", "struct_specifier"):
                name_node = self._find_child_by_type(node, "type_identifier")
                if name_node:
                    name = self._get_node_text(name_node, content_bytes)
                    kind = "class" if node.type == "class_specifier" else "struct"
                    
                    # Extract base classes
                    parent_classes = []
                    base_list = self._find_child_by_type(node, "base_class_clause")
                    if base_list:
                        for child in base_list.children:
                            if child.type == "type_identifier":
                                parent_classes.append(
                                    self._get_node_text(child, content_bytes)
                                )

                    qualified = f"{current_ns}::{name}" if current_ns else name
                    if not current_ns:
                        qualified = f"{self._current_module}::{name}"

                    classes.append(ClassEntity(
                        name=name,
                        qualified_name=qualified,
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        parent_classes=parent_classes,
                        attributes={"kind": kind, "namespace": current_ns},
                    ))

            for child in node.children:
                visit_node(child, current_ns)

        visit_node(tree.root_node)
        return classes

    def _extract_functions(self, tree, content_bytes: bytes) -> List[FunctionEntity]:
        """Extract function definitions and declarations."""
        functions = []
        
        def visit_node(node, class_name: Optional[str] = None, namespace: str = ""):
            current_class = class_name
            current_ns = namespace
            
            # Track namespace
            if node.type == "namespace_definition":
                name_node = self._find_child_by_type(node, "namespace_identifier")
                if name_node:
                    ns_name = self._get_node_text(name_node, content_bytes)
                    current_ns = f"{namespace}::{ns_name}" if namespace else ns_name

            # Track class context
            if node.type in ("class_specifier", "struct_specifier"):
                name_node = self._find_child_by_type(node, "type_identifier")
                if name_node:
                    current_class = self._get_node_text(name_node, content_bytes)

            if node.type == "function_definition":
                func_info = self._extract_function_info(node, content_bytes)
                if func_info:
                    name, params, return_type, is_static, is_virtual = func_info
                    
                    # Determine if method
                    is_method = current_class is not None
                    
                    # Build qualified name
                    if current_class:
                        qualified = f"{current_ns}::{current_class}::{name}" if current_ns else f"{self._current_module}::{current_class}::{name}"
                    else:
                        qualified = f"{current_ns}::{name}" if current_ns else f"{self._current_module}::{name}"

                    functions.append(FunctionEntity(
                        name=name,
                        qualified_name=qualified,
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        is_method=is_method,
                        is_static=is_static,
                        parent_class=current_class,
                        parameters=params,
                        return_type=return_type,
                        attributes={"is_virtual": is_virtual, "namespace": current_ns},
                    ))

            for child in node.children:
                visit_node(child, current_class, current_ns)

        visit_node(tree.root_node)
        return functions

    def _extract_function_info(self, node, content_bytes: bytes):
        """Extract function name, parameters, and return type."""
        name = None
        params = []
        return_type = None
        is_static = False
        is_virtual = False
        
        # Get declarator
        declarator = self._find_child_by_type(node, "function_declarator")
        if not declarator:
            # Could be in a different structure
            for child in node.children:
                if child.type in ("function_declarator", "pointer_declarator"):
                    declarator = child
                    break
        
        if declarator:
            # Get function name
            name_node = None
            for child in declarator.children:
                if child.type == "identifier":
                    name_node = child
                    break
                elif child.type == "field_identifier":
                    name_node = child
                    break
                elif child.type == "qualified_identifier":
                    # Get the last identifier in qualified name
                    for sub in child.children:
                        if sub.type == "identifier":
                            name_node = sub
            
            if name_node:
                name = self._get_node_text(name_node, content_bytes)
            
            # Get parameters
            param_list = self._find_child_by_type(declarator, "parameter_list")
            if param_list:
                for child in param_list.children:
                    if child.type == "parameter_declaration":
                        # Get parameter name
                        for sub in child.children:
                            if sub.type in ("identifier", "pointer_declarator"):
                                params.append(self._get_node_text(sub, content_bytes))
                                break
        
        # Get return type (first type specifier before declarator)
        for child in node.children:
            if child.type in ("type_identifier", "primitive_type", 
                             "sized_type_specifier", "template_type"):
                return_type = self._get_node_text(child, content_bytes)
                break
        
        # Check for static/virtual
        for child in node.children:
            text = self._get_node_text(child, content_bytes)
            if text == "static":
                is_static = True
            if text == "virtual":
                is_virtual = True
        
        if name:
            return (name, params, return_type, is_static, is_virtual)
        return None

    def _extract_imports(self, tree, content_bytes: bytes) -> List[ImportEntity]:
        """Extract include directives."""
        imports = []
        
        def visit_node(node):
            if node.type == "preproc_include":
                # Get the path (system_lib_string or string_literal)
                path_node = None
                for child in node.children:
                    if child.type in ("system_lib_string", "string_literal"):
                        path_node = child
                        break
                
                if path_node:
                    path = self._get_node_text(path_node, content_bytes)
                    # Remove < > or " "
                    path = path.strip('<>"')
                    is_system = path_node.type == "system_lib_string"
                    
                    imports.append(ImportEntity(
                        name=path.split("/")[-1].split(".")[0],
                        qualified_name=f"{self._current_module}:include:{path}",
                        language=self.LANGUAGE,
                        module_path=path,
                        is_external=self._is_external_import(path, is_system),
                        location=self._node_to_location(node),
                        attributes={"is_system": is_system},
                    ))

            for child in node.children:
                visit_node(child)

        visit_node(tree.root_node)
        return imports

    def _is_external_import(self, path: str, is_system: bool = False) -> bool:
        """Determine if include refers to external/standard library."""
        if is_system:
            return False
        
        # C++ standard library headers (no extension)
        stdlib_headers = {
            "algorithm", "array", "atomic", "bitset", "cassert", "cctype",
            "cerrno", "cfenv", "cfloat", "chrono", "cinttypes", "climits",
            "clocale", "cmath", "complex", "condition_variable", "csetjmp",
            "csignal", "cstdarg", "cstddef", "cstdint", "cstdio", "cstdlib",
            "cstring", "ctime", "cwchar", "cwctype", "deque", "exception",
            "filesystem", "format", "forward_list", "fstream", "functional",
            "future", "initializer_list", "iomanip", "ios", "iosfwd",
            "iostream", "istream", "iterator", "limits", "list", "locale",
            "map", "memory", "mutex", "new", "numeric", "optional", "ostream",
            "queue", "random", "ratio", "regex", "scoped_allocator", "set",
            "shared_mutex", "span", "sstream", "stack", "stdexcept", "streambuf",
            "string", "string_view", "system_error", "thread", "tuple",
            "type_traits", "typeindex", "typeinfo", "unordered_map",
            "unordered_set", "utility", "valarray", "variant", "vector",
        }
        
        header_name = path.split("/")[-1].replace(".h", "").replace(".hpp", "")
        return header_name not in stdlib_headers

    def _get_call_target(self, node, content_bytes: bytes) -> Optional[str]:
        """Extract function name from call expression."""
        if node.type == "call_expression":
            func = node.children[0] if node.children else None
            if func:
                if func.type == "identifier":
                    return self._get_node_text(func, content_bytes)
                elif func.type == "field_expression":
                    field = self._find_child_by_type(func, "field_identifier")
                    if field:
                        return self._get_node_text(field, content_bytes)
        return None


@AnalyzerRegistry.register
class CAnalyzer(CppAnalyzer):
    """
    Tree-sitter based analyzer for C source code.
    
    Extends C++ analyzer with C-specific grammar.
    """

    LANGUAGE = "c"
    SUPPORTED_EXTENSIONS = [".c", ".h"]

    def __init__(self):
        super().__init__()
        self._ts_available = TREE_SITTER_C_AVAILABLE

    def _initialize_parser(self) -> bool:
        """Initialize tree-sitter parser with C grammar."""
        if not self._ts_available:
            return False

        try:
            self._language = Language(ts_c.language())
            self._parser = Parser(self._language)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize C parser: {e}")
            return False
