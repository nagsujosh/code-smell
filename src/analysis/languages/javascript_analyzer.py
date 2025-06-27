"""
JavaScript and TypeScript Analyzer Module.

Provides tree-sitter based static analysis for JavaScript and TypeScript
source files. Extracts classes, functions, imports, and relationships
with high accuracy using proper AST parsing.
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
    import tree_sitter_javascript as ts_javascript
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning("tree-sitter-javascript not available, using regex fallback")


@AnalyzerRegistry.register
class JavaScriptAnalyzer(BaseTreeSitterAnalyzer):
    """
    Tree-sitter based analyzer for JavaScript source code.
    
    Extracts:
        - Classes and their methods
        - Functions (declarations, expressions, arrow functions)
        - Import and require statements
        - Export declarations
        - Call relationships
    """

    LANGUAGE = "javascript"
    SUPPORTED_EXTENSIONS = [".js", ".jsx", ".mjs", ".cjs"]

    CLASS_NODE_TYPES = ["class_declaration", "class"]
    FUNCTION_NODE_TYPES = [
        "function_declaration",
        "function_expression",
        "arrow_function",
        "method_definition",
    ]
    IMPORT_NODE_TYPES = ["import_statement", "call_expression"]

    def __init__(self):
        super().__init__()
        self._ts_available = TREE_SITTER_AVAILABLE

    def _initialize_parser(self) -> bool:
        """Initialize tree-sitter parser with JavaScript grammar."""
        if not self._ts_available:
            return False

        try:
            self._language = Language(ts_javascript.language())
            self._parser = Parser(self._language)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize JavaScript parser: {e}")
            return False

    def _extract_classes(self, tree, content_bytes: bytes) -> List[ClassEntity]:
        """Extract class declarations from JavaScript/TypeScript source."""
        classes = []
        
        def visit_node(node):
            if node.type == "class_declaration":
                name_node = self._find_child_by_type(node, "identifier")
                if name_node:
                    name = self._get_node_text(name_node, content_bytes)
                    
                    # Extract parent class if extends
                    parent = None
                    heritage = self._find_child_by_type(node, "class_heritage")
                    if heritage:
                        extends_clause = self._find_child_by_type(heritage, "extends_clause")
                        if extends_clause:
                            parent_id = self._find_child_by_type(extends_clause, "identifier")
                            if parent_id:
                                parent = self._get_node_text(parent_id, content_bytes)

                    classes.append(ClassEntity(
                        name=name,
                        qualified_name=f"{self._current_module}.{name}",
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        parent_classes=[parent] if parent else [],
                    ))

            for child in node.children:
                visit_node(child)

        visit_node(tree.root_node)
        return classes

    def _extract_functions(self, tree, content_bytes: bytes) -> List[FunctionEntity]:
        """Extract functions and methods from JavaScript source."""
        functions = []
        
        def visit_node(node, parent_class: Optional[str] = None):
            # Track class context
            current_class = parent_class
            if node.type == "class_declaration":
                name_node = self._find_child_by_type(node, "identifier")
                if name_node:
                    current_class = self._get_node_text(name_node, content_bytes)

            # Function declaration
            if node.type == "function_declaration":
                name_node = self._find_child_by_type(node, "identifier")
                if name_node:
                    name = self._get_node_text(name_node, content_bytes)
                    is_async = any(
                        c.type == "async" for c in node.children
                    )
                    functions.append(FunctionEntity(
                        name=name,
                        qualified_name=f"{self._current_module}.{name}",
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        is_async=is_async,
                    ))

            # Method definition in class
            elif node.type == "method_definition" and current_class:
                name_node = self._find_child_by_type(node, "property_identifier")
                if name_node:
                    name = self._get_node_text(name_node, content_bytes)
                    is_async = any(c.type == "async" for c in node.children)
                    is_static = any(c.type == "static" for c in node.children)
                    
                    functions.append(FunctionEntity(
                        name=name,
                        qualified_name=f"{self._current_module}.{current_class}.{name}",
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        is_method=True,
                        is_async=is_async,
                        is_static=is_static,
                        parent_class=current_class,
                    ))

            # Variable declaration with function expression or arrow function
            elif node.type == "variable_declarator":
                name_node = self._find_child_by_type(node, "identifier")
                value_node = None
                for child in node.children:
                    if child.type in ("function_expression", "arrow_function"):
                        value_node = child
                        break
                
                if name_node and value_node:
                    name = self._get_node_text(name_node, content_bytes)
                    is_async = any(c.type == "async" for c in value_node.children)
                    functions.append(FunctionEntity(
                        name=name,
                        qualified_name=f"{self._current_module}.{name}",
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        is_async=is_async,
                    ))

            for child in node.children:
                visit_node(child, current_class)

        visit_node(tree.root_node)
        return functions

    def _extract_imports(self, tree, content_bytes: bytes) -> List[ImportEntity]:
        """Extract import and require statements."""
        imports = []
        
        def visit_node(node):
            # ES6 import statement
            if node.type == "import_statement":
                source_node = self._find_child_by_type(node, "string")
                if source_node:
                    # Remove quotes from string
                    module_path = self._get_node_text(source_node, content_bytes)
                    module_path = module_path.strip("'\"")
                    
                    imports.append(ImportEntity(
                        name=module_path.split("/")[-1],
                        qualified_name=f"{self._current_module}:import:{module_path}",
                        language=self.LANGUAGE,
                        module_path=module_path,
                        is_external=self._is_external_import(module_path),
                        location=self._node_to_location(node),
                    ))

            # CommonJS require
            elif node.type == "call_expression":
                func_node = self._find_child_by_type(node, "identifier")
                if func_node:
                    func_name = self._get_node_text(func_node, content_bytes)
                    if func_name == "require":
                        args = self._find_child_by_type(node, "arguments")
                        if args:
                            string_node = self._find_child_by_type(args, "string")
                            if string_node:
                                module_path = self._get_node_text(string_node, content_bytes)
                                module_path = module_path.strip("'\"")
                                
                                imports.append(ImportEntity(
                                    name=module_path.split("/")[-1],
                                    qualified_name=f"{self._current_module}:require:{module_path}",
                                    language=self.LANGUAGE,
                                    module_path=module_path,
                                    is_external=self._is_external_import(module_path),
                                    location=self._node_to_location(node),
                                ))

            for child in node.children:
                visit_node(child)

        visit_node(tree.root_node)
        return imports

    def _is_external_import(self, module_path: str) -> bool:
        """Determine if import refers to external package."""
        # Relative imports are internal
        if module_path.startswith(("./", "../", "/")):
            return False
        
        # Node.js built-in modules
        builtins = {
            "fs", "path", "http", "https", "url", "util", "events",
            "stream", "buffer", "crypto", "os", "child_process",
            "cluster", "net", "dns", "tls", "readline", "repl",
            "vm", "zlib", "assert", "console", "process", "timers",
        }
        if module_path in builtins or module_path.startswith("node:"):
            return False
        
        return True

    def _get_call_target(self, node, content_bytes: bytes) -> Optional[str]:
        """Extract function name from call expression."""
        if node.type == "call_expression":
            func = node.children[0] if node.children else None
            if func:
                if func.type == "identifier":
                    return self._get_node_text(func, content_bytes)
                elif func.type == "member_expression":
                    # Get the property name for method calls
                    prop = self._find_child_by_type(func, "property_identifier")
                    if prop:
                        return self._get_node_text(prop, content_bytes)
        return None


@AnalyzerRegistry.register
class TypeScriptAnalyzer(JavaScriptAnalyzer):
    """
    Tree-sitter based analyzer for TypeScript source code.
    
    Extends JavaScript analyzer with TypeScript-specific grammar
    and handles type annotations, interfaces, and type aliases.
    """

    LANGUAGE = "typescript"
    SUPPORTED_EXTENSIONS = [".ts", ".tsx", ".mts", ".cts"]

    def __init__(self):
        super().__init__()
        # TypeScript uses its own grammar
        try:
            import tree_sitter_typescript as ts_typescript
            self._ts_module = ts_typescript
            self._ts_available = True
        except ImportError:
            self._ts_available = False
            logger.warning("tree-sitter-typescript not available")

    def _initialize_parser(self) -> bool:
        """Initialize tree-sitter parser with TypeScript grammar."""
        if not self._ts_available:
            return False

        try:
            # TypeScript grammar provides both typescript and tsx
            self._language = Language(self._ts_module.language_typescript())
            self._parser = Parser(self._language)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize TypeScript parser: {e}")
            return False

    def _extract_classes(self, tree, content_bytes: bytes) -> List[ClassEntity]:
        """Extract classes and interfaces from TypeScript source."""
        classes = super()._extract_classes(tree, content_bytes)
        
        # Also extract interfaces as class-like entities
        def visit_node(node):
            if node.type == "interface_declaration":
                name_node = self._find_child_by_type(node, "type_identifier")
                if name_node:
                    name = self._get_node_text(name_node, content_bytes)
                    
                    classes.append(ClassEntity(
                        name=name,
                        qualified_name=f"{self._current_module}.{name}",
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        attributes={"kind": "interface"},
                    ))

            elif node.type == "type_alias_declaration":
                name_node = self._find_child_by_type(node, "type_identifier")
                if name_node:
                    name = self._get_node_text(name_node, content_bytes)
                    
                    classes.append(ClassEntity(
                        name=name,
                        qualified_name=f"{self._current_module}.{name}",
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        attributes={"kind": "type_alias"},
                    ))

            for child in node.children:
                visit_node(child)

        visit_node(tree.root_node)
        return classes
