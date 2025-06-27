"""
Ruby Analyzer Module.

Provides tree-sitter based static analysis for Ruby source files.
Extracts classes, modules, methods, and require statements with
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
    import tree_sitter_ruby as ts_ruby
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning("tree-sitter-ruby not available, using regex fallback")


@AnalyzerRegistry.register
class RubyAnalyzer(BaseTreeSitterAnalyzer):
    """
    Tree-sitter based analyzer for Ruby source code.
    
    Extracts:
        - Classes and modules
        - Instance and class methods
        - require and require_relative statements
        - attr_accessor and other DSL patterns
    """

    LANGUAGE = "ruby"
    SUPPORTED_EXTENSIONS = [".rb", ".rake", ".gemspec"]

    CLASS_NODE_TYPES = ["class", "module"]
    FUNCTION_NODE_TYPES = ["method", "singleton_method"]
    IMPORT_NODE_TYPES = ["call"]  # require/require_relative are method calls

    def __init__(self):
        super().__init__()
        self._ts_available = TREE_SITTER_AVAILABLE

    def _initialize_parser(self) -> bool:
        """Initialize tree-sitter parser with Ruby grammar."""
        if not self._ts_available:
            return False

        try:
            self._language = Language(ts_ruby.language())
            self._parser = Parser(self._language)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Ruby parser: {e}")
            return False

    def _extract_classes(self, tree, content_bytes: bytes) -> List[ClassEntity]:
        """Extract class and module definitions."""
        classes = []
        
        def visit_node(node, parent_module: str = ""):
            current_parent = parent_module
            
            if node.type == "class":
                name_node = self._find_child_by_type(node, "constant")
                if name_node:
                    name = self._get_node_text(name_node, content_bytes)
                    
                    # Extract superclass
                    parent_classes = []
                    superclass = self._find_child_by_type(node, "superclass")
                    if superclass:
                        super_const = self._find_child_by_type(superclass, "constant")
                        if super_const:
                            parent_classes.append(
                                self._get_node_text(super_const, content_bytes)
                            )

                    qualified = f"{parent_module}::{name}" if parent_module else f"{self._current_module}::{name}"

                    classes.append(ClassEntity(
                        name=name,
                        qualified_name=qualified,
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        parent_classes=parent_classes,
                        attributes={"kind": "class"},
                    ))
                    
                    current_parent = qualified

            elif node.type == "module":
                name_node = self._find_child_by_type(node, "constant")
                if name_node:
                    name = self._get_node_text(name_node, content_bytes)
                    
                    qualified = f"{parent_module}::{name}" if parent_module else f"{self._current_module}::{name}"

                    classes.append(ClassEntity(
                        name=name,
                        qualified_name=qualified,
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        attributes={"kind": "module"},
                    ))
                    
                    current_parent = qualified

            for child in node.children:
                visit_node(child, current_parent)

        visit_node(tree.root_node)
        return classes

    def _extract_functions(self, tree, content_bytes: bytes) -> List[FunctionEntity]:
        """Extract method definitions."""
        functions = []
        
        def visit_node(node, parent_class: Optional[str] = None):
            current_class = parent_class
            
            # Track class/module context
            if node.type in ("class", "module"):
                name_node = self._find_child_by_type(node, "constant")
                if name_node:
                    name = self._get_node_text(name_node, content_bytes)
                    current_class = f"{parent_class}::{name}" if parent_class else name

            # Instance method
            if node.type == "method":
                name_node = self._find_child_by_type(node, "identifier")
                if name_node:
                    name = self._get_node_text(name_node, content_bytes)
                    
                    # Extract parameters
                    params = self._extract_ruby_parameters(node, content_bytes)

                    qualified = f"{self._current_module}::{current_class}::{name}" if current_class else f"{self._current_module}::{name}"

                    functions.append(FunctionEntity(
                        name=name,
                        qualified_name=qualified,
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        is_method=current_class is not None,
                        parent_class=current_class,
                        parameters=params,
                    ))

            # Singleton (class) method
            elif node.type == "singleton_method":
                name_node = self._find_child_by_type(node, "identifier")
                if name_node:
                    name = self._get_node_text(name_node, content_bytes)
                    params = self._extract_ruby_parameters(node, content_bytes)

                    qualified = f"{self._current_module}::{current_class}::self.{name}" if current_class else f"{self._current_module}::self.{name}"

                    functions.append(FunctionEntity(
                        name=name,
                        qualified_name=qualified,
                        language=self.LANGUAGE,
                        location=self._node_to_location(node),
                        is_method=True,
                        is_static=True,
                        parent_class=current_class,
                        parameters=params,
                    ))

            for child in node.children:
                visit_node(child, current_class)

        visit_node(tree.root_node)
        return functions

    def _extract_imports(self, tree, content_bytes: bytes) -> List[ImportEntity]:
        """Extract require and require_relative statements."""
        imports = []
        
        def visit_node(node):
            if node.type == "call":
                # Get method name
                method_node = self._find_child_by_type(node, "identifier")
                if method_node:
                    method_name = self._get_node_text(method_node, content_bytes)
                    
                    if method_name in ("require", "require_relative", "load"):
                        # Get argument
                        args = self._find_child_by_type(node, "argument_list")
                        if args:
                            for child in args.children:
                                if child.type == "string":
                                    string_content = self._find_child_by_type(
                                        child, "string_content"
                                    )
                                    if string_content:
                                        path = self._get_node_text(
                                            string_content, content_bytes
                                        )
                                    else:
                                        # Direct string
                                        path = self._get_node_text(child, content_bytes)
                                        path = path.strip("'\"")
                                    
                                    is_relative = method_name == "require_relative"
                                    
                                    imports.append(ImportEntity(
                                        name=path.split("/")[-1],
                                        qualified_name=f"{self._current_module}:{method_name}:{path}",
                                        language=self.LANGUAGE,
                                        module_path=path,
                                        is_external=self._is_external_import(path, is_relative),
                                        location=self._node_to_location(node),
                                        attributes={"method": method_name},
                                    ))
                                    break

            for child in node.children:
                visit_node(child)

        visit_node(tree.root_node)
        return imports

    def _extract_ruby_parameters(self, node, content_bytes: bytes) -> List[str]:
        """Extract parameters from method definition."""
        params = []
        method_params = self._find_child_by_type(node, "method_parameters")
        if method_params:
            for child in method_params.children:
                if child.type == "identifier":
                    params.append(self._get_node_text(child, content_bytes))
                elif child.type in ("optional_parameter", "keyword_parameter",
                                   "splat_parameter", "block_parameter"):
                    ident = self._find_child_by_type(child, "identifier")
                    if ident:
                        params.append(self._get_node_text(ident, content_bytes))
        return params

    def _is_external_import(self, path: str, is_relative: bool = False) -> bool:
        """Determine if require refers to external gem."""
        if is_relative:
            return False
        
        # Ruby standard library modules
        stdlib_modules = {
            "abbrev", "base64", "benchmark", "bigdecimal", "cgi", "csv",
            "date", "debug", "delegate", "digest", "drb", "English",
            "erb", "fileutils", "find", "forwardable", "getoptlong",
            "io/console", "ipaddr", "irb", "json", "logger", "matrix",
            "minitest", "monitor", "mutex_m", "net/ftp", "net/http",
            "net/imap", "net/pop", "net/smtp", "nkf", "observer",
            "open-uri", "open3", "openssl", "optparse", "ostruct",
            "pathname", "pp", "prettyprint", "prime", "pstore", "psych",
            "racc", "rdoc", "readline", "reline", "resolv", "resolv-replace",
            "ripper", "rss", "rubygems", "securerandom", "set", "shellwords",
            "singleton", "socket", "stringio", "strscan", "syslog",
            "tempfile", "thread", "time", "timeout", "tmpdir", "tracer",
            "tsort", "un", "uri", "weakref", "webrick", "yaml", "zlib",
        }
        
        return path not in stdlib_modules

    def _get_call_target(self, node, content_bytes: bytes) -> Optional[str]:
        """Extract method name from call expression."""
        if node.type == "call":
            # Direct method call
            method_node = self._find_child_by_type(node, "identifier")
            if method_node:
                return self._get_node_text(method_node, content_bytes)
        return None
