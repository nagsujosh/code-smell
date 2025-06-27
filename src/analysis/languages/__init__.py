"""
Language-specific analyzers.

This package contains pluggable analyzers for different programming
languages. Analyzers use tree-sitter for accurate AST-based parsing
where available, with regex-based fallback for edge cases.

Supported Languages:
    - Python: AST-based (native Python ast module)
    - JavaScript/TypeScript: Tree-sitter based
    - Java: Tree-sitter based
    - Go: Tree-sitter based
    - Rust: Tree-sitter based
    - C/C++: Tree-sitter based
    - Ruby: Tree-sitter based
    - PHP: Tree-sitter based
    - Kotlin: Regex-based (tree-sitter optional)
"""

# Base classes
from src.analysis.languages.base_treesitter_analyzer import BaseTreeSitterAnalyzer

# Language analyzers - importing registers them with the registry
from src.analysis.languages import python_analyzer
from src.analysis.languages import javascript_analyzer
from src.analysis.languages import java_analyzer
from src.analysis.languages import go_analyzer
from src.analysis.languages import rust_analyzer
from src.analysis.languages import cpp_analyzer
from src.analysis.languages import ruby_analyzer
from src.analysis.languages import php_analyzer

__all__ = [
    "BaseTreeSitterAnalyzer",
    "python_analyzer",
    "javascript_analyzer",
    "java_analyzer",
    "go_analyzer",
    "rust_analyzer",
    "cpp_analyzer",
    "ruby_analyzer",
    "php_analyzer",
]
