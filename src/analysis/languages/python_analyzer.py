"""
Python language analyzer.

Provides static analysis capabilities for Python source code
using the AST module for parsing and entity extraction.
"""

import ast
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from src.analysis.registry import AnalyzerRegistry, BaseLanguageAnalyzer
from src.analysis.analyzer import AnalysisResult
from src.analysis.entities import (
    CodeEntity,
    FileEntity,
    ModuleEntity,
    ClassEntity,
    FunctionEntity,
    ImportEntity,
    Relationship,
    RelationshipType,
    Location,
    EntityType,
)

logger = logging.getLogger(__name__)


@AnalyzerRegistry.register
class PythonAnalyzer(BaseLanguageAnalyzer):
    """
    Static analyzer for Python source code.
    
    Uses Python's AST module to extract code entities and
    relationships from source files.
    """

    LANGUAGE = "python"
    SUPPORTED_EXTENSIONS = [".py", ".pyw", ".pyi"]

    def __init__(self):
        self._current_file = None
        self._current_module = None

    def analyze_file(self, file_path: str, content: str) -> AnalysisResult:
        """
        Analyze a Python source file.
        
        Args:
            file_path: Path to the file.
            content: Source code content.
            
        Returns:
            AnalysisResult with extracted entities and relationships.
        """
        self._current_file = file_path
        self._current_module = self._path_to_module(file_path)

        result = AnalysisResult()

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            result.errors.append({
                "file": file_path,
                "error": f"Syntax error: {e}",
                "line": e.lineno,
            })
            return result

        file_entity = FileEntity(
            name=file_path.split("/")[-1],
            file_path=file_path,
            language=self.LANGUAGE,
        )
        result.entities.append(file_entity)

        entities = self._extract_entities(tree, file_path)
        result.entities.extend(entities)

        relationships = self._extract_relationships(
            tree, file_path, file_entity, entities
        )
        result.relationships.extend(relationships)

        result.metrics = {
            "total_lines": len(content.split("\n")),
            "classes": len([e for e in entities if e.entity_type == EntityType.CLASS]),
            "functions": len([e for e in entities if e.entity_type in (EntityType.FUNCTION, EntityType.METHOD)]),
            "imports": len([e for e in entities if e.entity_type in (EntityType.IMPORT, EntityType.EXTERNAL_DEPENDENCY)]),
        }

        return result

    def _path_to_module(self, file_path: str) -> str:
        """Convert file path to module name."""
        module = file_path.replace("/", ".").replace("\\", ".")
        if module.endswith(".py"):
            module = module[:-3]
        return module

    def _extract_entities(
        self, tree: ast.AST, file_path: str
    ) -> List[CodeEntity]:
        """Extract all code entities from AST."""
        entities = []
        self._visit_node(tree, entities, file_path, parent_class=None)
        return entities

    def _visit_node(
        self,
        node: ast.AST,
        entities: List[CodeEntity],
        file_path: str,
        parent_class: Optional[str] = None,
    ) -> None:
        """Recursively visit AST nodes and extract entities."""
        if isinstance(node, ast.ClassDef):
            class_entity = self._extract_class(node, file_path, parent_class)
            entities.append(class_entity)

            for child in ast.iter_child_nodes(node):
                self._visit_node(child, entities, file_path, node.name)

        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            func_entity = self._extract_function(node, file_path, parent_class)
            entities.append(func_entity)

        elif isinstance(node, ast.Import):
            for alias in node.names:
                import_entity = self._extract_import(
                    alias.name, alias.asname, file_path, node
                )
                entities.append(import_entity)

        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                import_entity = self._extract_import_from(
                    node.module, alias.name, alias.asname, file_path, node
                )
                entities.append(import_entity)

        else:
            for child in ast.iter_child_nodes(node):
                self._visit_node(child, entities, file_path, parent_class)

    def _extract_class(
        self,
        node: ast.ClassDef,
        file_path: str,
        parent_class: Optional[str] = None,
    ) -> ClassEntity:
        """Extract a class entity from an AST node."""
        qualified_name = f"{self._current_module}.{node.name}"
        if parent_class:
            qualified_name = f"{self._current_module}.{parent_class}.{node.name}"

        base_classes = []
        for base in node.bases:
            base_name = self._get_name(base)
            if base_name:
                base_classes.append(base_name)

        decorators = []
        for decorator in node.decorator_list:
            dec_name = self._get_name(decorator)
            if dec_name:
                decorators.append(dec_name)

        docstring = ast.get_docstring(node)

        is_abstract = "ABC" in base_classes or "abstractmethod" in decorators

        return ClassEntity(
            name=node.name,
            qualified_name=qualified_name,
            language=self.LANGUAGE,
            base_classes=base_classes,
            is_abstract=is_abstract,
            decorators=decorators,
            docstring=docstring,
            location=Location(
                file_path=file_path,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                column_start=node.col_offset,
            ),
        )

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: str,
        parent_class: Optional[str] = None,
    ) -> FunctionEntity:
        """Extract a function or method entity from an AST node."""
        is_method = parent_class is not None
        is_async = isinstance(node, ast.AsyncFunctionDef)

        if parent_class:
            qualified_name = f"{self._current_module}.{parent_class}.{node.name}"
        else:
            qualified_name = f"{self._current_module}.{node.name}"

        parameters = []
        for arg in node.args.args:
            param = {
                "name": arg.arg,
                "type": self._get_annotation(arg.annotation),
            }
            parameters.append(param)

        return_type = self._get_annotation(node.returns)

        decorators = []
        is_static = False
        is_classmethod = False
        for decorator in node.decorator_list:
            dec_name = self._get_name(decorator)
            if dec_name:
                decorators.append(dec_name)
                if dec_name == "staticmethod":
                    is_static = True
                elif dec_name == "classmethod":
                    is_classmethod = True

        docstring = ast.get_docstring(node)

        complexity = self._calculate_complexity(node)

        return FunctionEntity(
            name=node.name,
            qualified_name=qualified_name,
            language=self.LANGUAGE,
            parameters=parameters,
            return_type=return_type,
            is_method=is_method,
            is_async=is_async,
            is_static=is_static,
            is_classmethod=is_classmethod,
            decorators=decorators,
            complexity=complexity,
            docstring=docstring,
            location=Location(
                file_path=file_path,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                column_start=node.col_offset,
            ),
        )

    def _extract_import(
        self,
        module_name: str,
        alias: Optional[str],
        file_path: str,
        node: ast.Import,
    ) -> ImportEntity:
        """Extract an import entity."""
        is_external = self._is_external_import(module_name)

        return ImportEntity(
            name=alias or module_name,
            qualified_name=f"{self._current_module}:import:{module_name}",
            language=self.LANGUAGE,
            module_path=module_name,
            alias=alias,
            is_external=is_external,
            location=Location(
                file_path=file_path,
                line_start=node.lineno,
                line_end=node.lineno,
            ),
        )

    def _extract_import_from(
        self,
        module: Optional[str],
        name: str,
        alias: Optional[str],
        file_path: str,
        node: ast.ImportFrom,
    ) -> ImportEntity:
        """Extract an import-from entity."""
        module_path = module or ""
        is_relative = node.level > 0
        is_external = not is_relative and self._is_external_import(module_path)

        full_name = f"{module_path}.{name}" if module_path else name

        return ImportEntity(
            name=alias or name,
            qualified_name=f"{self._current_module}:import:{full_name}",
            language=self.LANGUAGE,
            module_path=module_path,
            imported_names=[name],
            alias=alias,
            is_relative=is_relative,
            is_external=is_external,
            location=Location(
                file_path=file_path,
                line_start=node.lineno,
                line_end=node.lineno,
            ),
        )

    def _extract_relationships(
        self,
        tree: ast.AST,
        file_path: str,
        file_entity: FileEntity,
        entities: List[CodeEntity],
    ) -> List[Relationship]:
        """Extract relationships between entities."""
        relationships = []

        entity_map = {e.qualified_name: e for e in entities}

        for entity in entities:
            if entity.entity_type in (EntityType.CLASS, EntityType.FUNCTION, EntityType.METHOD):
                relationships.append(Relationship(
                    source_id=file_entity.id,
                    target_id=entity.id,
                    relationship_type=RelationshipType.CONTAINS,
                ))

        for entity in entities:
            if isinstance(entity, ClassEntity):
                for base in entity.base_classes:
                    base_qualified = f"{self._current_module}.{base}"
                    if base_qualified in entity_map:
                        relationships.append(Relationship(
                            source_id=entity.id,
                            target_id=entity_map[base_qualified].id,
                            relationship_type=RelationshipType.INHERITS,
                        ))

        function_names = {
            e.name: e for e in entities
            if e.entity_type in (EntityType.FUNCTION, EntityType.METHOD)
        }

        for entity in entities:
            if isinstance(entity, FunctionEntity) and entity.location:
                calls = self._extract_function_calls(tree, entity)
                for call_name in calls:
                    if call_name in function_names:
                        target = function_names[call_name]
                        relationships.append(Relationship(
                            source_id=entity.id,
                            target_id=target.id,
                            relationship_type=RelationshipType.CALLS,
                        ))

        for entity in entities:
            if isinstance(entity, ImportEntity):
                relationships.append(Relationship(
                    source_id=file_entity.id,
                    target_id=entity.id,
                    relationship_type=RelationshipType.IMPORTS,
                ))

        return relationships

    def _extract_function_calls(
        self, tree: ast.AST, function_entity: FunctionEntity
    ) -> Set[str]:
        """Extract function calls made within a function."""
        calls = set()

        class CallVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
                self.generic_visit(node)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == function_entity.name:
                    CallVisitor().visit(node)
                    break

        return calls

    def _get_name(self, node: ast.AST) -> Optional[str]:
        """Get the name from an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value_name = self._get_name(node.value)
            if value_name:
                return f"{value_name}.{node.attr}"
            return node.attr
        elif isinstance(node, ast.Call):
            return self._get_name(node.func)
        elif isinstance(node, ast.Subscript):
            return self._get_name(node.value)
        return None

    def _get_annotation(self, node: Optional[ast.AST]) -> Optional[str]:
        """Extract type annotation as string."""
        if node is None:
            return None
        return self._get_name(node)

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.IfExp):
                complexity += 1
            elif isinstance(child, ast.comprehension):
                complexity += 1

        return complexity

    def _is_external_import(self, module_path: str) -> bool:
        """Check if an import is from an external package."""
        stdlib_modules = {
            "abc", "aifc", "argparse", "array", "ast", "asyncio",
            "atexit", "base64", "bdb", "binascii", "binhex", "bisect",
            "builtins", "bz2", "calendar", "cgi", "cgitb", "chunk",
            "cmath", "cmd", "code", "codecs", "codeop", "collections",
            "colorsys", "compileall", "concurrent", "configparser",
            "contextlib", "contextvars", "copy", "copyreg", "cProfile",
            "crypt", "csv", "ctypes", "curses", "dataclasses", "datetime",
            "dbm", "decimal", "difflib", "dis", "distutils", "doctest",
            "email", "encodings", "enum", "errno", "faulthandler", "fcntl",
            "filecmp", "fileinput", "fnmatch", "fractions", "ftplib",
            "functools", "gc", "getopt", "getpass", "gettext", "glob",
            "graphlib", "grp", "gzip", "hashlib", "heapq", "hmac", "html",
            "http", "idlelib", "imaplib", "imghdr", "imp", "importlib",
            "inspect", "io", "ipaddress", "itertools", "json", "keyword",
            "lib2to3", "linecache", "locale", "logging", "lzma", "mailbox",
            "mailcap", "marshal", "math", "mimetypes", "mmap", "modulefinder",
            "multiprocessing", "netrc", "nis", "nntplib", "numbers",
            "operator", "optparse", "os", "ossaudiodev", "pathlib", "pdb",
            "pickle", "pickletools", "pipes", "pkgutil", "platform", "plistlib",
            "poplib", "posix", "posixpath", "pprint", "profile", "pstats",
            "pty", "pwd", "py_compile", "pyclbr", "pydoc", "queue", "quopri",
            "random", "re", "readline", "reprlib", "resource", "rlcompleter",
            "runpy", "sched", "secrets", "select", "selectors", "shelve",
            "shlex", "shutil", "signal", "site", "smtpd", "smtplib", "sndhdr",
            "socket", "socketserver", "spwd", "sqlite3", "ssl", "stat",
            "statistics", "string", "stringprep", "struct", "subprocess",
            "sunau", "symtable", "sys", "sysconfig", "syslog", "tabnanny",
            "tarfile", "telnetlib", "tempfile", "termios", "test", "textwrap",
            "threading", "time", "timeit", "tkinter", "token", "tokenize",
            "tomllib", "trace", "traceback", "tracemalloc", "tty", "turtle",
            "turtledemo", "types", "typing", "unicodedata", "unittest", "urllib",
            "uu", "uuid", "venv", "warnings", "wave", "weakref", "webbrowser",
            "winreg", "winsound", "wsgiref", "xdrlib", "xml", "xmlrpc",
            "zipapp", "zipfile", "zipimport", "zlib", "zoneinfo",
        }

        root_module = module_path.split(".")[0]
        return root_module not in stdlib_modules

    def extract_entities(self, file_path: str, content: str) -> List[CodeEntity]:
        """Extract entities (compatibility method)."""
        result = self.analyze_file(file_path, content)
        return result.entities

    def extract_relationships(
        self, file_path: str, content: str, entities: List[CodeEntity]
    ) -> List[Relationship]:
        """Extract relationships (compatibility method)."""
        result = self.analyze_file(file_path, content)
        return result.relationships

