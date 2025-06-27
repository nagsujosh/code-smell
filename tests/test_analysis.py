"""
Unit tests for analysis module components.
"""

import unittest
from pathlib import Path

from src.analysis.detector import LanguageDetector, LanguageDetectionResult
from src.analysis.entities import (
    CodeEntity,
    FileEntity,
    ClassEntity,
    FunctionEntity,
    ImportEntity,
    Relationship,
    EntityType,
    RelationshipType,
    Location,
)
from src.analysis.registry import AnalyzerRegistry, BaseLanguageAnalyzer
from src.analysis.languages.python_analyzer import PythonAnalyzer


class TestLanguageDetector(unittest.TestCase):
    """Tests for language detection."""

    def setUp(self):
        self.detector = LanguageDetector()

    def test_detect_python(self):
        """Test Python language detection."""
        result = self.detector.detect_file_language(Path("test.py"))
        
        self.assertEqual(result.language, "python")
        self.assertEqual(result.confidence, 1.0)
        self.assertEqual(result.method, "extension")

    def test_detect_javascript(self):
        """Test JavaScript language detection."""
        for ext in [".js", ".jsx", ".mjs"]:
            result = self.detector.detect_file_language(Path(f"test{ext}"))
            self.assertEqual(result.language, "javascript")

    def test_detect_typescript(self):
        """Test TypeScript language detection."""
        for ext in [".ts", ".tsx"]:
            result = self.detector.detect_file_language(Path(f"test{ext}"))
            self.assertEqual(result.language, "typescript")

    def test_detect_unknown(self):
        """Test unknown language detection."""
        result = self.detector.detect_file_language(Path("test.xyz"))
        
        self.assertEqual(result.language, "unknown")
        self.assertEqual(result.confidence, 0.0)

    def test_detect_from_shebang(self):
        """Test language detection from shebang."""
        content = "#!/usr/bin/env python3\nprint('hello')"
        result = self.detector.detect_file_language(Path("script"), content)
        
        self.assertEqual(result.language, "python")
        self.assertEqual(result.method, "shebang")

    def test_supported_languages(self):
        """Test listing supported languages."""
        languages = self.detector.get_supported_languages()
        
        self.assertIn("python", languages)
        self.assertIn("javascript", languages)
        self.assertIn("java", languages)
        self.assertIn("go", languages)
        self.assertIn("rust", languages)

    def test_extensions_for_language(self):
        """Test getting extensions for a language."""
        extensions = self.detector.get_extensions_for_language("python")
        
        self.assertIn(".py", extensions)
        self.assertIn(".pyw", extensions)
        self.assertIn(".pyi", extensions)


class TestCodeEntities(unittest.TestCase):
    """Tests for code entity data structures."""

    def test_file_entity(self):
        """Test file entity creation."""
        entity = FileEntity(
            name="main.py",
            file_path="src/main.py",
            language="python",
        )
        
        self.assertEqual(entity.entity_type, EntityType.FILE)
        self.assertEqual(entity.name, "main.py")
        self.assertEqual(entity.language, "python")
        self.assertIsNotNone(entity.id)

    def test_class_entity(self):
        """Test class entity creation."""
        entity = ClassEntity(
            name="Calculator",
            qualified_name="math.Calculator",
            language="python",
            base_classes=["BaseCalculator"],
            is_abstract=False,
            decorators=["dataclass"],
        )
        
        self.assertEqual(entity.entity_type, EntityType.CLASS)
        self.assertEqual(entity.name, "Calculator")
        self.assertEqual(entity.base_classes, ["BaseCalculator"])
        self.assertEqual(entity.decorators, ["dataclass"])

    def test_function_entity(self):
        """Test function entity creation."""
        entity = FunctionEntity(
            name="calculate",
            qualified_name="math.calculate",
            language="python",
            parameters=[
                {"name": "x", "type": "int"},
                {"name": "y", "type": "int"},
            ],
            return_type="int",
            is_async=True,
            complexity=5,
        )
        
        self.assertEqual(entity.entity_type, EntityType.FUNCTION)
        self.assertEqual(len(entity.parameters), 2)
        self.assertEqual(entity.return_type, "int")
        self.assertTrue(entity.is_async)
        self.assertEqual(entity.complexity, 5)

    def test_import_entity(self):
        """Test import entity creation."""
        entity = ImportEntity(
            name="numpy",
            qualified_name="main:import:numpy",
            language="python",
            module_path="numpy",
            imported_names=["array", "ndarray"],
            is_external=True,
        )
        
        self.assertEqual(entity.entity_type, EntityType.EXTERNAL_DEPENDENCY)
        self.assertEqual(entity.module_path, "numpy")
        self.assertTrue(entity.is_external)

    def test_entity_id_stability(self):
        """Test that entity IDs are stable."""
        entity1 = FunctionEntity(
            name="test",
            qualified_name="module.test",
            language="python",
        )
        entity2 = FunctionEntity(
            name="test",
            qualified_name="module.test",
            language="python",
        )
        
        self.assertEqual(entity1.id, entity2.id)

    def test_relationship(self):
        """Test relationship creation."""
        rel = Relationship(
            source_id="source123",
            target_id="target456",
            relationship_type=RelationshipType.CALLS,
            attributes={"line": 42},
        )
        
        self.assertEqual(rel.source_id, "source123")
        self.assertEqual(rel.target_id, "target456")
        self.assertEqual(rel.relationship_type, RelationshipType.CALLS)
        self.assertEqual(rel.attributes["line"], 42)

    def test_location(self):
        """Test source location."""
        loc = Location(
            file_path="src/main.py",
            line_start=10,
            line_end=20,
            column_start=4,
        )
        
        data = loc.to_dict()
        
        self.assertEqual(data["file_path"], "src/main.py")
        self.assertEqual(data["line_start"], 10)
        self.assertEqual(data["line_end"], 20)


class TestAnalyzerRegistry(unittest.TestCase):
    """Tests for analyzer registry."""

    def setUp(self):
        AnalyzerRegistry.clear()

    def tearDown(self):
        AnalyzerRegistry.clear()
        from src.analysis.languages import python_analyzer

    def test_register_analyzer(self):
        """Test analyzer registration."""
        @AnalyzerRegistry.register
        class TestAnalyzer(BaseLanguageAnalyzer):
            LANGUAGE = "test_lang"
        
        self.assertTrue(AnalyzerRegistry.has_analyzer("test_lang"))

    def test_get_analyzer(self):
        """Test getting registered analyzer."""
        @AnalyzerRegistry.register
        class TestAnalyzer(BaseLanguageAnalyzer):
            LANGUAGE = "test_lang2"
        
        analyzer = AnalyzerRegistry.get_analyzer("test_lang2")
        
        self.assertIsNotNone(analyzer)
        self.assertIsInstance(analyzer, TestAnalyzer)

    def test_list_languages(self):
        """Test listing registered languages."""
        @AnalyzerRegistry.register
        class TestAnalyzer1(BaseLanguageAnalyzer):
            LANGUAGE = "lang1"
        
        @AnalyzerRegistry.register
        class TestAnalyzer2(BaseLanguageAnalyzer):
            LANGUAGE = "lang2"
        
        languages = AnalyzerRegistry.list_languages()
        
        self.assertIn("lang1", languages)
        self.assertIn("lang2", languages)

    def test_unknown_analyzer(self):
        """Test getting unknown analyzer returns None."""
        analyzer = AnalyzerRegistry.get_analyzer("nonexistent")
        self.assertIsNone(analyzer)


class TestPythonAnalyzer(unittest.TestCase):
    """Tests for Python analyzer."""

    def setUp(self):
        self.analyzer = PythonAnalyzer()

    def test_analyze_function(self):
        """Test analyzing a simple function."""
        code = '''
def greet(name: str) -> str:
    """Greet a person."""
    return f"Hello, {name}!"
'''
        result = self.analyzer.analyze_file("test.py", code)
        
        self.assertTrue(len(result.entities) > 0)
        
        functions = [
            e for e in result.entities
            if e.entity_type == EntityType.FUNCTION
        ]
        self.assertEqual(len(functions), 1)
        self.assertEqual(functions[0].name, "greet")

    def test_analyze_class(self):
        """Test analyzing a class."""
        code = '''
class Calculator:
    """A simple calculator."""
    
    def add(self, x, y):
        return x + y
    
    def subtract(self, x, y):
        return x - y
'''
        result = self.analyzer.analyze_file("calc.py", code)
        
        classes = [
            e for e in result.entities
            if e.entity_type == EntityType.CLASS
        ]
        self.assertEqual(len(classes), 1)
        self.assertEqual(classes[0].name, "Calculator")
        
        methods = [
            e for e in result.entities
            if e.entity_type == EntityType.METHOD
        ]
        self.assertEqual(len(methods), 2)

    def test_analyze_imports(self):
        """Test analyzing imports."""
        code = '''
import os
from pathlib import Path
from typing import List, Dict
'''
        result = self.analyzer.analyze_file("imports.py", code)
        
        imports = [
            e for e in result.entities
            if e.entity_type in (EntityType.IMPORT, EntityType.EXTERNAL_DEPENDENCY)
        ]
        self.assertTrue(len(imports) >= 3)

    def test_analyze_inheritance(self):
        """Test analyzing class inheritance."""
        code = '''
class Animal:
    pass

class Dog(Animal):
    def bark(self):
        pass
'''
        result = self.analyzer.analyze_file("animals.py", code)
        
        dog_class = None
        for e in result.entities:
            if isinstance(e, ClassEntity) and e.name == "Dog":
                dog_class = e
                break
        
        self.assertIsNotNone(dog_class)
        self.assertIn("Animal", dog_class.base_classes)

    def test_analyze_decorators(self):
        """Test analyzing decorators."""
        code = '''
class MyClass:
    @staticmethod
    def static_method():
        pass
    
    @classmethod
    def class_method(cls):
        pass
'''
        result = self.analyzer.analyze_file("decorators.py", code)
        
        static = None
        classm = None
        for e in result.entities:
            if isinstance(e, FunctionEntity):
                if e.name == "static_method":
                    static = e
                elif e.name == "class_method":
                    classm = e
        
        self.assertIsNotNone(static)
        self.assertTrue(static.is_static)
        self.assertIsNotNone(classm)
        self.assertTrue(classm.is_classmethod)

    def test_analyze_async_function(self):
        """Test analyzing async functions."""
        code = '''
async def fetch_data(url: str):
    pass
'''
        result = self.analyzer.analyze_file("async.py", code)
        
        functions = [
            e for e in result.entities
            if e.entity_type == EntityType.FUNCTION
        ]
        self.assertEqual(len(functions), 1)
        self.assertTrue(functions[0].is_async)

    def test_syntax_error_handling(self):
        """Test handling of syntax errors."""
        code = '''
def broken(
    # missing closing paren
'''
        result = self.analyzer.analyze_file("broken.py", code)
        
        self.assertTrue(len(result.errors) > 0)

    def test_complexity_calculation(self):
        """Test cyclomatic complexity calculation."""
        code = '''
def complex_function(x):
    if x > 0:
        if x > 10:
            return "big"
        else:
            return "small"
    elif x < 0:
        return "negative"
    else:
        for i in range(x):
            if i % 2 == 0:
                continue
        return "zero"
'''
        result = self.analyzer.analyze_file("complex.py", code)
        
        func = None
        for e in result.entities:
            if isinstance(e, FunctionEntity) and e.name == "complex_function":
                func = e
                break
        
        self.assertIsNotNone(func)
        self.assertGreater(func.complexity, 1)


if __name__ == "__main__":
    unittest.main()

