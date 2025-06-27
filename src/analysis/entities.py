"""
Language-neutral code entity definitions.

Defines the intermediate representation (IR) for code entities
extracted during static analysis. These entities form the basis
for semantic graph construction.
"""

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EntityType(Enum):
    """Types of code entities that can be extracted."""
    FILE = "file"
    MODULE = "module"
    PACKAGE = "package"
    CLASS = "class"
    INTERFACE = "interface"
    STRUCT = "struct"
    FUNCTION = "function"
    METHOD = "method"
    IMPORT = "import"
    VARIABLE = "variable"
    CONSTANT = "constant"
    EXTERNAL_DEPENDENCY = "external_dependency"


class RelationshipType(Enum):
    """Types of relationships between code entities."""
    CONTAINS = "contains"
    DEFINES = "defines"
    CALLS = "calls"
    IMPORTS = "imports"
    DEPENDS_ON = "depends_on"
    INHERITS = "inherits"
    IMPLEMENTS = "implements"
    USES = "uses"
    RETURNS = "returns"
    PARAMETER_OF = "parameter_of"


@dataclass
class Location:
    """Source code location information."""
    file_path: str
    line_start: int
    line_end: int
    column_start: int = 0
    column_end: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "column_start": self.column_start,
            "column_end": self.column_end,
        }


@dataclass
class CodeEntity:
    """
    Base class for all code entities.
    
    Represents a single identifiable element in the codebase
    with associated metadata and source information.
    """

    entity_type: EntityType
    name: str
    qualified_name: str
    language: str
    location: Optional[Location] = None
    docstring: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    _id: Optional[str] = field(default=None, repr=False)

    @property
    def id(self) -> str:
        """Generate a unique, stable identifier for this entity."""
        if self._id is None:
            key = f"{self.language}:{self.entity_type.value}:{self.qualified_name}"
            self._id = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self._id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "entity_type": self.entity_type.value,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "language": self.language,
            "location": self.location.to_dict() if self.location else None,
            "docstring": self.docstring,
            "attributes": self.attributes,
        }


@dataclass
class FileEntity(CodeEntity):
    """Represents a source file."""

    def __init__(
        self,
        name: str,
        file_path: str,
        language: str,
        **kwargs
    ):
        super().__init__(
            entity_type=EntityType.FILE,
            name=name,
            qualified_name=file_path,
            language=language,
            **kwargs
        )
        self.file_path = file_path


@dataclass
class ModuleEntity(CodeEntity):
    """Represents a module or package."""

    def __init__(
        self,
        name: str,
        qualified_name: str,
        language: str,
        is_package: bool = False,
        **kwargs
    ):
        entity_type = EntityType.PACKAGE if is_package else EntityType.MODULE
        super().__init__(
            entity_type=entity_type,
            name=name,
            qualified_name=qualified_name,
            language=language,
            **kwargs
        )
        self.is_package = is_package


@dataclass
class ClassEntity(CodeEntity):
    """Represents a class, interface, or struct."""

    def __init__(
        self,
        name: str,
        qualified_name: str,
        language: str,
        base_classes: List[str] = None,
        interfaces: List[str] = None,
        is_abstract: bool = False,
        is_interface: bool = False,
        decorators: List[str] = None,
        **kwargs
    ):
        entity_type = EntityType.INTERFACE if is_interface else EntityType.CLASS
        super().__init__(
            entity_type=entity_type,
            name=name,
            qualified_name=qualified_name,
            language=language,
            **kwargs
        )
        self.base_classes = base_classes or []
        self.interfaces = interfaces or []
        self.is_abstract = is_abstract
        self.decorators = decorators or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update({
            "base_classes": self.base_classes,
            "interfaces": self.interfaces,
            "is_abstract": self.is_abstract,
            "decorators": self.decorators,
        })
        return data


@dataclass
class FunctionEntity(CodeEntity):
    """Represents a function or method."""

    def __init__(
        self,
        name: str,
        qualified_name: str,
        language: str,
        parameters: List[Dict[str, Any]] = None,
        return_type: Optional[str] = None,
        is_method: bool = False,
        is_async: bool = False,
        is_static: bool = False,
        is_classmethod: bool = False,
        decorators: List[str] = None,
        complexity: int = 1,
        **kwargs
    ):
        entity_type = EntityType.METHOD if is_method else EntityType.FUNCTION
        super().__init__(
            entity_type=entity_type,
            name=name,
            qualified_name=qualified_name,
            language=language,
            **kwargs
        )
        self.parameters = parameters or []
        self.return_type = return_type
        self.is_async = is_async
        self.is_static = is_static
        self.is_classmethod = is_classmethod
        self.decorators = decorators or []
        self.complexity = complexity

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update({
            "parameters": self.parameters,
            "return_type": self.return_type,
            "is_async": self.is_async,
            "is_static": self.is_static,
            "is_classmethod": self.is_classmethod,
            "decorators": self.decorators,
            "complexity": self.complexity,
        })
        return data


@dataclass
class ImportEntity(CodeEntity):
    """Represents an import or dependency."""

    def __init__(
        self,
        name: str,
        qualified_name: str,
        language: str,
        module_path: str,
        imported_names: List[str] = None,
        is_relative: bool = False,
        is_external: bool = False,
        alias: Optional[str] = None,
        **kwargs
    ):
        entity_type = (
            EntityType.EXTERNAL_DEPENDENCY if is_external else EntityType.IMPORT
        )
        super().__init__(
            entity_type=entity_type,
            name=name,
            qualified_name=qualified_name,
            language=language,
            **kwargs
        )
        self.module_path = module_path
        self.imported_names = imported_names or []
        self.is_relative = is_relative
        self.is_external = is_external
        self.alias = alias

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update({
            "module_path": self.module_path,
            "imported_names": self.imported_names,
            "is_relative": self.is_relative,
            "is_external": self.is_external,
            "alias": self.alias,
        })
        return data


@dataclass
class Relationship:
    """Represents a relationship between two code entities."""

    source_id: str
    target_id: str
    relationship_type: RelationshipType
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Generate unique identifier for this relationship."""
        key = f"{self.source_id}:{self.relationship_type.value}:{self.target_id}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type.value,
            "attributes": self.attributes,
        }

