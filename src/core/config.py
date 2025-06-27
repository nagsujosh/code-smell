"""
Configuration management for the Semantic Codebase Graph Engine.

Provides centralized configuration for all pipeline stages with
sensible defaults and validation.
"""

import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class IngestionConfig:
    """Configuration for repository ingestion."""

    # Patterns to ignore during file discovery
    ignore_patterns: List[str] = field(default_factory=lambda: [
        "*.pyc", "__pycache__", ".git", ".svn", ".hg",
        "node_modules", "vendor", ".venv", "venv", "env",
        "*.min.js", "*.min.css", "*.map", "*.lock",
        ".idea", ".vscode", ".DS_Store", "*.egg-info",
        "build", "dist", "target", "out", "bin", "obj",
        "*.log", "*.tmp", "*.cache", "coverage", ".coverage",
    ])

    # Maximum file size to process (in bytes)
    max_file_size: int = 1024 * 1024  # 1MB

    # Clone depth for remote repositories (0 = full clone)
    clone_depth: int = 1

    # Timeout for git operations (seconds)
    git_timeout: int = 300


@dataclass
class AnalysisConfig:
    """Configuration for static analysis."""

    # Supported language extensions mapping
    language_extensions: Dict[str, str] = field(default_factory=lambda: {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".c": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
    })

    # Minimum confidence threshold for language detection
    language_confidence_threshold: float = 0.7

    # Maximum lines to analyze per file (0 = no limit)
    max_lines_per_file: int = 10000


@dataclass
class GraphConfig:
    """Configuration for semantic graph construction."""

    # Node types to include in the graph
    node_types: Set[str] = field(default_factory=lambda: {
        "repository", "file", "module", "class", "interface",
        "function", "method", "external_dependency",
    })

    # Edge types to include in the graph
    edge_types: Set[str] = field(default_factory=lambda: {
        "contains", "defines", "calls", "imports", "depends_on",
        "inherits", "implements",
    })

    # Maximum depth for call graph analysis
    max_call_depth: int = 5


@dataclass
class EmbeddingConfig:
    """Configuration for semantic embedding generation."""

    # Model identifier for code embeddings
    model_name: str = "microsoft/codebert-base"

    # Cache directory for models
    model_cache_dir: str = "./model_cache"

    # Maximum token length for embedding
    max_token_length: int = 512

    # Batch size for embedding generation
    batch_size: int = 32

    # Device for inference (cpu, cuda, mps, or auto)
    device: str = "auto"

    # Number of significant code lines to extract
    significant_lines_count: int = 50


@dataclass
class SimilarityConfig:
    """Configuration for similarity analysis."""

    # Weight for structural similarity in hybrid score
    structural_weight: float = 0.4

    # Weight for semantic similarity in hybrid score
    semantic_weight: float = 0.6

    # Threshold for considering nodes as similar
    similarity_threshold: float = 0.7

    # Methods to use for similarity computation
    similarity_methods: List[str] = field(default_factory=lambda: [
        "cosine", "jaccard", "graph_edit_distance",
    ])


@dataclass
class StorageConfig:
    """Configuration for graph storage."""

    # Base directory for storage
    storage_dir: str = "./data/graphs"

    # Format for graph serialization (json, pickle)
    serialization_format: str = "json"

    # Enable compression for stored graphs
    enable_compression: bool = True


@dataclass
class PipelineConfig:
    """Master configuration combining all stage configurations."""

    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    # Enable verbose logging
    verbose: bool = False

    # Working directory for cloned repositories
    work_dir: str = "./data/repos"
    
    # Directory for reports
    reports_dir: str = "./data/reports"


class Config:
    """
    Central configuration manager providing access to all settings.
    
    Supports loading from environment variables and configuration files.
    """

    _instance: Optional["Config"] = None
    _config: PipelineConfig = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = PipelineConfig()
        return cls._instance

    @classmethod
    def get(cls) -> PipelineConfig:
        """Get the current pipeline configuration."""
        if cls._instance is None:
            cls()
        return cls._instance._config

    @classmethod
    def load_from_file(cls, config_path: str) -> PipelineConfig:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file.
            
        Returns:
            Loaded PipelineConfig instance.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            data = json.load(f)

        instance = cls()
        instance._config = cls._dict_to_config(data)
        return instance._config

    @classmethod
    def load_from_env(cls) -> PipelineConfig:
        """
        Load configuration from environment variables.
        
        Environment variables are prefixed with SCGE_ (Semantic Codebase Graph Engine).
        
        Returns:
            PipelineConfig with environment overrides applied.
        """
        instance = cls()
        config = instance._config

        # Override embedding settings from environment
        if os.getenv("SCGE_MODEL_NAME"):
            config.embedding.model_name = os.getenv("SCGE_MODEL_NAME")

        if os.getenv("SCGE_MODEL_CACHE_DIR"):
            config.embedding.model_cache_dir = os.getenv("SCGE_MODEL_CACHE_DIR")

        if os.getenv("SCGE_DEVICE"):
            config.embedding.device = os.getenv("SCGE_DEVICE")

        # Override storage settings from environment
        if os.getenv("SCGE_STORAGE_DIR"):
            config.storage.storage_dir = os.getenv("SCGE_STORAGE_DIR")

        # Override verbosity
        if os.getenv("SCGE_VERBOSE"):
            config.verbose = os.getenv("SCGE_VERBOSE").lower() in ("true", "1", "yes")

        return config

    @staticmethod
    def _dict_to_config(data: dict) -> PipelineConfig:
        """Convert a dictionary to PipelineConfig."""
        config = PipelineConfig()

        if "ingestion" in data:
            config.ingestion = IngestionConfig(**data["ingestion"])

        if "analysis" in data:
            config.analysis = AnalysisConfig(**data["analysis"])

        if "graph" in data:
            graph_data = data["graph"]
            if "node_types" in graph_data:
                graph_data["node_types"] = set(graph_data["node_types"])
            if "edge_types" in graph_data:
                graph_data["edge_types"] = set(graph_data["edge_types"])
            config.graph = GraphConfig(**graph_data)

        if "embedding" in data:
            config.embedding = EmbeddingConfig(**data["embedding"])

        if "similarity" in data:
            config.similarity = SimilarityConfig(**data["similarity"])

        if "storage" in data:
            config.storage = StorageConfig(**data["storage"])

        if "verbose" in data:
            config.verbose = data["verbose"]

        if "work_dir" in data:
            config.work_dir = data["work_dir"]

        return config

    @classmethod
    def save_to_file(cls, config_path: str) -> None:
        """
        Save current configuration to a JSON file.
        
        Args:
            config_path: Path to save the configuration file.
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config = cls.get()
        data = cls._config_to_dict(config)

        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _config_to_dict(config: PipelineConfig) -> dict:
        """Convert PipelineConfig to a dictionary."""
        return {
            "ingestion": {
                "ignore_patterns": config.ingestion.ignore_patterns,
                "max_file_size": config.ingestion.max_file_size,
                "clone_depth": config.ingestion.clone_depth,
                "git_timeout": config.ingestion.git_timeout,
            },
            "analysis": {
                "language_extensions": config.analysis.language_extensions,
                "language_confidence_threshold": config.analysis.language_confidence_threshold,
                "max_lines_per_file": config.analysis.max_lines_per_file,
            },
            "graph": {
                "node_types": list(config.graph.node_types),
                "edge_types": list(config.graph.edge_types),
                "max_call_depth": config.graph.max_call_depth,
            },
            "embedding": {
                "model_name": config.embedding.model_name,
                "model_cache_dir": config.embedding.model_cache_dir,
                "max_token_length": config.embedding.max_token_length,
                "batch_size": config.embedding.batch_size,
                "device": config.embedding.device,
                "significant_lines_count": config.embedding.significant_lines_count,
            },
            "similarity": {
                "structural_weight": config.similarity.structural_weight,
                "semantic_weight": config.similarity.semantic_weight,
                "similarity_threshold": config.similarity.similarity_threshold,
                "similarity_methods": config.similarity.similarity_methods,
            },
            "storage": {
                "storage_dir": config.storage.storage_dir,
                "serialization_format": config.storage.serialization_format,
                "enable_compression": config.storage.enable_compression,
            },
            "verbose": config.verbose,
            "work_dir": config.work_dir,
        }

