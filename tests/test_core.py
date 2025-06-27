"""
Unit tests for core module components.
"""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.core.config import (
    Config,
    PipelineConfig,
    IngestionConfig,
    AnalysisConfig,
    GraphConfig,
    EmbeddingConfig,
    SimilarityConfig,
    StorageConfig,
)
from src.core.pipeline import Pipeline, PipelineStage, PipelineState, StageStatus
from src.core.exceptions import (
    PipelineError,
    IngestionError,
    AnalysisError,
    GraphConstructionError,
    EmbeddingError,
    SimilarityError,
    LanguageNotSupportedError,
    RepositoryValidationError,
)


class TestConfig(unittest.TestCase):
    """Tests for configuration management."""

    def test_default_config(self):
        """Test that default configuration is created correctly."""
        config = PipelineConfig()
        
        self.assertIsInstance(config.ingestion, IngestionConfig)
        self.assertIsInstance(config.analysis, AnalysisConfig)
        self.assertIsInstance(config.graph, GraphConfig)
        self.assertIsInstance(config.embedding, EmbeddingConfig)
        self.assertIsInstance(config.similarity, SimilarityConfig)
        self.assertIsInstance(config.storage, StorageConfig)

    def test_ingestion_config_defaults(self):
        """Test ingestion configuration defaults."""
        config = IngestionConfig()
        
        self.assertIn("*.pyc", config.ignore_patterns)
        self.assertIn("__pycache__", config.ignore_patterns)
        self.assertIn(".git", config.ignore_patterns)
        self.assertEqual(config.max_file_size, 1024 * 1024)
        self.assertEqual(config.clone_depth, 1)
        self.assertEqual(config.git_timeout, 300)

    def test_similarity_config_weights(self):
        """Test similarity configuration weight defaults."""
        config = SimilarityConfig()
        
        self.assertEqual(config.structural_weight, 0.4)
        self.assertEqual(config.semantic_weight, 0.6)
        self.assertAlmostEqual(
            config.structural_weight + config.semantic_weight, 1.0
        )

    def test_config_save_and_load(self):
        """Test configuration serialization and deserialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            
            Config.save_to_file(str(config_path))
            
            self.assertTrue(config_path.exists())
            
            with open(config_path) as f:
                data = json.load(f)
            
            self.assertIn("ingestion", data)
            self.assertIn("embedding", data)
            self.assertIn("similarity", data)

    def test_graph_config_node_types(self):
        """Test graph configuration node types."""
        config = GraphConfig()
        
        expected_types = {
            "repository", "file", "module", "class",
            "interface", "function", "method", "external_dependency"
        }
        self.assertEqual(config.node_types, expected_types)

    def test_embedding_config_defaults(self):
        """Test embedding configuration defaults."""
        config = EmbeddingConfig()
        
        self.assertEqual(config.model_name, "microsoft/codebert-base")
        self.assertEqual(config.max_token_length, 512)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.device, "auto")


class TestPipelineState(unittest.TestCase):
    """Tests for pipeline state management."""

    def test_state_creation(self):
        """Test pipeline state creation."""
        state = PipelineState(
            pipeline_id="test-123",
            source_repo="/path/to/repo",
        )
        
        self.assertEqual(state.pipeline_id, "test-123")
        self.assertEqual(state.source_repo, "/path/to/repo")
        self.assertIsNone(state.target_repo)
        self.assertIsInstance(state.created_at, datetime)

    def test_stage_status_tracking(self):
        """Test stage status tracking."""
        state = PipelineState(
            pipeline_id="test-123",
            source_repo="/path/to/repo",
        )
        
        self.assertEqual(
            state.get_stage_status("ingestion"),
            StageStatus.PENDING
        )
        
        state.record_stage_start("ingestion")
        self.assertEqual(
            state.get_stage_status("ingestion"),
            StageStatus.RUNNING
        )
        
        state.record_stage_completion("ingestion", {"files": 10})
        self.assertEqual(
            state.get_stage_status("ingestion"),
            StageStatus.COMPLETED
        )
        self.assertTrue(state.is_stage_completed("ingestion"))

    def test_stage_failure_tracking(self):
        """Test stage failure tracking."""
        state = PipelineState(
            pipeline_id="test-123",
            source_repo="/path/to/repo",
        )
        
        state.record_stage_start("analysis")
        state.record_stage_failure("analysis", "Parse error")
        
        self.assertEqual(
            state.get_stage_status("analysis"),
            StageStatus.FAILED
        )
        self.assertEqual(
            state.stage_results["analysis"].error,
            "Parse error"
        )

    def test_state_serialization(self):
        """Test state serialization to dictionary."""
        state = PipelineState(
            pipeline_id="test-123",
            source_repo="/path/to/source",
            target_repo="/path/to/target",
        )
        
        state.record_stage_start("ingestion")
        state.record_stage_completion("ingestion", {})
        
        data = state.to_dict()
        
        self.assertEqual(data["pipeline_id"], "test-123")
        self.assertEqual(data["source_repo"], "/path/to/source")
        self.assertEqual(data["target_repo"], "/path/to/target")
        self.assertIn("ingestion", data["stage_results"])

    def test_state_save_and_load(self):
        """Test state persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            
            state = PipelineState(
                pipeline_id="test-456",
                source_repo="/path/to/repo",
            )
            state.record_stage_start("ingestion")
            state.record_stage_completion("ingestion", {"count": 5})
            
            state.save(state_path)
            
            self.assertTrue(state_path.exists())
            
            loaded_state = PipelineState.load(state_path)
            
            self.assertEqual(loaded_state.pipeline_id, "test-456")
            self.assertTrue(loaded_state.is_stage_completed("ingestion"))


class TestExceptions(unittest.TestCase):
    """Tests for custom exceptions."""

    def test_pipeline_error(self):
        """Test base pipeline error."""
        error = PipelineError("Test error", stage="test", details={"key": "value"})
        
        self.assertEqual(str(error), "[test] Test error")
        self.assertEqual(error.stage, "test")
        self.assertEqual(error.details, {"key": "value"})

    def test_ingestion_error(self):
        """Test ingestion error."""
        error = IngestionError("Clone failed")
        
        self.assertEqual(error.stage, "Ingestion")
        self.assertIn("Clone failed", str(error))

    def test_language_not_supported_error(self):
        """Test language not supported error."""
        error = LanguageNotSupportedError("brainfuck")
        
        self.assertIn("brainfuck", str(error))
        self.assertEqual(error.details["language"], "brainfuck")

    def test_repository_validation_error(self):
        """Test repository validation error."""
        error = RepositoryValidationError("/invalid/path", "Does not exist")
        
        self.assertIn("Does not exist", str(error))
        self.assertEqual(error.details["path"], "/invalid/path")


if __name__ == "__main__":
    unittest.main()

