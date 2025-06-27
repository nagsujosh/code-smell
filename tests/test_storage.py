"""
Unit tests for storage module components.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.core.config import StorageConfig
from src.graph.semantic_graph import SemanticGraph, GraphNode, GraphEdge
from src.storage.backend import JSONStorageBackend, StorageMetadata


class TestJSONStorageBackend(unittest.TestCase):
    """Tests for JSON storage backend."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = StorageConfig(
            storage_dir=self.tmpdir,
            enable_compression=True,
        )
        self.backend = JSONStorageBackend(self.config)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _create_sample_graph(self, name="test"):
        """Helper to create sample graph."""
        graph = SemanticGraph(name=name)
        graph.add_node(GraphNode(
            id="n1", node_type="file", name="main.py",
            qualified_name="main.py", language="python"
        ))
        graph.add_node(GraphNode(
            id="n2", node_type="function", name="main",
            qualified_name="main.main", language="python",
            embedding=np.random.rand(768),
        ))
        graph.add_edge(GraphEdge("n1", "n2", "contains"))
        return graph

    def test_save_graph(self):
        """Test saving a graph."""
        graph = self._create_sample_graph()
        
        metadata = self.backend.save(graph, "test_graph")
        
        self.assertIsInstance(metadata, StorageMetadata)
        self.assertEqual(metadata.graph_name, "test_graph")
        self.assertEqual(metadata.node_count, 2)
        self.assertEqual(metadata.edge_count, 1)
        self.assertTrue(metadata.has_embeddings)

    def test_load_graph(self):
        """Test loading a saved graph."""
        graph = self._create_sample_graph()
        self.backend.save(graph, "test_graph")
        
        loaded = self.backend.load("test_graph")
        
        self.assertEqual(loaded.node_count, 2)
        self.assertEqual(loaded.edge_count, 1)
        self.assertIsNotNone(loaded.get_node("n2").embedding)

    def test_delete_graph(self):
        """Test deleting a graph."""
        graph = self._create_sample_graph()
        self.backend.save(graph, "to_delete")
        
        self.assertTrue(self.backend.exists("to_delete"))
        
        result = self.backend.delete("to_delete")
        
        self.assertTrue(result)
        self.assertFalse(self.backend.exists("to_delete"))

    def test_exists(self):
        """Test existence check."""
        graph = self._create_sample_graph()
        
        self.assertFalse(self.backend.exists("nonexistent"))
        
        self.backend.save(graph, "existing")
        
        self.assertTrue(self.backend.exists("existing"))

    def test_list_graphs(self):
        """Test listing saved graphs."""
        graph1 = self._create_sample_graph("graph1")
        graph2 = self._create_sample_graph("graph2")
        
        self.backend.save(graph1, "graph1")
        self.backend.save(graph2, "graph2")
        
        graphs = self.backend.list_graphs()
        
        self.assertEqual(len(graphs), 2)
        names = {g.graph_name for g in graphs}
        self.assertIn("graph1", names)
        self.assertIn("graph2", names)

    def test_compression(self):
        """Test that compression is applied."""
        graph = self._create_sample_graph()
        self.backend.save(graph, "compressed_graph")
        
        graph_dir = Path(self.tmpdir) / "compressed_graph"
        
        self.assertTrue((graph_dir / "graph.json.gz").exists())

    def test_storage_stats(self):
        """Test storage statistics."""
        graph = self._create_sample_graph()
        self.backend.save(graph, "stats_test")
        
        stats = self.backend.get_storage_stats()
        
        self.assertEqual(stats["graph_count"], 1)
        self.assertGreater(stats["total_size_bytes"], 0)
        self.assertEqual(stats["storage_dir"], self.tmpdir)

    def test_metadata_persistence(self):
        """Test metadata persistence across instances."""
        graph = self._create_sample_graph()
        self.backend.save(graph, "persist_test", metadata={"source": "test"})
        
        new_backend = JSONStorageBackend(self.config)
        
        metadata = new_backend.get_metadata("persist_test")
        
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.graph_name, "persist_test")


class TestStorageMetadata(unittest.TestCase):
    """Tests for storage metadata."""

    def test_metadata_creation(self):
        """Test metadata creation."""
        from datetime import datetime
        
        metadata = StorageMetadata(
            graph_name="test",
            repository_source="/path/to/repo",
            created_at=datetime.now(),
            node_count=10,
            edge_count=15,
            has_embeddings=True,
            embedding_dimension=768,
            storage_format="json.gz",
            file_path="/storage/test",
        )
        
        self.assertEqual(metadata.graph_name, "test")
        self.assertEqual(metadata.node_count, 10)
        self.assertTrue(metadata.has_embeddings)

    def test_metadata_serialization(self):
        """Test metadata serialization."""
        from datetime import datetime
        
        metadata = StorageMetadata(
            graph_name="test",
            repository_source="/path/to/repo",
            created_at=datetime.now(),
            node_count=10,
            edge_count=15,
            has_embeddings=True,
            embedding_dimension=768,
            storage_format="json.gz",
            file_path="/storage/test",
        )
        
        data = metadata.to_dict()
        
        self.assertIn("graph_name", data)
        self.assertIn("created_at", data)
        self.assertIn("node_count", data)
        
        loaded = StorageMetadata.from_dict(data)
        
        self.assertEqual(loaded.graph_name, metadata.graph_name)
        self.assertEqual(loaded.node_count, metadata.node_count)


if __name__ == "__main__":
    unittest.main()

