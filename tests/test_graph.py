"""
Unit tests for graph module components.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.graph.semantic_graph import SemanticGraph, GraphNode, GraphEdge


class TestGraphNode(unittest.TestCase):
    """Tests for graph node data structure."""

    def test_node_creation(self):
        """Test node creation."""
        node = GraphNode(
            id="node123",
            node_type="function",
            name="calculate",
            qualified_name="math.calculate",
            language="python",
        )
        
        self.assertEqual(node.id, "node123")
        self.assertEqual(node.node_type, "function")
        self.assertEqual(node.name, "calculate")
        self.assertEqual(node.language, "python")

    def test_node_with_attributes(self):
        """Test node with custom attributes."""
        node = GraphNode(
            id="node456",
            node_type="class",
            name="Calculator",
            qualified_name="math.Calculator",
            language="python",
            attributes={"is_abstract": True, "method_count": 5},
        )
        
        self.assertEqual(node.attributes["is_abstract"], True)
        self.assertEqual(node.attributes["method_count"], 5)

    def test_node_with_embedding(self):
        """Test node with embedding vector."""
        embedding = np.random.rand(768).astype(np.float32)
        node = GraphNode(
            id="node789",
            node_type="function",
            name="test",
            qualified_name="test.test",
            language="python",
            embedding=embedding,
        )
        
        self.assertIsNotNone(node.embedding)
        self.assertEqual(len(node.embedding), 768)

    def test_node_to_dict(self):
        """Test node serialization."""
        node = GraphNode(
            id="node123",
            node_type="function",
            name="test",
            qualified_name="mod.test",
            language="python",
            attributes={"complexity": 3},
        )
        
        data = node.to_dict()
        
        self.assertEqual(data["id"], "node123")
        self.assertEqual(data["node_type"], "function")
        self.assertEqual(data["attributes"]["complexity"], 3)

    def test_node_to_dict_with_embedding(self):
        """Test node serialization with embedding."""
        embedding = np.array([1.0, 2.0, 3.0])
        node = GraphNode(
            id="node123",
            node_type="function",
            name="test",
            qualified_name="mod.test",
            language="python",
            embedding=embedding,
        )
        
        data = node.to_dict(include_embedding=True)
        
        self.assertIn("embedding", data)
        self.assertEqual(data["embedding"], [1.0, 2.0, 3.0])


class TestGraphEdge(unittest.TestCase):
    """Tests for graph edge data structure."""

    def test_edge_creation(self):
        """Test edge creation."""
        edge = GraphEdge(
            source_id="source123",
            target_id="target456",
            edge_type="calls",
        )
        
        self.assertEqual(edge.source_id, "source123")
        self.assertEqual(edge.target_id, "target456")
        self.assertEqual(edge.edge_type, "calls")

    def test_edge_with_attributes(self):
        """Test edge with custom attributes."""
        edge = GraphEdge(
            source_id="source123",
            target_id="target456",
            edge_type="imports",
            attributes={"line": 10, "alias": "np"},
        )
        
        self.assertEqual(edge.attributes["line"], 10)
        self.assertEqual(edge.attributes["alias"], "np")

    def test_edge_to_dict(self):
        """Test edge serialization."""
        edge = GraphEdge(
            source_id="s1",
            target_id="t1",
            edge_type="contains",
            attributes={"order": 1},
        )
        
        data = edge.to_dict()
        
        self.assertEqual(data["source_id"], "s1")
        self.assertEqual(data["target_id"], "t1")
        self.assertEqual(data["edge_type"], "contains")


class TestSemanticGraph(unittest.TestCase):
    """Tests for semantic graph."""

    def setUp(self):
        self.graph = SemanticGraph(name="test_graph")

    def test_empty_graph(self):
        """Test empty graph creation."""
        self.assertEqual(self.graph.node_count, 0)
        self.assertEqual(self.graph.edge_count, 0)

    def test_add_node(self):
        """Test adding nodes."""
        node = GraphNode(
            id="n1",
            node_type="file",
            name="main.py",
            qualified_name="main.py",
            language="python",
        )
        self.graph.add_node(node)
        
        self.assertEqual(self.graph.node_count, 1)
        self.assertIsNotNone(self.graph.get_node("n1"))

    def test_add_edge(self):
        """Test adding edges."""
        node1 = GraphNode(
            id="n1", node_type="file", name="a.py",
            qualified_name="a.py", language="python"
        )
        node2 = GraphNode(
            id="n2", node_type="function", name="func",
            qualified_name="a.func", language="python"
        )
        
        self.graph.add_node(node1)
        self.graph.add_node(node2)
        
        edge = GraphEdge(
            source_id="n1",
            target_id="n2",
            edge_type="contains",
        )
        self.graph.add_edge(edge)
        
        self.assertEqual(self.graph.edge_count, 1)

    def test_get_nodes_by_type(self):
        """Test getting nodes by type."""
        self.graph.add_node(GraphNode(
            id="f1", node_type="file", name="a.py",
            qualified_name="a.py", language="python"
        ))
        self.graph.add_node(GraphNode(
            id="f2", node_type="file", name="b.py",
            qualified_name="b.py", language="python"
        ))
        self.graph.add_node(GraphNode(
            id="fn1", node_type="function", name="func",
            qualified_name="a.func", language="python"
        ))
        
        files = self.graph.get_nodes_by_type("file")
        functions = self.graph.get_nodes_by_type("function")
        
        self.assertEqual(len(files), 2)
        self.assertEqual(len(functions), 1)

    def test_get_neighbors(self):
        """Test getting neighboring nodes."""
        self.graph.add_node(GraphNode(
            id="n1", node_type="file", name="a.py",
            qualified_name="a.py", language="python"
        ))
        self.graph.add_node(GraphNode(
            id="n2", node_type="function", name="func1",
            qualified_name="a.func1", language="python"
        ))
        self.graph.add_node(GraphNode(
            id="n3", node_type="function", name="func2",
            qualified_name="a.func2", language="python"
        ))
        
        self.graph.add_edge(GraphEdge("n1", "n2", "contains"))
        self.graph.add_edge(GraphEdge("n1", "n3", "contains"))
        
        neighbors = self.graph.get_neighbors("n1", direction="out")
        
        self.assertEqual(len(neighbors), 2)

    def test_get_node_types(self):
        """Test getting all node types."""
        self.graph.add_node(GraphNode(
            id="f1", node_type="file", name="a.py",
            qualified_name="a.py", language="python"
        ))
        self.graph.add_node(GraphNode(
            id="c1", node_type="class", name="MyClass",
            qualified_name="a.MyClass", language="python"
        ))
        
        types = self.graph.get_node_types()
        
        self.assertIn("file", types)
        self.assertIn("class", types)

    def test_get_type_distribution(self):
        """Test getting type distribution."""
        self.graph.add_node(GraphNode(
            id="f1", node_type="file", name="a.py",
            qualified_name="a.py", language="python"
        ))
        self.graph.add_node(GraphNode(
            id="f2", node_type="file", name="b.py",
            qualified_name="b.py", language="python"
        ))
        self.graph.add_node(GraphNode(
            id="fn1", node_type="function", name="func",
            qualified_name="a.func", language="python"
        ))
        
        dist = self.graph.get_type_distribution()
        
        self.assertEqual(dist["nodes"]["file"], 2)
        self.assertEqual(dist["nodes"]["function"], 1)

    def test_get_statistics(self):
        """Test getting graph statistics."""
        self.graph.add_node(GraphNode(
            id="n1", node_type="file", name="a.py",
            qualified_name="a.py", language="python"
        ))
        self.graph.add_node(GraphNode(
            id="n2", node_type="function", name="func",
            qualified_name="a.func", language="python"
        ))
        self.graph.add_edge(GraphEdge("n1", "n2", "contains"))
        
        stats = self.graph.get_statistics()
        
        self.assertEqual(stats["node_count"], 2)
        self.assertEqual(stats["edge_count"], 1)
        self.assertIn("density", stats)
        self.assertIn("avg_degree", stats)

    def test_save_and_load(self):
        """Test graph persistence."""
        self.graph.add_node(GraphNode(
            id="n1", node_type="file", name="a.py",
            qualified_name="a.py", language="python"
        ))
        self.graph.add_node(GraphNode(
            id="n2", node_type="function", name="func",
            qualified_name="a.func", language="python",
            embedding=np.array([1.0, 2.0, 3.0]),
        ))
        self.graph.add_edge(GraphEdge("n1", "n2", "contains"))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph.json"
            
            self.graph.save(path, include_embeddings=True)
            
            self.assertTrue(path.exists())
            
            loaded = SemanticGraph.load(path)
            
            self.assertEqual(loaded.node_count, 2)
            self.assertEqual(loaded.edge_count, 1)
            
            node2 = loaded.get_node("n2")
            self.assertIsNotNone(node2.embedding)

    def test_subgraph(self):
        """Test subgraph extraction."""
        for i in range(5):
            self.graph.add_node(GraphNode(
                id=f"n{i}", node_type="function", name=f"func{i}",
                qualified_name=f"mod.func{i}", language="python"
            ))
        
        self.graph.add_edge(GraphEdge("n0", "n1", "calls"))
        self.graph.add_edge(GraphEdge("n1", "n2", "calls"))
        self.graph.add_edge(GraphEdge("n3", "n4", "calls"))
        
        subgraph = self.graph.get_subgraph({"n0", "n1", "n2"})
        
        self.assertEqual(subgraph.node_count, 3)
        self.assertEqual(subgraph.edge_count, 2)

    def test_shortest_path(self):
        """Test shortest path finding."""
        self.graph.add_node(GraphNode(
            id="n1", node_type="function", name="func1",
            qualified_name="a.func1", language="python"
        ))
        self.graph.add_node(GraphNode(
            id="n2", node_type="function", name="func2",
            qualified_name="a.func2", language="python"
        ))
        self.graph.add_node(GraphNode(
            id="n3", node_type="function", name="func3",
            qualified_name="a.func3", language="python"
        ))
        
        self.graph.add_edge(GraphEdge("n1", "n2", "calls"))
        self.graph.add_edge(GraphEdge("n2", "n3", "calls"))
        
        path = self.graph.find_shortest_path("n1", "n3")
        
        self.assertEqual(path, ["n1", "n2", "n3"])

    def test_nodes_with_embeddings(self):
        """Test getting nodes with embeddings."""
        self.graph.add_node(GraphNode(
            id="n1", node_type="function", name="func1",
            qualified_name="a.func1", language="python",
            embedding=np.array([1.0, 2.0]),
        ))
        self.graph.add_node(GraphNode(
            id="n2", node_type="function", name="func2",
            qualified_name="a.func2", language="python",
        ))
        
        with_embeddings = self.graph.get_nodes_with_embeddings()
        
        self.assertEqual(len(with_embeddings), 1)
        self.assertEqual(with_embeddings[0].id, "n1")


if __name__ == "__main__":
    unittest.main()

