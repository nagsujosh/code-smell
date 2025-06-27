"""
Unit tests for similarity module components.
"""

import unittest

import numpy as np

from src.graph.semantic_graph import SemanticGraph, GraphNode, GraphEdge
from src.similarity.structural import StructuralAnalyzer, StructuralSimilarity
from src.similarity.semantic import SemanticAnalyzer, SemanticSimilarity
from src.similarity.hybrid import HybridAnalyzer, SimilarityResult
from src.core.config import SimilarityConfig


class TestStructuralAnalyzer(unittest.TestCase):
    """Tests for structural similarity analysis."""

    def setUp(self):
        self.analyzer = StructuralAnalyzer()

    def _create_sample_graph(self, name, num_files=3, num_functions=5):
        """Helper to create sample graphs."""
        graph = SemanticGraph(name=name)
        
        graph.add_node(GraphNode(
            id=f"{name}_repo", node_type="repository", name=name,
            qualified_name=name, language="mixed"
        ))
        
        for i in range(num_files):
            graph.add_node(GraphNode(
                id=f"{name}_file{i}", node_type="file", name=f"file{i}.py",
                qualified_name=f"file{i}.py", language="python"
            ))
            graph.add_edge(GraphEdge(f"{name}_repo", f"{name}_file{i}", "contains"))
        
        for i in range(num_functions):
            file_idx = i % num_files
            graph.add_node(GraphNode(
                id=f"{name}_func{i}", node_type="function", name=f"func{i}",
                qualified_name=f"file{file_idx}.func{i}", language="python"
            ))
            graph.add_edge(GraphEdge(f"{name}_file{file_idx}", f"{name}_func{i}", "contains"))
        
        return graph

    def test_identical_graphs(self):
        """Test similarity of identical graphs."""
        graph1 = self._create_sample_graph("repo1")
        graph2 = self._create_sample_graph("repo2")
        
        result = self.analyzer.compute_similarity(graph1, graph2)
        
        self.assertIsInstance(result, StructuralSimilarity)
        self.assertAlmostEqual(result.score, 1.0, places=2)
        self.assertAlmostEqual(result.node_type_similarity, 1.0, places=2)
        self.assertAlmostEqual(result.edge_type_similarity, 1.0, places=2)

    def test_different_sizes(self):
        """Test similarity of different sized graphs."""
        graph1 = self._create_sample_graph("small", num_files=2, num_functions=3)
        graph2 = self._create_sample_graph("large", num_files=10, num_functions=20)
        
        result = self.analyzer.compute_similarity(graph1, graph2)
        
        self.assertLess(result.size_ratio, 0.5)

    def test_empty_graphs(self):
        """Test similarity of empty graphs."""
        graph1 = SemanticGraph(name="empty1")
        graph2 = SemanticGraph(name="empty2")
        
        result = self.analyzer.compute_similarity(graph1, graph2)
        
        self.assertIsInstance(result, StructuralSimilarity)

    def test_dependency_overlap(self):
        """Test dependency overlap calculation."""
        graph1 = SemanticGraph(name="repo1")
        graph2 = SemanticGraph(name="repo2")
        
        graph1.add_node(GraphNode(
            id="dep1", node_type="external_dependency", name="numpy",
            qualified_name="numpy", language="python",
            attributes={"module_path": "numpy"}
        ))
        graph1.add_node(GraphNode(
            id="dep2", node_type="external_dependency", name="pandas",
            qualified_name="pandas", language="python",
            attributes={"module_path": "pandas"}
        ))
        
        graph2.add_node(GraphNode(
            id="dep3", node_type="external_dependency", name="numpy",
            qualified_name="numpy", language="python",
            attributes={"module_path": "numpy"}
        ))
        graph2.add_node(GraphNode(
            id="dep4", node_type="external_dependency", name="scipy",
            qualified_name="scipy", language="python",
            attributes={"module_path": "scipy"}
        ))
        
        result = self.analyzer.compute_similarity(graph1, graph2)
        
        self.assertGreater(result.dependency_overlap, 0)
        self.assertLess(result.dependency_overlap, 1)

    def test_result_to_dict(self):
        """Test result serialization."""
        graph1 = self._create_sample_graph("repo1")
        graph2 = self._create_sample_graph("repo2")
        
        result = self.analyzer.compute_similarity(graph1, graph2)
        data = result.to_dict()
        
        self.assertIn("score", data)
        self.assertIn("node_type_similarity", data)
        self.assertIn("edge_type_similarity", data)
        self.assertIn("details", data)


class TestSemanticAnalyzer(unittest.TestCase):
    """Tests for semantic similarity analysis."""

    def setUp(self):
        self.analyzer = SemanticAnalyzer(similarity_threshold=0.5)

    def _create_graph_with_embeddings(self, name, embeddings):
        """Helper to create graph with embeddings."""
        graph = SemanticGraph(name=name)
        
        for i, emb in enumerate(embeddings):
            graph.add_node(GraphNode(
                id=f"{name}_func{i}", node_type="function", name=f"func{i}",
                qualified_name=f"mod.func{i}", language="python",
                embedding=emb,
            ))
        
        return graph

    def test_identical_embeddings(self):
        """Test similarity of identical embeddings."""
        emb = np.random.rand(768)
        graph1 = self._create_graph_with_embeddings("repo1", [emb, emb])
        graph2 = self._create_graph_with_embeddings("repo2", [emb, emb])
        
        result = self.analyzer.compute_similarity(graph1, graph2)
        
        self.assertIsInstance(result, SemanticSimilarity)
        self.assertAlmostEqual(result.function_similarity, 1.0, places=2)

    def test_orthogonal_embeddings(self):
        """Test similarity of orthogonal embeddings."""
        emb1 = np.zeros(768)
        emb1[0] = 1.0
        emb2 = np.zeros(768)
        emb2[1] = 1.0
        
        graph1 = self._create_graph_with_embeddings("repo1", [emb1])
        graph2 = self._create_graph_with_embeddings("repo2", [emb2])
        
        result = self.analyzer.compute_similarity(graph1, graph2)
        
        self.assertEqual(len(result.matched_pairs), 0)

    def test_matched_pairs(self):
        """Test matched pair reporting."""
        emb = np.random.rand(768)
        graph1 = self._create_graph_with_embeddings("repo1", [emb])
        graph2 = self._create_graph_with_embeddings("repo2", [emb])
        
        result = self.analyzer.compute_similarity(graph1, graph2)
        
        self.assertGreater(len(result.matched_pairs), 0)
        self.assertIn("node1_name", result.matched_pairs[0])
        self.assertIn("node2_name", result.matched_pairs[0])
        self.assertIn("similarity", result.matched_pairs[0])

    def test_empty_graphs(self):
        """Test similarity with empty graphs."""
        graph1 = SemanticGraph(name="empty1")
        graph2 = SemanticGraph(name="empty2")
        
        result = self.analyzer.compute_similarity(graph1, graph2)
        
        self.assertEqual(result.score, 0.0)

    def test_compute_node_similarity(self):
        """Test direct node similarity computation."""
        emb = np.random.rand(768)
        node1 = GraphNode(
            id="n1", node_type="function", name="func1",
            qualified_name="mod.func1", language="python",
            embedding=emb,
        )
        node2 = GraphNode(
            id="n2", node_type="function", name="func2",
            qualified_name="mod.func2", language="python",
            embedding=emb,
        )
        
        similarity = self.analyzer.compute_node_similarity(node1, node2)
        
        self.assertAlmostEqual(similarity, 1.0, places=5)

    def test_find_similar_nodes(self):
        """Test finding similar nodes in a graph."""
        base_emb = np.random.rand(768)
        similar_emb = base_emb + np.random.rand(768) * 0.1
        different_emb = np.random.rand(768)
        
        graph = SemanticGraph(name="test")
        graph.add_node(GraphNode(
            id="similar", node_type="function", name="similar",
            qualified_name="mod.similar", language="python",
            embedding=similar_emb,
        ))
        graph.add_node(GraphNode(
            id="different", node_type="function", name="different",
            qualified_name="mod.different", language="python",
            embedding=different_emb,
        ))
        
        query_node = GraphNode(
            id="query", node_type="function", name="query",
            qualified_name="mod.query", language="python",
            embedding=base_emb,
        )
        
        results = self.analyzer.find_similar_nodes(query_node, graph, top_k=2)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0].id, "similar")


class TestHybridAnalyzer(unittest.TestCase):
    """Tests for hybrid similarity analysis."""

    def setUp(self):
        config = SimilarityConfig(
            structural_weight=0.4,
            semantic_weight=0.6,
            similarity_threshold=0.5,
        )
        self.analyzer = HybridAnalyzer(config)

    def _create_test_graph(self, name):
        """Helper to create test graph."""
        graph = SemanticGraph(name=name)
        
        emb = np.random.rand(768)
        
        graph.add_node(GraphNode(
            id=f"{name}_file", node_type="file", name="main.py",
            qualified_name="main.py", language="python"
        ))
        graph.add_node(GraphNode(
            id=f"{name}_func", node_type="function", name="main",
            qualified_name="main.main", language="python",
            embedding=emb,
        ))
        graph.add_edge(GraphEdge(f"{name}_file", f"{name}_func", "contains"))
        
        return graph

    def test_hybrid_similarity(self):
        """Test hybrid similarity computation."""
        graph1 = self._create_test_graph("repo1")
        graph2 = self._create_test_graph("repo2")
        
        result = self.analyzer.compute_similarity(graph1, graph2)
        
        self.assertIsInstance(result, SimilarityResult)
        self.assertGreaterEqual(result.overall_score, 0)
        self.assertLessEqual(result.overall_score, 1)

    def test_weight_application(self):
        """Test that weights are applied correctly."""
        graph1 = self._create_test_graph("repo1")
        graph2 = self._create_test_graph("repo2")
        
        result = self.analyzer.compute_similarity(graph1, graph2)
        
        expected = (
            0.4 * result.structural.score +
            0.6 * result.semantic.score
        )
        self.assertAlmostEqual(result.overall_score, expected, places=5)

    def test_result_explainability(self):
        """Test result explainability features."""
        graph1 = self._create_test_graph("repo1")
        graph2 = self._create_test_graph("repo2")
        
        result = self.analyzer.compute_similarity(graph1, graph2)
        
        self.assertIsNotNone(result.explanation)
        self.assertGreater(len(result.explanation), 0)
        self.assertIsInstance(result.limitations, list)

    def test_result_serialization(self):
        """Test result serialization."""
        graph1 = self._create_test_graph("repo1")
        graph2 = self._create_test_graph("repo2")
        
        result = self.analyzer.compute_similarity(graph1, graph2)
        data = result.to_dict()
        
        self.assertIn("overall_score", data)
        self.assertIn("structural_score", data)
        self.assertIn("semantic_score", data)
        self.assertIn("breakdown", data)

    def test_interpret_score(self):
        """Test score interpretation."""
        graph1 = self._create_test_graph("repo1")
        graph2 = self._create_test_graph("repo2")
        
        result = self.analyzer.compute_similarity(graph1, graph2)
        interpretation = result._interpret_score()
        
        self.assertIsInstance(interpretation, str)
        self.assertGreater(len(interpretation), 0)

    def test_set_weights(self):
        """Test weight adjustment."""
        self.analyzer.set_weights(structural_weight=0.7, semantic_weight=0.3)
        
        self.assertAlmostEqual(
            self.analyzer.config.structural_weight + 
            self.analyzer.config.semantic_weight,
            1.0,
            places=5
        )


if __name__ == "__main__":
    unittest.main()

