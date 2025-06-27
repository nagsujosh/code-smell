#!/usr/bin/env python3
"""
Test script for AST graph processor only (no ML dependencies).
"""

import sys
from pathlib import Path

# Add src/utils to path directly
sys.path.insert(0, str(Path(__file__).parent / "src" / "utils"))

from ast_graph_processor import ASTGraphProcessor


def test_ast_processor():
    """Test the AST graph processor without ML dependencies."""
    print("Testing AST Graph Processor (No ML Dependencies)")
    print("=" * 50)
    
    # Sample Python code
    sample_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, x, y):
        return x + y
    
    def multiply(self, x, y):
        return x * y

# Test the functions
calc = Calculator()
result = calc.add(5, 3)
print(f"Result: {result}")
"""
    
    try:
        processor = ASTGraphProcessor()
        
        # Parse to graph
        print("1. Parsing code to graph...")
        graph = processor.parse_to_graph(sample_code)
        print(f"   Graph nodes: {graph.number_of_nodes()}")
        print(f"   Graph edges: {graph.number_of_edges()}")
        
        # Get features
        print("\n2. Extracting features...")
        features = processor.graph_to_features(graph)
        print(f"   Function count: {features['function_count']}")
        print(f"   Class count: {features['class_count']}")
        print(f"   Import count: {features['import_count']}")
        print(f"   Total nodes: {features['total_nodes']}")
        print(f"   Total edges: {features['total_edges']}")
        print(f"   Max depth: {features['max_depth']}")
        print(f"   Average degree: {features['avg_degree']:.2f}")
        
        # Complexity metrics
        complexity = features['complexity_metrics']
        print(f"   Control flow count: {complexity['control_flow_count']}")
        print(f"   Data structure count: {complexity['data_structure_count']}")
        print(f"   Function call count: {complexity['function_call_count']}")
        print(f"   Assignment count: {complexity['assignment_count']}")
        print(f"   Cyclomatic complexity: {complexity['cyclomatic_complexity']}")
        
        # Node type distribution
        print(f"\n3. Node type distribution:")
        for node_type, count in features['node_type_distribution'].items():
            print(f"   {node_type}: {count}")
        
        # Get vector
        print(f"\n4. Creating feature vector...")
        vector = processor.process_code(sample_code)
        print(f"   Vector shape: {vector.shape}")
        print(f"   Vector (first 10 values): {vector[:10]}")
        
        print("\n✅ AST processing test completed successfully!")
        return vector
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_ast_processor() 