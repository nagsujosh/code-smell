"""
AST-based graph processor for code embedding.
"""

import ast
import logging
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class ASTGraphProcessor:
    """Convert Python code to graph representation using AST."""
    
    def __init__(self):
        """Initialize the AST graph processor."""
        self.node_types = {
            'FunctionDef': 'function',
            'ClassDef': 'class',
            'Import': 'import',
            'ImportFrom': 'import',
            'Assign': 'assignment',
            'AugAssign': 'assignment',
            'For': 'loop',
            'While': 'loop',
            'If': 'conditional',
            'Try': 'exception',
            'Call': 'function_call',
            'Name': 'variable',
            'Attribute': 'attribute',
            'Constant': 'constant',
            'List': 'data_structure',
            'Dict': 'data_structure',
            'Tuple': 'data_structure',
            'Set': 'data_structure',
            'Return': 'control_flow',
            'Break': 'control_flow',
            'Continue': 'control_flow',
            'Raise': 'control_flow',
            'With': 'context',
            'AsyncFunctionDef': 'function',
            'AsyncFor': 'loop',
            'AsyncWith': 'context',
        }
        
        self.edge_types = {
            'calls': 'calls',
            'contains': 'contains',
            'uses': 'uses',
            'defines': 'defines',
            'inherits': 'inherits',
            'imports': 'imports',
            'assigns': 'assigns',
            'returns': 'returns',
            'raises': 'raises',
        }
    
    def parse_to_graph(self, code: str) -> nx.DiGraph:
        """
        Parse Python code and convert to a directed graph.
        
        Args:
            code: Python source code
            
        Returns:
            NetworkX directed graph representing the code structure
        """
        try:
            tree = ast.parse(code)
            graph = nx.DiGraph()
            
            # Add root node
            graph.add_node('root', type='module', name='root', line=0)
            
            # Process AST nodes
            self._process_node(tree, graph, 'root')
            
            return graph
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in code: {e}")
            # Return empty graph for invalid code
            return nx.DiGraph()
        except Exception as e:
            logger.error(f"Error parsing code to graph: {e}")
            return nx.DiGraph()
    
    def _process_node(self, node: ast.AST, graph: nx.DiGraph, parent_id: str):
        """Process an AST node and add it to the graph."""
        if node is None:
            return
        
        # Create node ID
        node_id = f"{type(node).__name__}_{id(node)}"
        
        # Get node type
        node_type = self.node_types.get(type(node).__name__, 'unknown')
        
        # Extract node information
        node_info = self._extract_node_info(node, node_type)
        
        # Add node to graph
        graph.add_node(node_id, **node_info)
        
        # Add edge from parent
        if parent_id != node_id:
            graph.add_edge(parent_id, node_id, type='contains')
        
        # Process child nodes based on node type
        self._process_children(node, graph, node_id)
    
    def _extract_node_info(self, node: ast.AST, node_type: str) -> Dict[str, Any]:
        """Extract information from an AST node."""
        info = {
            'type': node_type,
            'ast_type': type(node).__name__,
            'line': getattr(node, 'lineno', 0),
            'col': getattr(node, 'col_offset', 0),
        }
        
        # Extract specific information based on node type
        if isinstance(node, ast.FunctionDef):
            info.update({
                'name': node.name,
                'args': [arg.arg for arg in node.args.args],
                'decorators': [self._get_decorator_name(d) for d in node.decorator_list]
            })
        elif isinstance(node, ast.ClassDef):
            info.update({
                'name': node.name,
                'bases': [self._get_base_name(base) for base in node.bases],
                'decorators': [self._get_decorator_name(d) for d in node.decorator_list]
            })
        elif isinstance(node, ast.Name):
            info.update({
                'name': node.id,
                'ctx': type(node.ctx).__name__
            })
        elif isinstance(node, ast.Call):
            info.update({
                'func_name': self._get_call_name(node.func),
                'args_count': len(node.args),
                'keywords_count': len(node.keywords)
            })
        elif isinstance(node, ast.Import):
            info.update({
                'names': [alias.name for alias in node.names]
            })
        elif isinstance(node, ast.ImportFrom):
            info.update({
                'module': node.module,
                'names': [alias.name for alias in node.names]
            })
        elif isinstance(node, ast.Assign):
            info.update({
                'targets_count': len(node.targets),
                'value_type': type(node.value).__name__
            })
        elif isinstance(node, ast.Constant):
            info.update({
                'value_type': type(node.value).__name__,
                'value_repr': str(node.value)[:50]  # Truncate long values
            })
        
        return info
    
    def _process_children(self, node: ast.AST, graph: nx.DiGraph, node_id: str):
        """Process child nodes of an AST node."""
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for child in value:
                    if isinstance(child, ast.AST):
                        self._process_node(child, graph, node_id)
            elif isinstance(value, ast.AST):
                self._process_node(value, graph, node_id)
    
    def _get_decorator_name(self, decorator: ast.AST) -> str:
        """Get the name of a decorator."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return self._get_attribute_name(decorator)
        elif isinstance(decorator, ast.Call):
            return self._get_call_name(decorator.func)
        return 'unknown'
    
    def _get_base_name(self, base: ast.AST) -> str:
        """Get the name of a base class."""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return self._get_attribute_name(base)
        return 'unknown'
    
    def _get_call_name(self, func: ast.AST) -> str:
        """Get the name of a function call."""
        if isinstance(func, ast.Name):
            return func.id
        elif isinstance(func, ast.Attribute):
            return self._get_attribute_name(func)
        return 'unknown'
    
    def _get_attribute_name(self, attr: ast.Attribute) -> str:
        """Get the full name of an attribute."""
        if isinstance(attr.value, ast.Name):
            return f"{attr.value.id}.{attr.attr}"
        elif isinstance(attr.value, ast.Attribute):
            return f"{self._get_attribute_name(attr.value)}.{attr.attr}"
        return attr.attr
    
    def graph_to_features(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Convert graph to feature dictionary for embedding.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary of graph features
        """
        if not graph.nodes():
            return self._empty_features()
        
        features = {}
        
        # Node type distribution
        node_types = [data.get('type', 'unknown') for _, data in graph.nodes(data=True)]
        features['node_type_distribution'] = self._count_distribution(node_types)
        
        # Function and class information
        functions = [data for _, data in graph.nodes(data=True) if data.get('type') == 'function']
        classes = [data for _, data in graph.nodes(data=True) if data.get('type') == 'class']
        
        features['function_count'] = len(functions)
        features['class_count'] = len(classes)
        features['function_names'] = [f.get('name', '') for f in functions]
        features['class_names'] = [c.get('name', '') for c in classes]
        
        # Import information
        imports = [data for _, data in graph.nodes(data=True) if data.get('type') == 'import']
        features['import_count'] = len(imports)
        features['import_names'] = []
        for imp in imports:
            if 'names' in imp:
                features['import_names'].extend(imp['names'])
        
        # Graph structure metrics
        features['total_nodes'] = graph.number_of_nodes()
        features['total_edges'] = graph.number_of_edges()
        features['max_depth'] = self._calculate_max_depth(graph)
        features['avg_degree'] = sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0
        
        # Edge type distribution
        edge_types = [data.get('type', 'unknown') for _, _, data in graph.edges(data=True)]
        features['edge_type_distribution'] = self._count_distribution(edge_types)
        
        # Code complexity metrics
        features['complexity_metrics'] = self._calculate_complexity_metrics(graph)
        
        return features
    
    def _empty_features(self) -> Dict[str, Any]:
        """Return empty feature dictionary."""
        return {
            'node_type_distribution': {},
            'function_count': 0,
            'class_count': 0,
            'function_names': [],
            'class_names': [],
            'import_count': 0,
            'import_names': [],
            'total_nodes': 0,
            'total_edges': 0,
            'max_depth': 0,
            'avg_degree': 0,
            'edge_type_distribution': {},
            'complexity_metrics': {}
        }
    
    def _count_distribution(self, items: List[str]) -> Dict[str, int]:
        """Count distribution of items."""
        distribution = defaultdict(int)
        for item in items:
            distribution[item] += 1
        return dict(distribution)
    
    def _calculate_max_depth(self, graph: nx.DiGraph) -> int:
        """Calculate maximum depth of the graph."""
        if not graph.nodes():
            return 0
        
        depths = {}
        for node in graph.nodes():
            depths[node] = len(list(nx.shortest_path(graph, 'root', node))) - 1
        
        return max(depths.values()) if depths else 0
    
    def _calculate_complexity_metrics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Calculate code complexity metrics."""
        metrics = {}
        
        # Count different types of nodes
        node_types = [data.get('type', 'unknown') for _, data in graph.nodes(data=True)]
        type_counts = self._count_distribution(node_types)
        
        metrics['control_flow_count'] = sum(type_counts.get(t, 0) for t in ['conditional', 'loop', 'exception'])
        metrics['data_structure_count'] = sum(type_counts.get(t, 0) for t in ['data_structure'])
        metrics['function_call_count'] = type_counts.get('function_call', 0)
        metrics['assignment_count'] = type_counts.get('assignment', 0)
        
        # Calculate cyclomatic complexity (simplified)
        metrics['cyclomatic_complexity'] = (
            metrics['control_flow_count'] + 
            metrics['function_call_count'] + 
            1  # Base complexity
        )
        
        return metrics
    
    def features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Convert features to a numerical vector for embedding.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Numerical vector representation
        """
        vector = []
        
        # Basic counts
        vector.extend([
            features['function_count'],
            features['class_count'],
            features['import_count'],
            features['total_nodes'],
            features['total_edges'],
            features['max_depth'],
            features['avg_degree']
        ])
        
        # Complexity metrics
        complexity = features['complexity_metrics']
        vector.extend([
            complexity['control_flow_count'],
            complexity['data_structure_count'],
            complexity['function_call_count'],
            complexity['assignment_count'],
            complexity['cyclomatic_complexity']
        ])
        
        # Node type distribution (normalized)
        node_dist = features['node_type_distribution']
        total_nodes = features['total_nodes'] or 1
        for node_type in ['function', 'class', 'import', 'assignment', 'loop', 'conditional', 'function_call', 'variable', 'constant']:
            vector.append(node_dist.get(node_type, 0) / total_nodes)
        
        # Edge type distribution (normalized)
        edge_dist = features['edge_type_distribution']
        total_edges = features['total_edges'] or 1
        for edge_type in ['contains', 'calls', 'uses', 'defines', 'imports', 'assigns']:
            vector.append(edge_dist.get(edge_type, 0) / total_edges)
        
        return np.array(vector, dtype=np.float32)
    
    def process_code(self, code: str) -> np.ndarray:
        """
        Process code and return vector representation.
        
        Args:
            code: Python source code
            
        Returns:
            Vector representation of the code
        """
        graph = self.parse_to_graph(code)
        features = self.graph_to_features(graph)
        return self.features_to_vector(features) 