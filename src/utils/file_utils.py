"""
File utility functions for code similarity detection.
"""

import os
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np


class FileUtils:
    """Utility class for file operations."""
    
    def __init__(self):
        """Initialize the file utils."""
        pass
    
    def detect_language(self, file_path: Union[str, Path]) -> str:
        """
        Detect the programming language from file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Language name
        """
        extension = Path(file_path).suffix.lower()
        
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.cs': 'csharp',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.r': 'r',
            '.m': 'matlab',
            '.sh': 'bash',
            '.sql': 'sql',
        }
        
        return language_map.get(extension, 'unknown')
    
    def get_file_hash(self, file_path: Union[str, Path]) -> str:
        """
        Calculate SHA-256 hash of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            SHA-256 hash string
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get comprehensive information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get file stats
        stat = file_path.stat()
        
        # Detect language from extension
        language = self.detect_language(file_path)
        
        # Calculate hash
        file_hash = self.get_file_hash(file_path)
        
        info = {
            "file_path": str(file_path.absolute()),
            "file_name": file_path.name,
            "file_extension": file_path.suffix,
            "language": language,
            "size_bytes": stat.st_size,
            "modified_time": stat.st_mtime,
            "created_time": stat.st_ctime,
            "file_hash": file_hash,
            "content_length": len(content),
            "content": content
        }
        
        return info
    
    def save_embedding(self, embedding: np.ndarray, file_path: Union[str, Path], 
                      model_name: str, output_dir: Union[str, Path]) -> Path:
        """
        Save an embedding to disk.
        
        Args:
            embedding: Embedding vector
            file_path: Original file path
            model_name: Name of the model used
            output_dir: Directory to save the embedding
            
        Returns:
            Path to the saved embedding file
        """
        file_path = Path(file_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename based on file hash and model
        file_hash = self.get_file_hash(file_path)
        embedding_filename = f"{file_hash}_{model_name}.npy"
        embedding_path = output_dir / embedding_filename
        
        # Save embedding
        np.save(embedding_path, embedding)
        
        # Save metadata
        import datetime
        metadata = {
            "original_file": str(file_path.absolute()),
            "model_name": model_name,
            "embedding_shape": embedding.shape,
            "file_hash": file_hash,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        metadata_path = embedding_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return embedding_path
    
    def load_embedding(self, embedding_path: Union[str, Path]) -> tuple:
        """
        Load an embedding from disk.
        
        Args:
            embedding_path: Path to the embedding file
            
        Returns:
            Tuple of (embedding, metadata)
        """
        embedding_path = Path(embedding_path)
        
        if not embedding_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        
        # Load embedding
        embedding = np.load(embedding_path)
        
        # Load metadata
        metadata_path = embedding_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        return embedding, metadata
    
    def find_code_files(self, directory: Union[str, Path], 
                       extensions: Optional[List[str]] = None) -> List[Path]:
        """
        Find all code files in a directory.
        
        Args:
            directory: Directory to search
            extensions: List of file extensions to include (None for all)
            
        Returns:
            List of file paths
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if extensions is None:
            extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', 
                         '.rs', '.php', '.rb', '.cs', '.swift', '.kt', '.scala']
        
        files = []
        for ext in extensions:
            files.extend(directory.rglob(f"*{ext}"))
        
        return sorted(files)
    
    def save_similarity_results(self, results: List[Dict[str, Any]], 
                               output_path: Union[str, Path]):
        """
        Save similarity results to a file.
        
        Args:
            results: List of similarity result dictionaries
            output_path: Path to save the results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(results)
        
        # Save as CSV
        df.to_csv(output_path.with_suffix('.csv'), index=False)
        
        # Save as JSON for more detailed format
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(results, f, indent=2)
    
    def load_similarity_results(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load similarity results from a file.
        
        Args:
            file_path: Path to the results file
            
        Returns:
            List of similarity result dictionaries
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Results file not found: {file_path}")
        
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            return df.to_dict('records')
        elif file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def create_sample_files(self, output_dir: Union[str, Path]):
        """
        Create sample code files for testing.
        
        Args:
            output_dir: Directory to create sample files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample 1: Dijkstra's algorithm (original)
        dijkstra_original = '''import heapq

def dijkstra(graph, source):
    distance = {node: float('inf') for node in graph}
    distance[source] = 0
    visited = set()
    min_heap = [(0, source)]

    while min_heap:
        curr_dist, curr_node = heapq.heappop(min_heap)
        if curr_node in visited:
            continue
        visited.add(curr_node)

        for neighbor, weight in graph[curr_node]:
            new_dist = curr_dist + weight
            if new_dist < distance[neighbor]:
                distance[neighbor] = new_dist
                heapq.heappush(min_heap, (new_dist, neighbor))

    return distance

if __name__ == "__main__":
    g = {
        'A': [('B', 1), ('C', 4)],
        'B': [('C', 2), ('D', 5)],
        'C': [('D', 1)],
        'D': []
    }

    src = 'A'
    shortest = dijkstra(g, src)
    print(f"Shortest distances from {src}:")
    for node in sorted(shortest):
        print(f"{node}: {shortest[node]}")'''
        
        # Sample 2: Dijkstra's algorithm (modified)
        dijkstra_modified = '''import heapq

def shortest_path(graph_data, start_node):
    # initialize distances to all nodes
    dist_map = {key: float("inf") for key in graph_data}
    dist_map[start_node] = 0
    visited_set = set()
    pq = [(0, start_node)]

    while pq:
        dist, node = heapq.heappop(pq)
        if node in visited_set:
            continue
        visited_set.add(node)

        for adj, cost in graph_data[node]:
            temp = dist + cost
            if temp < dist_map[adj]:
                dist_map[adj] = temp
                heapq.heappush(pq, (temp, adj))

    return dist_map

if __name__ == "__main__":
    network = {
        'A': [('B', 1), ('C', 4)],
        'B': [('C', 2), ('D', 5)],
        'C': [('D', 1)],
        'D': []
    }

    origin = 'A'
    result = shortest_path(network, origin)
    print("From", origin)
    for loc in sorted(result):
        print(f"{loc}: {result[loc]}")'''
        
        # Sample 3: Different algorithm (BFS)
        bfs_algorithm = '''from collections import deque

def breadth_first_search(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    result = []
    
    while queue:
        vertex = queue.popleft()
        result.append(vertex)
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result

if __name__ == "__main__":
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }
    
    start_node = 'A'
    traversal = breadth_first_search(graph, start_node)
    print(f"BFS traversal starting from {start_node}: {traversal}")'''
        
        # Write sample files
        samples = [
            ("dijkstra_original.py", dijkstra_original),
            ("dijkstra_modified.py", dijkstra_modified),
            ("bfs_algorithm.py", bfs_algorithm)
        ]
        
        for filename, content in samples:
            file_path = output_dir / filename
            with open(file_path, 'w') as f:
                f.write(content)
        
        print(f"Created {len(samples)} sample files in {output_dir}")
    
    def get_file_size_mb(self, file_path: Union[str, Path]) -> float:
        """
        Get file size in megabytes.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File size in MB
        """
        file_path = Path(file_path)
        return file_path.stat().st_size / (1024 * 1024) 