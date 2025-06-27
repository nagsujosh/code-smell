"""
Similarity calculation for code embeddings.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import cosine, euclidean

from src.similarity.metrics import SimilarityMetrics

logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """Calculate similarity between code embeddings."""
    
    def __init__(self, threshold: float = 0.8):
        """
        Initialize the similarity calculator.
        
        Args:
            threshold: Similarity threshold for considering codes similar
        """
        self.threshold = threshold
        self.metrics = SimilarityMetrics()
    
    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Ensure embeddings are 2D for sklearn
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        
        # Ensure result is in [0, 1] range
        return max(0.0, min(1.0, similarity))
    
    def euclidean_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate Euclidean similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Euclidean similarity score (0-1, where 1 is most similar)
        """
        # Calculate Euclidean distance
        distance = euclidean(embedding1.flatten(), embedding2.flatten())
        
        # Convert to similarity (inverse of distance, normalized)
        # We use a simple normalization - you might want to adjust this
        max_distance = np.sqrt(embedding1.size)  # Maximum possible distance
        similarity = 1.0 - (distance / max_distance)
        
        return max(0.0, min(1.0, similarity))
    
    def manhattan_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate Manhattan similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Manhattan similarity score (0-1, where 1 is most similar)
        """
        # Calculate Manhattan distance
        distance = np.sum(np.abs(embedding1.flatten() - embedding2.flatten()))
        
        # Convert to similarity
        max_distance = np.sum(np.abs(embedding1.flatten())) + np.sum(np.abs(embedding2.flatten()))
        similarity = 1.0 - (distance / max_distance) if max_distance > 0 else 1.0
        
        return max(0.0, min(1.0, similarity))
    
    def compare_embeddings(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                          methods: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compare two embeddings using multiple similarity metrics.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            methods: List of similarity methods to use
            
        Returns:
            Dictionary with similarity scores for each method
        """
        if methods is None:
            methods = ['cosine', 'euclidean', 'manhattan']
        
        results = {}
        
        for method in methods:
            if method == 'cosine':
                results['cosine'] = self.cosine_similarity(embedding1, embedding2)
            elif method == 'euclidean':
                results['euclidean'] = self.euclidean_similarity(embedding1, embedding2)
            elif method == 'manhattan':
                results['manhattan'] = self.manhattan_similarity(embedding1, embedding2)
            else:
                logger.warning(f"Unknown similarity method: {method}")
        
        return results
    
    def is_similar(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                   method: str = 'cosine') -> bool:
        """
        Check if two embeddings are similar based on threshold.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            method: Similarity method to use
            
        Returns:
            True if embeddings are similar, False otherwise
        """
        if method == 'cosine':
            similarity = self.cosine_similarity(embedding1, embedding2)
        elif method == 'euclidean':
            similarity = self.euclidean_similarity(embedding1, embedding2)
        elif method == 'manhattan':
            similarity = self.manhattan_similarity(embedding1, embedding2)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        return similarity >= self.threshold
    
    def batch_similarity(self, embeddings: List[np.ndarray], 
                        method: str = 'cosine') -> np.ndarray:
        """
        Calculate similarity matrix for a batch of embeddings.
        
        Args:
            embeddings: List of embedding vectors
            method: Similarity method to use
            
        Returns:
            Similarity matrix
        """
        if not embeddings:
            return np.array([])
        
        # Convert to 2D array
        embeddings_array = np.array([emb.flatten() for emb in embeddings])
        
        if method == 'cosine':
            return cosine_similarity(embeddings_array)
        elif method == 'euclidean':
            distances = euclidean_distances(embeddings_array)
            # Convert distances to similarities
            max_distance = np.max(distances)
            if max_distance > 0:
                return 1.0 - (distances / max_distance)
            else:
                return np.ones_like(distances)
        else:
            raise ValueError(f"Batch similarity not supported for method: {method}")
    
    def find_similar_pairs(self, embeddings: List[np.ndarray], 
                          file_paths: Optional[List[Path]] = None,
                          method: str = 'cosine') -> List[Dict[str, Any]]:
        """
        Find pairs of similar embeddings.
        
        Args:
            embeddings: List of embedding vectors
            file_paths: List of corresponding file paths
            method: Similarity method to use
            
        Returns:
            List of similar pairs with their similarity scores
        """
        if len(embeddings) < 2:
            return []
        
        # Calculate similarity matrix
        similarity_matrix = self.batch_similarity(embeddings, method)
        
        similar_pairs = []
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = similarity_matrix[i][j]
                
                if similarity >= self.threshold:
                    pair_info = {
                        'index1': i,
                        'index2': j,
                        'similarity': float(similarity),
                        'method': method,
                        'is_similar': True
                    }
                    
                    if file_paths:
                        pair_info['file1'] = str(file_paths[i])
                        pair_info['file2'] = str(file_paths[j])
                    
                    similar_pairs.append(pair_info)
        
        # Sort by similarity score (descending)
        similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similar_pairs
    
    def compare_files(self, file1_path: Path, file2_path: Path, 
                     embedding1: np.ndarray, embedding2: np.ndarray,
                     method: str = 'cosine') -> Dict[str, Any]:
        """
        Compare two code files and return detailed similarity information.
        
        Args:
            file1_path: Path to first file
            file2_path: Path to second file
            embedding1: Embedding of first file
            embedding2: Embedding of second file
            method: Similarity method to use
            
        Returns:
            Dictionary with comparison results
        """
        # Calculate similarity
        if method == 'cosine':
            similarity = self.cosine_similarity(embedding1, embedding2)
        elif method == 'euclidean':
            similarity = self.euclidean_similarity(embedding1, embedding2)
        elif method == 'manhattan':
            similarity = self.manhattan_similarity(embedding1, embedding2)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        # Determine if files are similar
        is_similar = similarity >= self.threshold
        
        # Calculate additional metrics
        additional_metrics = self.metrics.calculate_metrics(embedding1, embedding2)
        
        result = {
            'file1': str(file1_path),
            'file2': str(file2_path),
            'similarity_score': float(similarity),
            'similarity_method': method,
            'is_similar': is_similar,
            'threshold': self.threshold,
            'additional_metrics': additional_metrics
        }
        
        return result
    
    def set_threshold(self, threshold: float):
        """
        Set the similarity threshold.
        
        Args:
            threshold: New threshold value (0-1)
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        self.threshold = threshold
        logger.info(f"Similarity threshold set to {threshold}")
    
    def get_threshold(self) -> float:
        """Get the current similarity threshold."""
        return self.threshold 