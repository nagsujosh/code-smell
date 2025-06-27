"""
Additional similarity metrics for code comparison.
"""

import numpy as np
from typing import Dict, Any


class SimilarityMetrics:
    """Additional metrics for code similarity analysis."""
    
    def __init__(self):
        """Initialize the metrics calculator."""
        pass
    
    def calculate_metrics(self, embedding1: np.ndarray, embedding2: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Dictionary with various metrics
        """
        metrics = {}
        
        # Flatten embeddings for calculations
        emb1_flat = embedding1.flatten()
        emb2_flat = embedding2.flatten()
        
        # Basic statistics
        metrics['mean_diff'] = float(np.mean(np.abs(emb1_flat - emb2_flat)))
        metrics['std_diff'] = float(np.std(emb1_flat - emb2_flat))
        metrics['max_diff'] = float(np.max(np.abs(emb1_flat - emb2_flat)))
        metrics['min_diff'] = float(np.min(np.abs(emb1_flat - emb2_flat)))
        
        # Correlation
        metrics['pearson_correlation'] = float(np.corrcoef(emb1_flat, emb2_flat)[0, 1])
        
        # Vector norms
        metrics['norm1'] = float(np.linalg.norm(emb1_flat))
        metrics['norm2'] = float(np.linalg.norm(emb2_flat))
        metrics['norm_ratio'] = float(metrics['norm1'] / metrics['norm2']) if metrics['norm2'] != 0 else 0
        
        # Angle between vectors
        dot_product = np.dot(emb1_flat, emb2_flat)
        angle = np.arccos(np.clip(dot_product / (metrics['norm1'] * metrics['norm2']), -1.0, 1.0))
        metrics['angle_radians'] = float(angle)
        metrics['angle_degrees'] = float(np.degrees(angle))
        
        # Percentage of similar dimensions
        threshold = 0.1  # Small threshold for considering dimensions similar
        similar_dims = np.sum(np.abs(emb1_flat - emb2_flat) < threshold)
        metrics['similar_dimensions_percentage'] = float(similar_dims / len(emb1_flat) * 100)
        
        # Entropy-based metrics
        metrics['entropy1'] = float(self._calculate_entropy(emb1_flat))
        metrics['entropy2'] = float(self._calculate_entropy(emb2_flat))
        metrics['entropy_diff'] = float(abs(metrics['entropy1'] - metrics['entropy2']))
        
        return metrics
    
    def _calculate_entropy(self, vector: np.ndarray) -> float:
        """
        Calculate entropy of a vector.
        
        Args:
            vector: Input vector
            
        Returns:
            Entropy value
        """
        # Normalize to probability distribution
        abs_vector = np.abs(vector)
        if np.sum(abs_vector) == 0:
            return 0
        
        prob_dist = abs_vector / np.sum(abs_vector)
        
        # Calculate entropy
        entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))
        return entropy
    
    def calculate_dimensionality_metrics(self, embeddings: list) -> Dict[str, Any]:
        """
        Calculate metrics related to embedding dimensionality.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Dictionary with dimensionality metrics
        """
        if not embeddings:
            return {}
        
        # Convert to array
        emb_array = np.array([emb.flatten() for emb in embeddings])
        
        metrics = {
            'num_embeddings': len(embeddings),
            'embedding_dimension': emb_array.shape[1],
            'mean_embedding_norm': float(np.mean([np.linalg.norm(emb) for emb in embeddings])),
            'std_embedding_norm': float(np.std([np.linalg.norm(emb) for emb in embeddings])),
            'embedding_variance': float(np.var(emb_array)),
            'embedding_std': float(np.std(emb_array)),
        }
        
        return metrics
    
    def calculate_clustering_metrics(self, embeddings: list, labels: list = None) -> Dict[str, Any]:
        """
        Calculate clustering-related metrics.
        
        Args:
            embeddings: List of embedding vectors
            labels: Optional cluster labels
            
        Returns:
            Dictionary with clustering metrics
        """
        if len(embeddings) < 2:
            return {}
        
        # Convert to array
        emb_array = np.array([emb.flatten() for emb in embeddings])
        
        metrics = {
            'num_embeddings': len(embeddings),
            'embedding_dimension': emb_array.shape[1],
        }
        
        # Calculate pairwise distances
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(emb_array)
        
        # Distance statistics
        upper_tri = distances[np.triu_indices_from(distances, k=1)]
        metrics['mean_pairwise_distance'] = float(np.mean(upper_tri))
        metrics['std_pairwise_distance'] = float(np.std(upper_tri))
        metrics['min_pairwise_distance'] = float(np.min(upper_tri))
        metrics['max_pairwise_distance'] = float(np.max(upper_tri))
        
        # Silhouette score if labels are provided
        if labels and len(set(labels)) > 1:
            from sklearn.metrics import silhouette_score
            try:
                metrics['silhouette_score'] = float(silhouette_score(emb_array, labels))
            except:
                metrics['silhouette_score'] = None
        
        return metrics
    
    def calculate_distribution_metrics(self, embeddings: list) -> Dict[str, Any]:
        """
        Calculate distribution-related metrics.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Dictionary with distribution metrics
        """
        if not embeddings:
            return {}
        
        # Convert to array
        emb_array = np.array([emb.flatten() for emb in embeddings])
        
        metrics = {
            'num_embeddings': len(embeddings),
            'embedding_dimension': emb_array.shape[1],
            'global_mean': float(np.mean(emb_array)),
            'global_std': float(np.std(emb_array)),
            'global_min': float(np.min(emb_array)),
            'global_max': float(np.max(emb_array)),
            'global_median': float(np.median(emb_array)),
        }
        
        # Per-dimension statistics
        dim_means = np.mean(emb_array, axis=0)
        dim_stds = np.std(emb_array, axis=0)
        
        metrics['dimension_mean_mean'] = float(np.mean(dim_means))
        metrics['dimension_mean_std'] = float(np.std(dim_means))
        metrics['dimension_std_mean'] = float(np.mean(dim_stds))
        metrics['dimension_std_std'] = float(np.std(dim_stds))
        
        return metrics 