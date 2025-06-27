"""
JSON-based storage system for code embeddings and similarity results.
"""

import json
import os
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)


class JSONStorage:
    """Simple JSON-based storage for code embeddings and results."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize JSON storage.
        
        Args:
            data_dir: Directory to store JSON files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # File paths
        self.files_path = self.data_dir / "files.json"
        self.embeddings_path = self.data_dir / "embeddings.json"
        self.similarity_path = self.data_dir / "similarity_results.json"
        self.vectors_path = self.data_dir / "vectors"
        self.vectors_path.mkdir(exist_ok=True)
        
        # Initialize files if they don't exist
        self._init_files()
    
    def _init_files(self):
        """Initialize JSON files if they don't exist."""
        if not self.files_path.exists():
            self._save_json(self.files_path, {"files": {}})
        
        if not self.embeddings_path.exists():
            self._save_json(self.embeddings_path, {"embeddings": {}})
        
        if not self.similarity_path.exists():
            self._save_json(self.similarity_path, {"results": {}})
    
    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_json(self, file_path: Path, data: Dict[str, Any]):
        """Save data to JSON file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving to {file_path}: {str(e)}")
            raise
    
    def _generate_id(self) -> str:
        """Generate a unique ID."""
        return hashlib.md5(f"{datetime.now().isoformat()}".encode()).hexdigest()[:12]
    
    def save_code_file(self, file_path: str, content: str, language: str) -> str:
        """
        Save a code file record.
        
        Args:
            file_path: Path to the file
            content: File content
            language: Programming language
            
        Returns:
            File ID
        """
        file_hash = hashlib.md5(content.encode()).hexdigest()
        file_id = self._generate_id()
        
        file_data = {
            "id": file_id,
            "file_path": file_path,
            "file_hash": file_hash,
            "language": language,
            "content": content,
            "created_at": datetime.now().isoformat(),
            "size_bytes": len(content.encode())
        }
        
        data = self._load_json(self.files_path)
        data["files"][file_id] = file_data
        self._save_json(self.files_path, data)
        
        logger.info(f"Saved code file: {file_id}")
        return file_id
    
    def get_code_file(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a code file by ID.
        
        Args:
            file_id: File ID
            
        Returns:
            File data or None
        """
        data = self._load_json(self.files_path)
        return data.get("files", {}).get(file_id)
    
    def get_code_file_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get a code file by hash.
        
        Args:
            file_hash: File hash
            
        Returns:
            File data or None
        """
        data = self._load_json(self.files_path)
        for file_data in data.get("files", {}).values():
            if file_data.get("file_hash") == file_hash:
                return file_data
        return None
    
    def save_embedding(self, file_id: str, model_name: str, embedding_vector: np.ndarray) -> str:
        """
        Save an embedding.
        
        Args:
            file_id: File ID
            model_name: Name of the model
            embedding_vector: Embedding vector
            
        Returns:
            Embedding ID
        """
        embedding_id = self._generate_id()
        
        # Save vector to separate file
        vector_file = self.vectors_path / f"{embedding_id}.npy"
        np.save(vector_file, embedding_vector)
        
        embedding_data = {
            "id": embedding_id,
            "file_id": file_id,
            "model_name": model_name,
            "embedding_dim": len(embedding_vector),
            "vector_file": str(vector_file),
            "created_at": datetime.now().isoformat()
        }
        
        data = self._load_json(self.embeddings_path)
        data["embeddings"][embedding_id] = embedding_data
        self._save_json(self.embeddings_path, data)
        
        logger.info(f"Saved embedding: {embedding_id}")
        return embedding_id
    
    def get_embedding(self, embedding_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an embedding by ID.
        
        Args:
            embedding_id: Embedding ID
            
        Returns:
            Embedding data or None
        """
        data = self._load_json(self.embeddings_path)
        return data.get("embeddings", {}).get(embedding_id)
    
    def load_embedding_vector(self, embedding_id: str) -> Optional[np.ndarray]:
        """
        Load embedding vector from file.
        
        Args:
            embedding_id: Embedding ID
            
        Returns:
            Embedding vector or None
        """
        embedding_data = self.get_embedding(embedding_id)
        if not embedding_data:
            return None
        
        vector_file = Path(embedding_data["vector_file"])
        if vector_file.exists():
            return np.load(vector_file)
        return None
    
    def get_embeddings_by_file(self, file_id: str) -> List[Dict[str, Any]]:
        """
        Get all embeddings for a file.
        
        Args:
            file_id: File ID
            
        Returns:
            List of embedding data
        """
        data = self._load_json(self.embeddings_path)
        return [
            embedding for embedding in data.get("embeddings", {}).values()
            if embedding.get("file_id") == file_id
        ]
    
    def get_embeddings_by_model(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Get all embeddings for a model.
        
        Args:
            model_name: Model name
            
        Returns:
            List of embedding data
        """
        data = self._load_json(self.embeddings_path)
        return [
            embedding for embedding in data.get("embeddings", {}).values()
            if embedding.get("model_name") == model_name
        ]
    
    def save_similarity_result(self, file_id: str, model_name: str, 
                             similar_files: List[Dict[str, Any]]) -> str:
        """
        Save similarity results.
        
        Args:
            file_id: File ID
            model_name: Model name
            similar_files: List of similar files with scores
            
        Returns:
            Result ID
        """
        result_id = self._generate_id()
        
        result_data = {
            "id": result_id,
            "file_id": file_id,
            "model_name": model_name,
            "similar_files": similar_files,
            "created_at": datetime.now().isoformat()
        }
        
        data = self._load_json(self.similarity_path)
        data["results"][result_id] = result_data
        self._save_json(self.similarity_path, data)
        
        logger.info(f"Saved similarity result: {result_id}")
        return result_id
    
    def get_similarity_results(self, file_id: Optional[str] = None, 
                             model_name: Optional[str] = None,
                             limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get similarity results.
        
        Args:
            file_id: Filter by file ID
            model_name: Filter by model name
            limit: Maximum number of results
            
        Returns:
            List of similarity results
        """
        data = self._load_json(self.similarity_path)
        results = list(data.get("results", {}).values())
        
        # Apply filters
        if file_id:
            results = [r for r in results if r.get("file_id") == file_id]
        if model_name:
            results = [r for r in results if r.get("model_name") == model_name]
        
        # Sort by creation date (newest first)
        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return results[:limit]
    
    def cleanup_old_results(self, days_old: int = 30):
        """
        Clean up old results and unused vector files.
        
        Args:
            days_old: Remove results older than this many days
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        # Clean up similarity results
        data = self._load_json(self.similarity_path)
        old_results = []
        for result_id, result_data in data.get("results", {}).items():
            created_at = datetime.fromisoformat(result_data.get("created_at", ""))
            if created_at < cutoff_date:
                old_results.append(result_id)
        
        for result_id in old_results:
            del data["results"][result_id]
        
        self._save_json(self.similarity_path, data)
        
        # Clean up unused vector files
        data = self._load_json(self.embeddings_path)
        used_vectors = set()
        for embedding_data in data.get("embeddings", {}).values():
            vector_file = embedding_data.get("vector_file")
            if vector_file:
                used_vectors.add(Path(vector_file).name)
        
        for vector_file in self.vectors_path.glob("*.npy"):
            if vector_file.name not in used_vectors:
                vector_file.unlink()
                logger.info(f"Removed unused vector file: {vector_file}")
        
        logger.info(f"Cleaned up {len(old_results)} old results")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        files_data = self._load_json(self.files_path)
        embeddings_data = self._load_json(self.embeddings_path)
        results_data = self._load_json(self.similarity_path)
        
        vector_files = list(self.vectors_path.glob("*.npy"))
        
        return {
            "total_files": len(files_data.get("files", {})),
            "total_embeddings": len(embeddings_data.get("embeddings", {})),
            "total_results": len(results_data.get("results", {})),
            "total_vector_files": len(vector_files),
            "storage_size_mb": sum(f.stat().st_size for f in vector_files) / (1024 * 1024)
        } 