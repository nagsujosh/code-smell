"""
Code embedding functionality using AST-based graph representation and CodeBERT.
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from src.utils.code_cleaner import clean_code
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    T5EncoderModel,
    pipeline
)

from src.models.model_registry import ModelRegistry, ModelConfig, ModelType
from src.utils.ast_graph_processor import ASTGraphProcessor


logger = logging.getLogger(__name__)


class CodeEmbedder:
    """Main class for embedding code using AST-based graph representation and CodeBERT."""
    
    def __init__(
        self, 
        model_name: str = "codebert",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_ast_features: bool = True,
        use_bert_features: bool = True
    ):
        """
        Initialize the code embedder.
        
        Args:
            model_name: Name of the model to use (from ModelRegistry)
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto)
            cache_dir: Directory to cache models
            use_ast_features: Whether to use AST graph features
            use_bert_features: Whether to use BERT embeddings
        """
        self.model_name = model_name
        self.config = ModelRegistry.get_model(model_name)
        self.use_ast_features = use_ast_features
        self.use_bert_features = use_bert_features
        
        # Set device
        if device is None:
            self.device = self._get_optimal_device()
        else:
            self.device = device
            
        self.cache_dir = cache_dir or os.getenv("MODEL_CACHE_DIR", "./model_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.ast_processor = ASTGraphProcessor()
        
        # Load model if using BERT features
        if self.use_bert_features:
            self._load_model()
        
        # Calculate embedding dimension
        self.embedding_dim = self._calculate_embedding_dim()
        
    def _get_optimal_device(self) -> str:
        """Get the optimal device for the model."""
        if self.config.requires_gpu:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                logger.warning(f"Model {self.model_name} requires GPU but none available. Using CPU.")
                return "cpu"
        else:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
    
    def _load_model(self):
        """Load the specified model and tokenizer."""
        logger.info(f"Loading model: {self.config.name} on device: {self.device}")
        
        try:
            if self.config.type == ModelType.BERT:
                self._load_bert_model()
            else:
                raise ValueError(f"Unsupported model type: {self.config.type}")
                
            logger.info(f"Successfully loaded {self.config.name}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.config.name}: {str(e)}")
            raise
    
    def _calculate_embedding_dim(self) -> int:
        """Calculate the total embedding dimension."""
        total_dim = 0
        
        if self.use_ast_features:
            # AST features dimension (calculated from ASTGraphProcessor)
            # Basic counts: 7
            # Complexity metrics: 5
            # Node type distribution: 9
            # Edge type distribution: 6
            total_dim += 27
        
        if self.use_bert_features:
            total_dim += self.config.embedding_dim
        
        return total_dim
    
    def _load_bert_model(self):
        """Load BERT-based models."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
        
        # Add special tokens if needed
        if self.config.special_tokens:
            self.tokenizer.add_special_tokens({'additional_special_tokens': self.config.special_tokens})
        
        self.model = AutoModel.from_pretrained(
            self.config.model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
        
        self.model.to(self.device)
        self.model.eval()
    

    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a text string using AST and BERT features.
        
        Args:
            text: Text to embed
            
        Returns:
            Combined embedding vector as numpy array
        """
        embedding_parts = []
        
        # Get AST features
        if self.use_ast_features:
            try:
                ast_features = self.ast_processor.process_code(text)
                embedding_parts.append(ast_features)
            except Exception as e:
                logger.warning(f"Failed to extract AST features: {e}")
                # Use zero vector as fallback
                ast_dim = 27  # Fixed dimension for AST features
                embedding_parts.append(np.zeros(ast_dim, dtype=np.float32))
        
        # Get BERT features
        if self.use_bert_features:
            try:
                bert_features = self._embed_text_bert(text)
                # Ensure BERT features are 1D
                if bert_features.ndim > 1:
                    bert_features = bert_features.flatten()
                embedding_parts.append(bert_features)
            except Exception as e:
                logger.warning(f"Failed to extract BERT features: {e}")
                # Use zero vector as fallback
                embedding_parts.append(np.zeros(self.config.embedding_dim, dtype=np.float32))
        
        # Combine features
        if embedding_parts:
            return np.concatenate(embedding_parts, axis=0)
        else:
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def _embed_text_bert(self, text: str) -> np.ndarray:
        """Embed text using BERT model."""
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=self.config.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Use mean pooling for BERT models
            embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            embeddings = (embeddings * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        
        return embeddings.cpu().numpy()
    
    def embed_file(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Embed a code file.
        
        Args:
            file_path: Path to the code file
            
        Returns:
            Embedding vector as numpy array
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read and preprocess the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        clean = clean_code(content)
        
        return self.embed_text(clean)
    
    def embed_files(self, file_paths: List[Union[str, Path]]) -> List[np.ndarray]:
        """
        Embed multiple code files.
        
        Args:
            file_paths: List of file paths to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for file_path in file_paths:
            try:
                embedding = self.embed_file(file_path)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to embed {file_path}: {str(e)}")
                # Add zero embedding as fallback
                embeddings.append(np.zeros(self.config.embedding_dim))
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension of the current model."""
        return self.embedding_dim
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "name": self.config.name,
            "model_id": self.config.model_id,
            "type": self.config.type.value,
            "languages": self.config.languages,
            "embedding_dim": self.embedding_dim,
            "max_length": self.config.max_length,
            "description": self.config.description,
            "size_mb": self.config.size_mb,
            "requires_gpu": self.config.requires_gpu,
            "device": self.device,
            "use_ast_features": self.use_ast_features,
            "use_bert_features": self.use_bert_features,
            "ast_dim": 27 if self.use_ast_features else 0,
            "bert_dim": self.config.embedding_dim if self.use_bert_features else 0
        }
    
    def get_ast_features_only(self, text: str) -> np.ndarray:
        """
        Get only AST features from text.
        
        Args:
            text: Text to process
            
        Returns:
            AST feature vector
        """
        return self.ast_processor.process_code(text)
    
    def get_bert_features_only(self, text: str) -> np.ndarray:
        """
        Get only BERT features from text.
        
        Args:
            text: Text to process
            
        Returns:
            BERT feature vector
        """
        if not self.use_bert_features:
            raise ValueError("BERT features are disabled")
        return self._embed_text_bert(text) 