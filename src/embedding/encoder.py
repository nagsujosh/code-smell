"""
Code encoder for generating semantic embeddings.

Uses pre-trained transformer models to generate vector
representations of code content.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.core.config import EmbeddingConfig
from src.embedding.extractor import SemanticPayload

logger = logging.getLogger(__name__)


class CodeEncoder:
    """
    Generates semantic embeddings for code using pre-trained models.
    
    Supports various code-specialized transformer models including
    CodeBERT, GraphCodeBERT, and UniXCoder.
    """

    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        self._embedding_dim = None

    def _get_device(self) -> str:
        """Determine the optimal device for inference."""
        if self.config.device != "auto":
            return self.config.device

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self) -> None:
        """Load the transformer model and tokenizer."""
        if self.model is not None:
            return

        model_name = self.config.model_name
        cache_dir = Path(self.config.model_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading model: {model_name} on device: {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                trust_remote_code=True,
            )

            self.model = AutoModel.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                trust_remote_code=True,
            )

            self.model.to(self.device)
            self.model.eval()

            self._embedding_dim = self.model.config.hidden_size

            logger.info(
                f"Model loaded successfully. "
                f"Embedding dimension: {self._embedding_dim}"
            )

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension of the loaded model."""
        if self._embedding_dim is None:
            self.load_model()
        return self._embedding_dim

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a text string to a vector embedding.
        
        Args:
            text: Text to encode.
            
        Returns:
            Embedding vector as numpy array.
        """
        if self.model is None:
            self.load_model()

        inputs = self.tokenizer(
            text,
            max_length=self.config.max_token_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

            embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]

            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
            sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask

        return pooled.cpu().numpy().flatten()

    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Encode multiple texts in batches.
        
        Args:
            texts: List of texts to encode.
            
        Returns:
            List of embedding vectors.
        """
        if self.model is None:
            self.load_model()

        embeddings = []
        batch_size = self.config.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                max_length=self.config.max_token_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

                batch_embeddings = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]

                mask_expanded = attention_mask.unsqueeze(-1).expand(
                    batch_embeddings.size()
                )
                sum_embeddings = torch.sum(
                    batch_embeddings * mask_expanded, dim=1
                )
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_embeddings / sum_mask

            for emb in pooled.cpu().numpy():
                embeddings.append(emb)

        return embeddings

    def encode_payload(self, payload: SemanticPayload) -> np.ndarray:
        """
        Encode a semantic payload to an embedding.
        
        Args:
            payload: Semantic payload to encode.
            
        Returns:
            Embedding vector.
        """
        text = payload.to_text(max_length=self.config.max_token_length * 4)
        return self.encode(text)

    def encode_payloads(
        self, payloads: Dict[str, SemanticPayload]
    ) -> Dict[str, np.ndarray]:
        """
        Encode multiple payloads efficiently.
        
        Args:
            payloads: Dictionary mapping IDs to payloads.
            
        Returns:
            Dictionary mapping IDs to embeddings.
        """
        if not payloads:
            return {}

        ids = list(payloads.keys())
        texts = [
            payloads[pid].to_text(max_length=self.config.max_token_length * 4)
            for pid in ids
        ]

        embeddings = self.encode_batch(texts)

        return dict(zip(ids, embeddings))

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Model unloaded")

