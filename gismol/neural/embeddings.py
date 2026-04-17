"""Embedding models for text, code, and COHObjects"""

import numpy as np
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from .components import NeuralComponent
from gismol.core.coh_object import COHObject
from gismol.core.repository import COHRepository
from gismol.core.daemons import IdentityConstraintDaemon, TriggerConstraintDaemon, GoalConstraintDaemon
from gismol.core.exceptions import MissingEmbeddingModel

class EmbeddingModel(NeuralComponent, ABC):
    """Base class for embedding models"""
    
    def __init__(self, name: str, embedding_dim: int = 384, **kwargs):
        super().__init__(name, input_dim=None, output_dim=embedding_dim, **kwargs)
        self.embedding_dim = embedding_dim
    
    @abstractmethod
    def embed(self, input_data: Any) -> np.ndarray:
        """Generate embedding for any input"""
        pass
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed text (default implementation)"""
        return self.embed(text)
    
    def embed_object(self, obj: 'COHObject') -> np.ndarray:
        """Embed a COHObject"""
        # Default: combine name and attributes hash
        text = f"{obj.name} {str(obj.attributes)}"
        return self.embed_text(text)


class TextEmbedder(EmbeddingModel):
    """Text embedding model using simple random projection (for demo)"""
    
    def __init__(self, name: str = "text_embedder", **kwargs):
        super().__init__(name, **kwargs)
        # In real implementation, would load SentenceTransformer
        self._vocab_size = 10000
        self._projection = np.random.randn(self.embedding_dim, self._vocab_size) * 0.01
    
    def _text_to_vector(self, text: str) -> np.ndarray:
        # Very simple: hash words to indices
        vec = np.zeros(self._vocab_size)
        for word in text.lower().split():
            idx = hash(word) % self._vocab_size
            vec[idx] += 1
        return vec
    
    def embed(self, text: str) -> np.ndarray:
        vec = self._text_to_vector(text)
        return self._projection @ vec
    
    def embed_with_constraints(self, text: str, constraints: Dict) -> np.ndarray:
        emb = self.embed(text)
        # Apply constraint transformations (e.g., ensure min similarity)
        return emb


class COHObjectEmbedder(EmbeddingModel):
    """Embedder for COHObjects"""
    
    def embed(self, obj: 'COHObject') -> np.ndarray:
        # Combine object's own embedding with its children's embeddings
        self_emb = self._embed_object_attributes(obj)
        if obj.children:
            child_embs = [self.embed(child) for child in obj.children]
            child_agg = np.mean(child_embs, axis=0)
            return 0.6 * self_emb + 0.4 * child_agg
        return self_emb
    
    def _embed_object_attributes(self, obj: 'COHObject') -> np.ndarray:
        # Create feature vector from attributes and constraints
        features = []
        features.append(hash(obj.name) % 1000 / 1000.0)
        for k, v in obj.attributes.items():
            if isinstance(v, (int, float)):
                features.append(float(v) % 1.0)
        # Pad or truncate to fixed dimension
        while len(features) < self.embedding_dim:
            features.append(0.0)
        return np.array(features[:self.embedding_dim])


class MultiModalEmbedder(EmbeddingModel):
    """Multi-modal embedding combining text and object features"""
    
    def __init__(self, name: str = "multi_modal", text_dim: int = 384, object_dim: int = 384, **kwargs):
        super().__init__(name, embedding_dim=text_dim + object_dim, **kwargs)
        self.text_embedder = TextEmbedder(name=f"{name}_text", embedding_dim=text_dim)
        self.object_embedder = COHObjectEmbedder(name=f"{name}_object", embedding_dim=object_dim)
    
    def embed(self, obj: 'COHObject') -> np.ndarray:
        text_emb = self.text_embedder.embed(obj.name + " " + str(obj.attributes))
        obj_emb = self.object_embedder.embed(obj)
        return np.concatenate([text_emb, obj_emb])


class CodeEmbedder(EmbeddingModel):
    """Embedding for source code"""
    
    def embed(self, code: str) -> np.ndarray:
        # Simple: hash lines
        lines = code.split('\n')
        vec = np.zeros(self.embedding_dim)
        for i, line in enumerate(lines[:self.embedding_dim]):
            vec[i] = hash(line) % 1000 / 1000.0
        return vec