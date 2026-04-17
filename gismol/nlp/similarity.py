"""Semantic similarity calculation"""

import numpy as np
from typing import List, Optional
from gismol.core import COHObject
from .embedders import TextEmbedder


class SimilarityCalculator:
    """Calculate semantic similarity between texts and objects"""
    
    def __init__(self, threshold: float = 0.7, embedder: Optional[TextEmbedder] = None):
        self.threshold = threshold
        self.embedder = embedder or TextEmbedder()
    
    def text_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.embedder.embed(text1)
        emb2 = self.embedder.embed(text2)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))
    
    def object_similarity(self, obj1: COHObject, obj2: COHObject, embedder=None) -> float:
        if embedder and hasattr(embedder, 'embed_object'):
            emb1 = embedder.embed_object(obj1)
            emb2 = embedder.embed_object(obj2)
        else:
            # Fallback: use text similarity on names
            emb1 = self.embedder.embed(obj1.name)
            emb2 = self.embedder.embed(obj2.name)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))
    
    def is_similar(self, similarity: float) -> bool:
        return similarity >= self.threshold