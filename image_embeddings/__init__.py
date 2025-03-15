"""Image Embeddings - A lightweight Python library for generating image embeddings"""

__version__ = "0.2.1"

from .embedder import ImageEmbedder
from .semantic_search import SemanticSearcher

__all__ = ["ImageEmbedder", "SemanticSearcher"]
