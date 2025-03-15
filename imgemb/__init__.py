"""Image embeddings and semantic search library."""

from .embedder import ImageEmbedder
from .semantic_search import SemanticSearcher
from .visualization import plot_similar_images

__version__ = "0.2.3"

__all__ = ["ImageEmbedder", "SemanticSearcher", "plot_similar_images"]
