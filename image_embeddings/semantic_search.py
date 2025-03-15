"""
Semantic search functionality using CLIP model.
"""

import os
from typing import List, Tuple, Optional
import torch
import clip
from PIL import Image
import numpy as np
from pathlib import Path


class SemanticSearcher:
    """Class for semantic image search using CLIP."""

    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        """Initialize the semantic searcher.

        Args:
            model_name (str): CLIP model variant to use
            device (Optional[str]): Device to run model on ('cuda' or 'cpu')
        """
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load CLIP model
        print(f"Loading CLIP model {model_name} on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)

        # Cache for image embeddings
        self._image_embeddings = {}
        self._image_paths = []

    def _get_image_embedding(self, image_path: str) -> torch.Tensor:
        """Get CLIP embedding for an image.

        Args:
            image_path (str): Path to image file

        Returns:
            torch.Tensor: Image embedding

        Raises:
            FileNotFoundError: If the image file does not exist
        """
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        # Generate embedding
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)

        return image_features

    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """Get CLIP embedding for text.

        Args:
            text (str): Input text

        Returns:
            torch.Tensor: Text embedding
        """
        # Tokenize and encode text
        text_tokens = clip.tokenize([text]).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)

        return text_features

    def index_directory(
        self, directory: str, extensions: List[str] = [".jpg", ".jpeg", ".png"]
    ) -> None:
        """Index all images in a directory.

        Args:
            directory (str): Directory containing images
            extensions (List[str]): Image file extensions to include

        Raises:
            FileNotFoundError: If the directory does not exist
        """
        # Check if directory exists
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Clear existing cache
        self._image_embeddings = {}
        self._image_paths = []

        # Get all image files
        for ext in extensions:
            self._image_paths.extend(Path(directory).glob(f"**/*{ext}"))

        print(f"Indexing {len(self._image_paths)} images...")

        # Generate embeddings
        for path in self._image_paths:
            str_path = str(path)
            try:
                embedding = self._get_image_embedding(str_path)
                self._image_embeddings[str_path] = embedding
            except Exception as e:
                print(f"Error processing {str_path}: {e}")

        print("Indexing complete!")

    def search(
        self, query: str, top_k: int = 5, threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        """Search for images matching a text query.

        Args:
            query (str): Text query
            top_k (int): Number of results to return
            threshold (float): Minimum similarity score (0 to 1)

        Returns:
            List[Tuple[str, float]]: List of (image_path, similarity_score) pairs

        Raises:
            ValueError: If no images have been indexed
        """
        if not self._image_embeddings:
            raise ValueError("No images indexed. Call index_directory() first.")

        # Get query embedding
        text_embedding = self._get_text_embedding(query)

        # Calculate similarities
        similarities = []
        for path, img_embedding in self._image_embeddings.items():
            # Normalize embeddings
            img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

            # Calculate similarity
            similarity = (img_embedding @ text_embedding.T).item()

            if similarity >= threshold:
                similarities.append((path, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]


def main():
    """Example usage."""
    # Initialize searcher
    searcher = SemanticSearcher()

    # Index a directory of images
    searcher.index_directory("path/to/images")

    # Search for images
    results = searcher.search("a photo of a dog", top_k=5)

    # Print results
    print("\nSearch Results:")
    for path, score in results:
        print(f"{path}: {score:.3f}")


if __name__ == "__main__":
    main()
