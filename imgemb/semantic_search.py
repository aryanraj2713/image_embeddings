"""
Semantic search functionality using OpenCLIP model.
"""

import os
from typing import List, Tuple, Optional
import torch
import open_clip
from PIL import Image
import numpy as np
from pathlib import Path
import sys


class SemanticSearcher:
    """Class for semantic image search using OpenCLIP model."""

    def __init__(self, device: str = "cuda", model_name: str = "ViT-B-32"):
        """Initialize the searcher.

        Args:
            device: Device to run the model on ('cuda' or 'cpu').
            model_name: OpenCLIP model variant to use.
        """
        # Set device based on availability
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, using CPU instead.")
            device = "cpu"
        self.device = device

        print(f"Loading OpenCLIP model {model_name} on {self.device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="laion2b_s34b_b79k", device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self._image_embeddings = None
        self._image_paths = []

    def _get_image_embedding(self, image_path: str) -> torch.Tensor:
        """Get embedding for an image.

        Args:
            image_path: Path to the image file.

        Returns:
            Tensor containing the image embedding.

        Raises:
            FileNotFoundError: If the image file does not exist.
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """Get embedding for a text query.

        Args:
            text: Text query.

        Returns:
            Tensor containing the text embedding.
        """
        text = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def index_directory(self, directory: str, extensions: List[str] = None) -> None:
        """Index all images in a directory.

        Args:
            directory: Path to directory containing images.
            extensions: List of image file extensions to include (e.g., ['.jpg', '.png']).

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        directory_path = Path(directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Set default extensions if none provided
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png"]

        # Get list of image files
        image_files = []
        for ext in extensions:
            ext = ext if ext.startswith(".") else f".{ext}"
            image_files.extend(directory_path.glob(f"**/*{ext}"))

        if not image_files:
            print("No images found in directory.")
            return

        print(f"Indexing {len(image_files)} images...")
        embeddings = []
        self._image_paths = []

        # Process each image
        for img_path in image_files:
            try:
                embedding = self._get_image_embedding(str(img_path))
                embeddings.append(embedding)
                self._image_paths.append(str(img_path))
            except Exception as e:
                print(f"Error processing {img_path}: {e}", file=sys.stderr)

        if embeddings:
            self._image_embeddings = torch.cat(embeddings)
        print("Indexing complete!")

    def search(
        self, query: str, top_k: int = 5, threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        """Search for images matching the text query.

        Args:
            query: Text query to search for.
            top_k: Number of top results to return.
            threshold: Minimum similarity score threshold (0.0 to 1.0).

        Returns:
            List of tuples containing (image_path, similarity_score).

        Raises:
            ValueError: If top_k is less than 1 or no images have been indexed.
        """
        if not isinstance(self._image_embeddings, torch.Tensor):
            raise ValueError("No images indexed. Call index_directory() first.")

        if top_k < 1:
            raise ValueError("top_k must be greater than 0")

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")

        # Get text embedding
        text_embedding = self._get_text_embedding(query)

        # Calculate cosine similarity with all image embeddings
        similarities = torch.nn.functional.cosine_similarity(
            text_embedding, self._image_embeddings
        )

        # Filter by threshold and get top k indices
        mask = similarities >= threshold
        filtered_similarities = similarities[mask]
        filtered_indices = torch.arange(len(similarities))[mask]

        if len(filtered_similarities) == 0:
            return []

        # Get top k results
        k = min(top_k, len(filtered_similarities))
        top_k_values, top_k_indices = torch.topk(filtered_similarities, k)

        # Map back to original indices
        original_indices = filtered_indices[top_k_indices]

        # Return results
        results = []
        for idx, score in zip(original_indices, top_k_values):
            results.append((self._image_paths[idx], float(score)))
        return results


def main():
    """Main function for testing."""
    try:
        searcher = SemanticSearcher()
        searcher.index_directory("test_images")
        results = searcher.search("a photo of a dog", top_k=5)
        for path, score in results:
            print(f"Score: {score:.3f} - {path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
