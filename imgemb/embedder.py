import cv2
import numpy as np
from typing import Union, List, Tuple
import os
import glob
import sys


class ImageEmbedder:
    """A class for generating and comparing image embeddings."""

    VALID_METHODS = ["average_color", "grid", "edge"]

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        method: str = "grid",
        grid_size: Tuple[int, int] = (4, 4),
        normalize: bool = True,
        color_space: str = "rgb",
    ):
        """Initialize the ImageEmbedder.

        Args:
            target_size: Tuple of (height, width) to resize images to before embedding
            method: Embedding method ('average_color', 'grid', or 'edge')
            grid_size: Grid size for grid-based embedding
            normalize: Whether to normalize embeddings
            color_space: Color space to use ('rgb' or 'hsv')
        """
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Invalid method: {method}. Must be one of {self.VALID_METHODS}"
            )

        if not isinstance(grid_size, tuple) or len(grid_size) != 2:
            raise ValueError("grid_size must be a tuple of (height, width)")

        self.target_size = target_size
        self.method = method
        self.grid_size = grid_size
        self.normalize = normalize
        self.color_space = color_space.lower()
        if self.color_space not in ["rgb", "hsv"]:
            raise ValueError("Color space must be 'rgb' or 'hsv'")

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess an image by loading and resizing it.

        Args:
            image_path: Path to the image file

        Returns:
            Preprocessed image as numpy array
        """
        # Load and resize image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image at {image_path}")

        img = cv2.resize(img, self.target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize pixel values
        img = img.astype(np.float32) / 255.0

        return img

    def embed_image(self, image_path: str) -> np.ndarray:
        """Generate an embedding for an image using the specified method.

        Args:
            image_path: Path to the image file

        Returns:
            Embedding vector as numpy array
        """
        img = self.preprocess_image(image_path)

        if self.method == "average_color":
            embedding = np.mean(img, axis=(0, 1))
        elif self.method == "grid":
            embedding = self._grid_embedder(img)
            # Skip normalization for 1x1 grid as it's already handled in _grid_embedder
            if self.grid_size == (1, 1):
                return embedding
        else:  # edge method
            # Compute edge features using Sobel operator
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            embedding = np.histogram(magnitude, bins=32, range=(0, 1))[0]

        if self.normalize:
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding

    def generate_embedding(self, image_path: str) -> np.ndarray:
        """Alias for embed_image for backward compatibility."""
        return self.embed_image(image_path)

    def compare_images(self, image1_path: str, image2_path: str) -> float:
        """Compare two images and return similarity score.

        Args:
            image1_path (str): Path to first image
            image2_path (str): Path to second image

        Returns:
            float: Similarity score between 0 and 1
        """
        # Get embeddings
        emb1 = self.embed_image(image1_path)
        emb2 = self.embed_image(image2_path)

        # Normalize embeddings
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)

        # Calculate cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)

        # Ensure similarity is between 0 and 1
        return float(np.clip(similarity, 0.0, 1.0))

    def find_similar_images(
        self, query_image: str, image_dir: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find similar images to a query image in a directory.

        Args:
            query_image: Path to query image
            image_dir: Directory containing images to search
            top_k: Number of similar images to return

        Returns:
            List of tuples containing (image_path, similarity_score)

        Raises:
            FileNotFoundError: If query image or directory does not exist
            ValueError: If no valid images found in directory or top_k is invalid
        """
        # Validate inputs
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        # Validate paths
        if not os.path.exists(query_image):
            raise FileNotFoundError(f"Query image not found: {query_image}")
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Directory not found: {image_dir}")

        # Get query embedding
        query_emb = self.embed_image(query_image)

        # Find all images in directory
        image_paths = []
        for ext in [".jpg", ".jpeg", ".png"]:
            image_paths.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))

        if not image_paths:
            return []  # Return empty list for empty directory

        # Get embeddings for all images
        embeddings = []
        for path in image_paths:
            try:
                emb = self.embed_image(path)
                embeddings.append(emb)
            except Exception as e:
                print(f"Warning: Could not process {path}: {e}", file=sys.stderr)

        if not embeddings:
            return []  # Return empty list if no images could be processed

        # Convert to numpy arrays
        embeddings = np.array(embeddings)

        # Compute similarities
        similarities = []
        for i, emb in enumerate(embeddings):
            sim = np.dot(query_emb, emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(emb)
            )
            similarities.append((image_paths[i], float(sim)))

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _grid_embedder(self, img: np.ndarray) -> np.ndarray:
        """Generate embedding by dividing image into grid cells and computing mean color.

        Args:
            img: Input image as numpy array

        Returns:
            Grid embedding as numpy array
        """
        # Convert to HSV if specified
        if self.color_space == "hsv":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Get grid dimensions
        h, w = img.shape[:2]
        gh, gw = self.grid_size
        cell_h, cell_w = h // gh, w // gw

        # Special case for 1x1 grid
        if gh == 1 and gw == 1:
            mean_color = np.mean(img, axis=(0, 1))
            # For single cell, we want to preserve the original pixel values
            # No need to normalize to unit length
            return mean_color.astype(np.float32)

        # Initialize feature array
        features = np.zeros((gh * gw * 3,), dtype=np.float32)

        # Compute mean color for each cell
        for i in range(gh):
            for j in range(gw):
                # Extract cell
                cell = img[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]
                # Compute mean color
                mean_color = np.mean(cell, axis=(0, 1))
                # Store in feature array
                features[i * gw * 3 + j * 3 : i * gw * 3 + j * 3 + 3] = mean_color

        # Normalize if requested
        if self.normalize:
            features = features / np.linalg.norm(features)

        return features
