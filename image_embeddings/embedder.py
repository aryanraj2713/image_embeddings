import cv2
import numpy as np
from typing import Union, List, Tuple
import os


class ImageEmbedder:
    """A class for generating and comparing image embeddings."""

    VALID_METHODS = ["average_color", "grid", "edge"]

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        method: str = "grid",
        grid_size: Tuple[int, int] = (4, 4),
        normalize: bool = True,
    ):
        """Initialize the ImageEmbedder.

        Args:
            target_size: Tuple of (height, width) to resize images to before embedding
            method: Embedding method ('average_color', 'grid', or 'edge')
            grid_size: Grid size for grid-based embedding
            normalize: Whether to normalize embeddings
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
            # Split image into grid cells and compute average color for each cell
            h, w = img.shape[:2]
            gh, gw = self.grid_size
            cell_h, cell_w = h // gh, w // gw

            embedding = []
            for i in range(gh):
                for j in range(gw):
                    cell = img[
                        i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w
                    ]
                    cell_avg = np.mean(cell, axis=(0, 1))
                    embedding.extend(cell_avg)
            embedding = np.array(embedding)
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

    def compare_images(self, image_path1: str, image_path2: str) -> float:
        """Compare two images by computing the cosine similarity of their embeddings.

        Args:
            image_path1: Path to first image
            image_path2: Path to second image

        Returns:
            Cosine similarity score between 0 and 1
        """
        # Generate embeddings
        embedding1 = self.embed_image(image_path1)
        embedding2 = self.embed_image(image_path2)

        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2)
        if self.normalize:
            similarity = (similarity + 1) / 2  # Scale to [0, 1]

        return similarity

    def find_similar_images(
        self, query_image: str, image_dir: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find the most similar images to a query image in a directory.

        Args:
            query_image: Path to the query image
            image_dir: Directory containing images to search through
            top_k: Number of similar images to return

        Returns:
            List of (image_path, similarity_score) tuples for the top_k most similar images
        """
        # Check if directory exists
        if not os.path.exists(image_dir):
            raise ValueError(f"Directory does not exist: {image_dir}")

        # Generate query embedding
        query_embedding = self.embed_image(query_image)

        # Get all images in directory
        image_files = [
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # Compare with all images
        similarities = []
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            try:
                embedding = self.embed_image(image_path)
                similarity = np.dot(query_embedding, embedding)
                if self.normalize:
                    similarity = (similarity + 1) / 2  # Scale to [0, 1]
                similarities.append((image_path, float(similarity)))
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
