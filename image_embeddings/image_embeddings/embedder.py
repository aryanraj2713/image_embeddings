import cv2
import numpy as np
from typing import Tuple, Union, Optional
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Abstract base class for image embedders."""
    
    @abstractmethod
    def embed(self, image: np.ndarray) -> np.ndarray:
        """Generate embedding for the given image."""
        pass


class AverageColorEmbedder(BaseEmbedder):
    """Generates embeddings based on average color values."""
    
    def embed(self, image: np.ndarray) -> np.ndarray:
        """
        Generate embedding by computing average color values.
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            np.ndarray: 1D array of average color values
        """
        return np.mean(image, axis=(0, 1))


class GridEmbedder(BaseEmbedder):
    """Generates embeddings based on grid-wise average colors."""
    
    def __init__(self, grid_size: Tuple[int, int] = (4, 4)):
        self.grid_size = grid_size
    
    def embed(self, image: np.ndarray) -> np.ndarray:
        """
        Generate embedding by dividing image into grid and computing average colors.
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            np.ndarray: Flattened array of grid-wise average colors
        """
        h, w = image.shape[:2]
        grid_h, grid_w = self.grid_size
        
        cell_h, cell_w = h // grid_h, w // grid_w
        
        embedding = []
        for i in range(grid_h):
            for j in range(grid_w):
                cell = image[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                cell_avg = np.mean(cell, axis=(0, 1))
                embedding.extend(cell_avg)
                
        return np.array(embedding)


class EdgeEmbedder(BaseEmbedder):
    """Generates embeddings based on edge information."""
    
    def embed(self, image: np.ndarray) -> np.ndarray:
        """
        Generate embedding using edge detection.
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            np.ndarray: Edge-based embedding
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return np.histogram(edges, bins=64, range=(0, 256))[0]


class ImageEmbedder:
    """Main class for generating image embeddings."""
    
    METHODS = {
        'average_color': AverageColorEmbedder,
        'grid': GridEmbedder,
        'edge': EdgeEmbedder
    }
    
    def __init__(
        self,
        method: str = 'average_color',
        grid_size: Tuple[int, int] = (4, 4),
        normalize: bool = True
    ):
        """
        Initialize the ImageEmbedder.
        
        Args:
            method (str): Embedding method ('average_color', 'grid', or 'edge')
            grid_size (tuple): Grid size for grid-based embedding
            normalize (bool): Whether to normalize the embeddings
        """
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Available methods: {list(self.METHODS.keys())}")
        
        self.method = method
        self.normalize = normalize
        
        if method == 'grid':
            self.embedder = self.METHODS[method](grid_size=grid_size)
        else:
            self.embedder = self.METHODS[method]()
    
    def embed_image(self, image_path: str) -> np.ndarray:
        """
        Generate embedding for an image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.ndarray: Image embedding
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        return self.embed(image)
    
    def embed(self, image: np.ndarray) -> np.ndarray:
        """
        Generate embedding for an image array.
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            np.ndarray: Image embedding
        """
        embedding = self.embedder.embed(image)
        
        if self.normalize:
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
        return embedding
