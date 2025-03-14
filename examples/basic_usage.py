"""
Basic usage example for the imgemb library.
This script demonstrates the core functionality of the library,
including different embedding methods and visualization.
"""

import numpy as np
from imgemb import ImageEmbedder
import matplotlib.pyplot as plt
import os


def plot_embedding(embedding: np.ndarray, title: str) -> None:
    """Helper function to visualize embeddings.

    Args:
        embedding (np.ndarray): The embedding vector to visualize
        title (str): Title for the plot
    """
    plt.figure(figsize=(10, 4))
    plt.plot(embedding)
    plt.title(title)
    plt.xlabel("Dimension")
    plt.ylabel("Value")
    plt.grid(True)


def compare_images(embedder: ImageEmbedder, image1_path: str, image2_path: str) -> None:
    """Compare two images using different similarity metrics.

    Args:
        embedder (ImageEmbedder): Initialized embedder instance
        image1_path (str): Path to first image
        image2_path (str): Path to second image
    """
    # Generate embeddings
    emb1 = embedder.embed_image(image1_path)
    emb2 = embedder.embed_image(image2_path)

    # Compare using different metrics
    cosine_sim = embedder.compare_images(image1_path, image2_path, metric="cosine")
    euclidean_dist = embedder.compare_images(
        image1_path, image2_path, metric="euclidean"
    )

    print(
        f"\nComparing {os.path.basename(image1_path)} and {os.path.basename(image2_path)}:"
    )
    print(f"Cosine similarity: {cosine_sim:.3f}")
    print(f"Euclidean distance: {euclidean_dist:.3f}")


def main():
    # Example image paths - replace with your images
    image_path = "examples/images/sample1.jpg"
    image_path2 = "examples/images/sample2.jpg"

    # Create embedders with different methods
    embedders = {
        "Average Color": ImageEmbedder(method="average_color", normalize=True),
        "Grid (4x4)": ImageEmbedder(
            method="grid", grid_size=(4, 4), normalize=True, color_space="rgb"
        ),
        "Edge": ImageEmbedder(method="edge", normalize=True),
    }

    # Generate and visualize embeddings
    plt.figure(figsize=(15, 10))
    for i, (name, embedder) in enumerate(embedders.items(), 1):
        try:
            # Generate embedding
            embedding = embedder.embed_image(image_path)

            # Print embedding information
            print(f"\n{name} Embedding:")
            print(f"Shape: {embedding.shape}")
            print(f"Range: [{embedding.min():.3f}, {embedding.max():.3f}]")
            print(f"Mean: {embedding.mean():.3f}")
            print(f"Std: {embedding.std():.3f}")

            # Plot embedding
            plt.subplot(len(embedders), 1, i)
            plt.plot(embedding)
            plt.title(f"{name} Embedding")
            plt.xlabel("Dimension")
            plt.ylabel("Value")
            plt.grid(True)

            # Compare with another image
            if os.path.exists(image_path2):
                compare_images(embedder, image_path, image_path2)

        except Exception as e:
            print(f"\nError with {name} embedder: {e}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
