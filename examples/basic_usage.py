"""
Basic usage example for the image_embeddings library.
This script demonstrates how to use different embedding methods
and compare their outputs.
"""

import numpy as np
from image_embeddings import ImageEmbedder
import matplotlib.pyplot as plt


def plot_embedding(embedding, title):
    """Helper function to visualize embeddings."""
    plt.figure(figsize=(10, 4))
    plt.plot(embedding)
    plt.title(title)
    plt.xlabel("Dimension")
    plt.ylabel("Value")
    plt.grid(True)


def main():
    # Example image path - replace with your image
    image_path = "examples/sample_image.jpg"

    # Create embedders with different methods
    embedders = {
        "Average Color": ImageEmbedder(method="average_color"),
        "Grid (4x4)": ImageEmbedder(method="grid", grid_size=(4, 4)),
        "Edge": ImageEmbedder(method="edge"),
    }

    # Generate and visualize embeddings
    plt.figure(figsize=(15, 10))
    for i, (name, embedder) in enumerate(embedders.items(), 1):
        # Generate embedding
        embedding = embedder.embed_image(image_path)

        # Print embedding information
        print(f"\n{name} Embedding:")
        print(f"Shape: {embedding.shape}")
        print(f"Range: [{embedding.min():.3f}, {embedding.max():.3f}]")

        # Plot embedding
        plt.subplot(len(embedders), 1, i)
        plt.plot(embedding)
        plt.title(f"{name} Embedding")
        plt.xlabel("Dimension")
        plt.ylabel("Value")
        plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
