"""
Example script demonstrating image similarity comparison using embeddings.
This script shows how to:
1. Generate embeddings for multiple images
2. Compare images using different similarity metrics
3. Find the most similar images in a dataset
"""

import os
import glob
import numpy as np
from image_embeddings import ImageEmbedder
from typing import List, Tuple
import matplotlib.pyplot as plt
import cv2


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)


def find_most_similar(
    query_embedding: np.ndarray,
    embeddings: List[np.ndarray],
    image_paths: List[str],
    metric: str = "cosine",
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Find the most similar images to a query image."""
    similarities = []

    for emb, path in zip(embeddings, image_paths):
        if metric == "cosine":
            score = cosine_similarity(query_embedding, emb)
            # Higher is better for cosine similarity
            similarities.append((path, score))
        else:
            score = euclidean_distance(query_embedding, emb)
            # Lower is better for euclidean distance
            similarities.append((path, -score))

    # Sort by similarity score (higher is better)
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def plot_similar_images(query_path: str, similar_images: List[Tuple[str, float]]):
    """Plot query image and its most similar matches."""
    n_images = len(similar_images) + 1
    plt.figure(figsize=(15, 3))

    # Plot query image
    plt.subplot(1, n_images, 1)
    query_img = cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2RGB)
    plt.imshow(query_img)
    plt.title("Query Image")
    plt.axis("off")

    # Plot similar images
    for i, (path, score) in enumerate(similar_images, 2):
        plt.subplot(1, n_images, i)
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(f"Score: {score:.2f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    # Initialize embedder (you can experiment with different methods)
    embedder = ImageEmbedder(method="grid", grid_size=(8, 8))

    # Directory containing images
    image_dir = "examples/images/*.jpg"

    # Generate embeddings for all images
    image_paths = glob.glob(image_dir)
    embeddings = []

    print("Generating embeddings...")
    for path in image_paths:
        embedding = embedder.embed_image(path)
        embeddings.append(embedding)

    # Select a query image (first image in this case)
    query_path = image_paths[0]
    query_embedding = embeddings[0]

    # Find similar images using both metrics
    print("\nFinding similar images...")
    cosine_matches = find_most_similar(
        query_embedding, embeddings, image_paths, metric="cosine"
    )
    euclidean_matches = find_most_similar(
        query_embedding, embeddings, image_paths, metric="euclidean"
    )

    # Print results
    print("\nTop matches (Cosine Similarity):")
    for path, score in cosine_matches:
        print(f"{os.path.basename(path)}: {score:.3f}")

    print("\nTop matches (Euclidean Distance):")
    for path, score in euclidean_matches:
        print(
            f"{os.path.basename(path)}: {-score:.3f}"
        )  # Convert back to actual distance

    # Visualize results
    print("\nPlotting results...")
    plot_similar_images(query_path, cosine_matches)


if __name__ == "__main__":
    main()
