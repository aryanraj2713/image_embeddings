"""
Example script demonstrating image similarity comparison using embeddings.
This script shows how to:
1. Generate embeddings for multiple images
2. Compare images using different similarity metrics
3. Find the most similar images in a dataset
4. Visualize similarity results
"""

import os
import glob
import numpy as np
from imgemb import ImageEmbedder
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import cv2
from pathlib import Path


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a (np.ndarray): First vector
        b (np.ndarray): Second vector

    Returns:
        float: Cosine similarity score (1.0 = most similar)
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors.

    Args:
        a (np.ndarray): First vector
        b (np.ndarray): Second vector

    Returns:
        float: Euclidean distance (0.0 = most similar)
    """
    return np.linalg.norm(a - b)


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Manhattan distance between two vectors.

    Args:
        a (np.ndarray): First vector
        b (np.ndarray): Second vector

    Returns:
        float: Manhattan distance (0.0 = most similar)
    """
    return np.sum(np.abs(a - b))


def find_most_similar(
    query_embedding: np.ndarray,
    embeddings: List[np.ndarray],
    image_paths: List[str],
    metric: str = "cosine",
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Find the most similar images to a query image.

    Args:
        query_embedding (np.ndarray): Query image embedding
        embeddings (List[np.ndarray]): List of embeddings to compare against
        image_paths (List[str]): List of corresponding image paths
        metric (str): Similarity metric ('cosine', 'euclidean', or 'manhattan')
        top_k (int): Number of similar images to return

    Returns:
        List[Tuple[str, float]]: List of (image_path, similarity_score) pairs
    """
    similarities = []

    for emb, path in zip(embeddings, image_paths):
        if metric == "cosine":
            score = cosine_similarity(query_embedding, emb)
            # Higher is better for cosine similarity
            similarities.append((path, score))
        elif metric == "manhattan":
            score = manhattan_distance(query_embedding, emb)
            # Lower is better for manhattan distance
            similarities.append((path, -score))
        else:  # euclidean
            score = euclidean_distance(query_embedding, emb)
            # Lower is better for euclidean distance
            similarities.append((path, -score))

    # Sort by similarity score (higher is better)
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def plot_similar_images(
    query_path: str, similar_images: List[Tuple[str, float]], metric: str = "cosine"
) -> None:
    """Plot query image and its most similar matches.

    Args:
        query_path (str): Path to query image
        similar_images (List[Tuple[str, float]]): List of similar image paths and scores
        metric (str): Similarity metric used
    """
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
        if metric == "cosine":
            plt.title(f"Similarity: {score:.2f}")
        else:
            plt.title(f"Distance: {-score:.2f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def analyze_results(
    similar_images: List[Tuple[str, float]], metric: str = "cosine"
) -> None:
    """Print analysis of similarity results.

    Args:
        similar_images (List[Tuple[str, float]]): List of similar image paths and scores
        metric (str): Similarity metric used
    """
    print("\nSimilarity Analysis:")
    print("-" * 50)

    for i, (path, score) in enumerate(similar_images, 1):
        filename = os.path.basename(path)
        if metric == "cosine":
            print(f"{i}. {filename}: Similarity = {score:.3f}")
        else:
            print(f"{i}. {filename}: Distance = {-score:.3f}")


def main():
    # Initialize embedder with grid method for better spatial awareness
    embedder = ImageEmbedder(
        method="grid", grid_size=(8, 8), normalize=True, color_space="rgb"
    )

    # Directory containing images
    image_dir = "examples/images/*.jpg"
    image_paths = glob.glob(image_dir)

    if not image_paths:
        print("No images found in the specified directory!")
        return

    print("Generating embeddings...")
    embeddings = []
    for path in image_paths:
        try:
            embedding = embedder.embed_image(path)
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            image_paths.remove(path)

    # Select a query image (first image in this case)
    query_path = image_paths[0]
    query_embedding = embeddings[0]

    # Find similar images using different metrics
    print("\nFinding similar images...")
    metrics = ["cosine", "euclidean", "manhattan"]

    for metric in metrics:
        print(f"\nResults using {metric.capitalize()} metric:")
        similar_images = find_most_similar(
            query_embedding, embeddings, image_paths, metric=metric
        )

        # Analyze and visualize results
        analyze_results(similar_images, metric)
        plot_similar_images(query_path, similar_images, metric)


if __name__ == "__main__":
    main()
