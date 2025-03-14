"""
Example script demonstrating image clustering using embeddings.
This script shows how to:
1. Generate embeddings for a collection of images
2. Cluster similar images using K-means
3. Visualize the clustering results
"""

import glob
import numpy as np
from image_embeddings import ImageEmbedder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import cv2
from typing import List, Dict
import os


def generate_embeddings(image_paths: List[str], embedder: ImageEmbedder) -> np.ndarray:
    """Generate embeddings for a list of images."""
    embeddings = []
    for path in image_paths:
        embedding = embedder.embed_image(path)
        embeddings.append(embedding)
    return np.array(embeddings)


def cluster_images(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """Cluster image embeddings using K-means."""
    # Standardize features
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(scaled_embeddings)


def plot_clusters(
    image_paths: List[str], clusters: np.ndarray, max_images_per_cluster: int = 5
):
    """Plot representative images from each cluster."""
    n_clusters = len(np.unique(clusters))

    # Group images by cluster
    cluster_images: Dict[int, List[str]] = {i: [] for i in range(n_clusters)}
    for path, cluster in zip(image_paths, clusters):
        cluster_images[cluster].append(path)

    # Plot settings
    n_cols = max_images_per_cluster
    n_rows = n_clusters
    plt.figure(figsize=(15, 3 * n_rows))

    for cluster in range(n_clusters):
        paths = cluster_images[cluster][:max_images_per_cluster]

        for i, path in enumerate(paths):
            plt.subplot(n_rows, n_cols, cluster * n_cols + i + 1)
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.title(f"Cluster {cluster}")
            plt.axis("off")

    plt.tight_layout()
    plt.show()


def analyze_clusters(image_paths: List[str], clusters: np.ndarray):
    """Print analysis of the clusters."""
    n_clusters = len(np.unique(clusters))

    print("\nCluster Analysis:")
    print("-----------------")

    for cluster in range(n_clusters):
        cluster_size = np.sum(clusters == cluster)
        print(f"\nCluster {cluster}:")
        print(f"Size: {cluster_size} images")
        print("Sample images:")

        # Print some example image names from this cluster
        sample_indices = np.where(clusters == cluster)[0][:3]
        for idx in sample_indices:
            print(f"- {os.path.basename(image_paths[idx])}")


def main():
    # Initialize embedder
    embedder = ImageEmbedder(method="grid", grid_size=(8, 8))

    # Directory containing images
    image_dir = "examples/images/*.jpg"
    image_paths = glob.glob(image_dir)

    if not image_paths:
        print("No images found in the specified directory!")
        return

    print("Generating embeddings...")
    embeddings = generate_embeddings(image_paths, embedder)

    # Determine number of clusters (you might want to adjust this)
    n_clusters = min(5, len(image_paths))

    print("Clustering images...")
    clusters = cluster_images(embeddings, n_clusters)

    # Analyze results
    analyze_clusters(image_paths, clusters)

    # Visualize results
    print("\nPlotting cluster representatives...")
    plot_clusters(image_paths, clusters)


if __name__ == "__main__":
    main()
