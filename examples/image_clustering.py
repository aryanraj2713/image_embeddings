"""
Example script demonstrating image clustering using embeddings.
This script shows how to:
1. Generate embeddings for a collection of images
2. Cluster similar images using various clustering methods
3. Analyze and visualize clustering results
4. Handle different clustering parameters
"""

import glob
import numpy as np
from imgemb import ImageEmbedder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import cv2
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path


def generate_embeddings(
    image_paths: List[str], embedder: ImageEmbedder, verbose: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """Generate embeddings for a list of images.

    Args:
        image_paths (List[str]): List of image file paths
        embedder (ImageEmbedder): Initialized embedder instance
        verbose (bool): Whether to print progress

    Returns:
        Tuple[np.ndarray, List[str]]: Array of embeddings and list of valid image paths
    """
    embeddings = []
    valid_paths = []

    for i, path in enumerate(image_paths):
        try:
            if verbose and (i + 1) % 10 == 0:
                print(f"Processing image {i + 1}/{len(image_paths)}")

            embedding = embedder.embed_image(path)
            embeddings.append(embedding)
            valid_paths.append(path)

        except Exception as e:
            print(f"Error processing {path}: {e}")

    return np.array(embeddings), valid_paths


def cluster_images(
    embeddings: np.ndarray,
    method: str = "kmeans",
    n_clusters: Optional[int] = 5,
    **kwargs,
) -> np.ndarray:
    """Cluster image embeddings using various methods.

    Args:
        embeddings (np.ndarray): Array of image embeddings
        method (str): Clustering method ('kmeans', 'dbscan', or 'hierarchical')
        n_clusters (Optional[int]): Number of clusters (for kmeans and hierarchical)
        **kwargs: Additional parameters for clustering algorithms

    Returns:
        np.ndarray: Cluster assignments for each image
    """
    # Standardize features
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    # Choose clustering method
    if method == "dbscan":
        clusterer = DBSCAN(
            eps=kwargs.get("eps", 0.5), min_samples=kwargs.get("min_samples", 5)
        )
    elif method == "hierarchical":
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters, linkage=kwargs.get("linkage", "ward")
        )
    else:  # kmeans
        clusterer = KMeans(
            n_clusters=n_clusters, random_state=kwargs.get("random_state", 42)
        )

    # Perform clustering
    clusters = clusterer.fit_predict(scaled_embeddings)

    # Calculate clustering quality if possible
    if method != "dbscan" and len(np.unique(clusters)) > 1:
        score = silhouette_score(scaled_embeddings, clusters)
        print(f"\nSilhouette score: {score:.3f}")

    return clusters


def plot_clusters(
    image_paths: List[str],
    clusters: np.ndarray,
    max_images_per_cluster: int = 5,
    figsize: Tuple[int, int] = (15, None),
) -> None:
    """Plot representative images from each cluster.

    Args:
        image_paths (List[str]): List of image file paths
        clusters (np.ndarray): Cluster assignments
        max_images_per_cluster (int): Maximum number of images to show per cluster
        figsize (Tuple[int, int]): Figure size (width, height)
    """
    n_clusters = len(np.unique(clusters))

    # Group images by cluster
    cluster_images: Dict[int, List[str]] = {i: [] for i in range(n_clusters)}
    for path, cluster in zip(image_paths, clusters):
        cluster_images[cluster].append(path)

    # Calculate figure height based on number of clusters
    height = figsize[0] * (n_clusters / 5)  # Adjust aspect ratio
    plt.figure(figsize=(figsize[0], height))

    # Plot settings
    n_cols = max_images_per_cluster
    n_rows = n_clusters

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


def analyze_clusters(
    image_paths: List[str], clusters: np.ndarray, show_paths: bool = False
) -> None:
    """Print detailed analysis of the clusters.

    Args:
        image_paths (List[str]): List of image file paths
        clusters (np.ndarray): Cluster assignments
        show_paths (bool): Whether to show full paths or just filenames
    """
    n_clusters = len(np.unique(clusters))

    print("\nCluster Analysis:")
    print("-" * 50)

    for cluster in range(n_clusters):
        cluster_mask = clusters == cluster
        cluster_size = np.sum(cluster_mask)
        cluster_paths = np.array(image_paths)[cluster_mask]

        print(f"\nCluster {cluster}:")
        print(f"Size: {cluster_size} images ({cluster_size/len(clusters)*100:.1f}%)")
        print("Sample images:")

        # Print some example image names from this cluster
        for path in cluster_paths[:3]:
            if show_paths:
                print(f"- {path}")
            else:
                print(f"- {os.path.basename(path)}")


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

    # Generate embeddings
    print("Generating embeddings...")
    embeddings, valid_paths = generate_embeddings(image_paths, embedder)

    # Try different clustering methods
    methods = {
        "kmeans": {"n_clusters": 5},
        "dbscan": {"eps": 0.5, "min_samples": 3},
        "hierarchical": {"n_clusters": 5, "linkage": "ward"},
    }

    for method_name, params in methods.items():
        print(f"\nClustering with {method_name.upper()}:")
        clusters = cluster_images(embeddings, method=method_name, **params)

        # Analyze results
        analyze_clusters(valid_paths, clusters)

        # Visualize results
        print(f"\nPlotting cluster representatives for {method_name}...")
        plot_clusters(valid_paths, clusters)


if __name__ == "__main__":
    main()
