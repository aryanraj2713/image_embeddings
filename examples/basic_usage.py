"""
Basic usage example for the imgemb library.
This script demonstrates the core functionality of the library,
including different embedding methods and visualization.
"""

import numpy as np
from imgemb import ImageEmbedder, plot_similar_images
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import cv2


def plot_embedding(embedding: np.ndarray, title: str) -> None:
    """Helper function to visualize embeddings.

    Args:
        embedding (np.ndarray): The embedding vector to visualize
        title (str): Title for the plot
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=embedding, mode="lines"))
    fig.update_layout(
        title=title, xaxis_title="Dimension", yaxis_title="Value", showlegend=False
    )
    fig.show()


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


def create_test_images():
    """Create test images for demonstration."""
    os.makedirs("test_images", exist_ok=True)

    # Create test images with different shapes and colors
    shapes = {
        "red_square": (
            (255, 0, 0),
            lambda img: cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1),
        ),
        "blue_circle": (
            (0, 0, 255),
            lambda img: cv2.circle(img, (150, 150), 50, (0, 0, 255), -1),
        ),
        "green_triangle": (
            (0, 255, 0),
            lambda img: cv2.fillPoly(
                img,
                [np.array([[150, 100], [100, 200], [200, 200]], np.int32)],
                (0, 255, 0),
            ),
        ),
        "yellow_star": (
            (255, 255, 0),
            lambda img: create_star(img, (150, 150), 50, (255, 255, 0)),
        ),
    }

    image_paths = []
    for name, (color, draw_func) in shapes.items():
        img_path = f"test_images/{name}.jpg"
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        draw_func(img)
        cv2.imwrite(img_path, img)
        image_paths.append(img_path)

    return image_paths


def create_star(img, center, size, color):
    """Helper function to create a star shape."""
    points = []
    for i in range(5):
        # Outer points
        angle = i * 2 * np.pi / 5 - np.pi / 2
        x = int(center[0] + size * np.cos(angle))
        y = int(center[1] + size * np.sin(angle))
        points.append([x, y])

        # Inner points
        angle += np.pi / 5
        x = int(center[0] + size * 0.5 * np.cos(angle))
        y = int(center[1] + size * 0.5 * np.sin(angle))
        points.append([x, y])

    pts = np.array(points, np.int32)
    cv2.fillPoly(img, [pts], color)


def main():
    """Demonstrate basic usage of the library."""
    # Create test images
    image_paths = create_test_images()
    query_image = image_paths[0]  # red square

    print("\nDemonstrating different embedding methods:")

    # Test average color method
    print("\n1. Average Color Method")
    embedder = ImageEmbedder(method="average_color")
    embedding = embedder.embed_image(query_image)
    print(f"- Embedding shape: {embedding.shape}")

    # Find similar images
    similar_images = embedder.find_similar_images(query_image, "test_images", top_k=3)
    print("\nSimilar images (average color):")
    for path, score in similar_images:
        print(f"- {os.path.basename(path)}: {score:.3f}")

    # Visualize results
    fig = plot_similar_images(
        query_image, similar_images, title="Similar Images (Average Color)"
    )
    fig.show()

    # Test grid method
    print("\n2. Grid Method")
    embedder = ImageEmbedder(method="grid", grid_size=(4, 4))
    embedding = embedder.embed_image(query_image)
    print(f"- Embedding shape: {embedding.shape}")

    # Find similar images
    similar_images = embedder.find_similar_images(query_image, "test_images", top_k=3)
    print("\nSimilar images (grid):")
    for path, score in similar_images:
        print(f"- {os.path.basename(path)}: {score:.3f}")

    # Visualize results
    fig = plot_similar_images(
        query_image, similar_images, title="Similar Images (Grid)"
    )
    fig.show()

    # Test edge method
    print("\n3. Edge Method")
    embedder = ImageEmbedder(method="edge")
    embedding = embedder.embed_image(query_image)
    print(f"- Embedding shape: {embedding.shape}")

    # Find similar images
    similar_images = embedder.find_similar_images(query_image, "test_images", top_k=3)
    print("\nSimilar images (edge):")
    for path, score in similar_images:
        print(f"- {os.path.basename(path)}: {score:.3f}")

    # Visualize results
    fig = plot_similar_images(
        query_image, similar_images, title="Similar Images (Edge)"
    )
    fig.show()

    # Clean up
    print("\nCleaning up test files...")
    for file in os.listdir("test_images"):
        os.remove(os.path.join("test_images", file))
    os.rmdir("test_images")
    print("Done!")


if __name__ == "__main__":
    main()
