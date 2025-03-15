"""
Test script demonstrating image embedding and similarity search functionality.
This script shows how to:
1. Initialize different types of embedders
2. Generate embeddings for images
3. Find similar images using different methods
4. Visualize the results using plotly
"""

import os
from imgemb import ImageEmbedder
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
import base64
import numpy as np
from typing import List, Tuple


def find_similar_images(
    embedder: ImageEmbedder, query_image: str, image_paths: List[str]
) -> Tuple[List[str], List[float]]:
    """Find similar images using the embedder and return paths and scores."""
    # Generate embeddings for all images and compute similarities
    similarities = []
    for path in image_paths:
        score = embedder.compare_images(query_image, path)
        similarities.append((path, score))

    # Sort by similarity score (higher is better)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Split into paths and scores
    paths, scores = zip(*similarities)
    return list(paths), list(scores)


def display_images_with_scores(image_paths, scores, title="Search Results"):
    """Display images with their similarity scores using plotly."""
    n_images = len(image_paths)
    fig = make_subplots(
        rows=1,
        cols=n_images,
        subplot_titles=[f"Score: {score:.3f}" for score in scores],
    )

    def img_to_base64(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode(".png", img)
        return base64.b64encode(buffer).decode()

    for idx, (img_path, score) in enumerate(zip(image_paths, scores), 1):
        try:
            # Add image to subplot
            fig.add_trace(
                go.Image(source=f"data:image/png;base64,{img_to_base64(img_path)}"),
                row=1,
                col=idx,
            )
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    fig.update_layout(
        title=title, height=400, showlegend=False, margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.show()


def main():
    # Initialize embedders with different methods
    embedders = {
        "average_color": ImageEmbedder(method="average_color", normalize=True),
        "grid": ImageEmbedder(method="grid", grid_size=(4, 4), normalize=True),
        "edge": ImageEmbedder(method="edge", normalize=True),
    }

    # Create a directory for test images if it doesn't exist
    os.makedirs("test_images", exist_ok=True)

    # Create some test images (simple shapes with different colors)
    test_images = {
        "red_square": (255, 0, 0),
        "blue_circle": (0, 0, 255),
        "green_triangle": (0, 255, 0),
        "yellow_star": (255, 255, 0),
    }

    image_paths = []
    for name, color in test_images.items():
        img_path = f"test_images/{name}.jpg"
        img = np.zeros((300, 300, 3), dtype=np.uint8)

        if "square" in name:
            cv2.rectangle(img, (100, 100), (200, 200), color, -1)
        elif "circle" in name:
            cv2.circle(img, (150, 150), 50, color, -1)
        elif "triangle" in name:
            pts = np.array([[150, 100], [100, 200], [200, 200]], np.int32)
            cv2.fillPoly(img, [pts], color)
        else:  # star
            center = (150, 150)
            pts = []
            for i in range(5):
                angle = i * 2 * np.pi / 5 - np.pi / 2
                outer_pt = (
                    int(center[0] + 50 * np.cos(angle)),
                    int(center[1] + 50 * np.sin(angle)),
                )
                pts.append(outer_pt)
                inner_angle = angle + np.pi / 5
                inner_pt = (
                    int(center[0] + 25 * np.cos(inner_angle)),
                    int(center[1] + 25 * np.sin(inner_angle)),
                )
                pts.append(inner_pt)
            pts = np.array(pts, np.int32)
            cv2.fillPoly(img, [pts], color)

        cv2.imwrite(img_path, img)
        image_paths.append(img_path)

    print("Testing image similarity search with different methods...")

    # Use the first image (red square) as query
    query_image = image_paths[0]

    for method_name, embedder in embedders.items():
        print(f"\nTesting {method_name} method:")
        # Find similar images
        similar_images, scores = find_similar_images(embedder, query_image, image_paths)

        # Display results
        display_images_with_scores(
            similar_images,
            scores,
            f"Results for {method_name} method (query: red square)",
        )

        # Print scores
        print("Results:")
        for img_path, score in zip(similar_images, scores):
            print(f"{os.path.basename(img_path)}: {score:.3f}")

    # Clean up
    print("\nCleaning up test images...")
    for path in image_paths:
        if os.path.exists(path):
            os.remove(path)
    os.rmdir("test_images")


if __name__ == "__main__":
    main()
