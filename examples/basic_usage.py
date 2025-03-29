"""
Basic usage example for the imgemb library.
This script demonstrates the core functionality of the library,
including different embedding methods and visualization.
"""

import numpy as np
from imgemb import ImageEmbedder, plot_similar_images
import os
import cv2


def compare_images(embedder: ImageEmbedder, image1_path: str, image2_path: str) -> None:
    """Compare two images using cosine similarity.

    Args:
        embedder (ImageEmbedder): Initialized embedder instance
        image1_path (str): Path to first image
        image2_path (str): Path to second image
    """
    # Compare images
    similarity = embedder.compare_images(image1_path, image2_path)

    print(
        f"\nComparing {os.path.basename(image1_path)} and {os.path.basename(image2_path)}:"
    )
    print(f"Similarity score: {similarity:.4f}")

    # Visualize results
    fig = plot_similar_images(
        image1_path, [(image2_path, similarity)], title=f"Similarity: {similarity:.4f}"
    )
    fig.show()


def create_test_images():
    """Create test images for demonstration."""
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

    return image_paths


def main():
    """Main function demonstrating basic usage."""
    # Create test images
    print("Creating test images...")
    image_paths = create_test_images()

    # Initialize embedders with different methods
    embedders = {
        "average_color": ImageEmbedder(method="average_color", normalize=True),
        "grid": ImageEmbedder(method="grid", grid_size=(4, 4), normalize=True),
        "edge": ImageEmbedder(method="edge", normalize=True),
    }

    # Compare images using different methods
    for method_name, embedder in embedders.items():
        print(f"\nTesting {method_name} method:")
        compare_images(embedder, image_paths[0], image_paths[1])

    # Clean up
    print("\nCleaning up test images...")
    for path in image_paths:
        if os.path.exists(path):
            os.remove(path)
    os.rmdir("test_images")


if __name__ == "__main__":
    main()
