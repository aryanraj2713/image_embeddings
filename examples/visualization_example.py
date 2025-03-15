"""
Example script demonstrating the visualization features of imgemb.
"""

import os
import numpy as np
import cv2
from imgemb import ImageEmbedder, plot_similar_images


def create_test_images(output_dir="test_images"):
    """Create test images for demonstration."""
    os.makedirs(output_dir, exist_ok=True)

    # Create images of different shapes and colors
    images = {
        "red_square.jpg": np.full((100, 100, 3), [255, 0, 0], dtype=np.uint8),
        "blue_circle.jpg": np.zeros((100, 100, 3), dtype=np.uint8),
        "green_triangle.jpg": np.zeros((100, 100, 3), dtype=np.uint8),
        "yellow_star.jpg": np.zeros((100, 100, 3), dtype=np.uint8),
        "purple_rectangle.jpg": np.full((100, 100, 3), [128, 0, 128], dtype=np.uint8),
    }

    # Draw shapes
    cv2.circle(images["blue_circle.jpg"], (50, 50), 40, (255, 0, 0), -1)
    pts = np.array([[50, 20], [20, 80], [80, 80]], np.int32)
    cv2.fillPoly(images["green_triangle.jpg"], [pts], (0, 255, 0))
    cv2.rectangle(images["purple_rectangle.jpg"], (20, 30), (80, 70), (128, 0, 128), -1)

    # Draw star
    center = (50, 50)
    outer_radius = 40
    inner_radius = 20
    num_points = 5
    points = []
    for i in range(num_points * 2):
        angle = i * np.pi / num_points
        radius = outer_radius if i % 2 == 0 else inner_radius
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        points.append([int(x), int(y)])
    pts = np.array(points, np.int32)
    cv2.fillPoly(images["yellow_star.jpg"], [pts], (0, 255, 255))

    # Save images
    for name, img in images.items():
        cv2.imwrite(os.path.join(output_dir, name), img)

    return output_dir


def main():
    """Main function demonstrating visualization features."""
    # Create test images
    image_dir = create_test_images()

    # Initialize embedder with different methods
    methods = ["average_color", "grid", "edge"]

    for method in methods:
        print(f"\nTesting {method} method:")

        # Initialize embedder with appropriate parameters
        kwargs = {}
        if method == "grid":
            kwargs["grid_size"] = (4, 4)

        embedder = ImageEmbedder(method=method, **kwargs)

        # Find similar images for each shape
        query_images = ["red_square.jpg", "blue_circle.jpg", "yellow_star.jpg"]

        for query in query_images:
            query_path = os.path.join(image_dir, query)
            similar_images = embedder.find_similar_images(
                query_path, image_dir, top_k=3
            )

            print(f"\nSimilar images to {query}:")
            for path, score in similar_images:
                print(f"  {os.path.basename(path)}: {score:.3f}")

            # Create visualization
            fig = plot_similar_images(
                query_path,
                similar_images,
                title=f"Similar Images to {query} ({method} method)",
            )

            # Save plot
            output_file = f"similar_{os.path.splitext(query)[0]}_{method}.html"
            fig.write_html(output_file)
            print(f"Saved visualization to {output_file}")


if __name__ == "__main__":
    main()
