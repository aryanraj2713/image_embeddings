"""
Comprehensive test script for imgemb package functionality.
Tests all major features including:
1. Image embedding with different methods
2. Image comparison and similarity search
3. Visualization functions
4. Error handling
5. Command-line interface operations
"""

import os
import numpy as np
import cv2
from imgemb import ImageEmbedder
import plotly.graph_objects as go
import subprocess
import json
import tempfile
import shutil


def create_test_images():
    """Create a set of test images with different shapes and colors."""
    os.makedirs("test_images", exist_ok=True)

    # Create test images
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


def test_embedding_methods():
    """Test all available embedding methods."""
    print("\nTesting embedding methods...")

    # Create test images
    image_paths = create_test_images()
    query_image = image_paths[0]  # red square

    # Test each embedding method
    methods = {
        "average_color": {"params": {"normalize": True}},
        "grid": {"params": {"grid_size": (4, 4), "normalize": True}},
        "edge": {"params": {"normalize": True}},
    }

    for method_name, config in methods.items():
        print(f"\nTesting {method_name} method:")
        try:
            # Initialize embedder
            embedder = ImageEmbedder(method=method_name, **config["params"])

            # Generate embedding for query image
            query_embedding = embedder.embed_image(query_image)
            print(f"- Generated embedding shape: {query_embedding.shape}")

            # Compare with all images
            for path in image_paths:
                similarity = embedder.compare_images(query_image, path)
                print(f"- Similarity with {os.path.basename(path)}: {similarity:.3f}")

            print("- Test passed")

        except Exception as e:
            print(f"Error testing {method_name} method: {str(e)}")


def test_error_handling():
    """Test error handling for various edge cases."""
    print("\nTesting error handling...")

    try:
        # Test invalid method
        print("Testing invalid method...")
        ImageEmbedder(method="invalid_method")
    except ValueError as e:
        print(f"- Successfully caught invalid method error: {str(e)}")

    try:
        # Test invalid image path
        print("\nTesting invalid image path...")
        embedder = ImageEmbedder(method="average_color")
        embedder.embed_image("nonexistent_image.jpg")
    except ValueError as e:
        print(f"- Successfully caught invalid image path error: {str(e)}")

    try:
        # Test invalid grid size
        print("\nTesting invalid grid size...")
        ImageEmbedder(method="grid", grid_size=(0, 0))
    except ValueError as e:
        print(f"- Successfully caught invalid grid size error: {str(e)}")


def test_cli():
    """Test CLI functionality."""
    print("\nTesting CLI functionality...")

    # Create temporary directory for test outputs
    temp_dir = tempfile.mkdtemp()
    os.makedirs(temp_dir, exist_ok=True)
    embeddings_file = os.path.join(temp_dir, "embeddings.json")

    try:
        # Test 'generate' command
        print("\nTesting 'generate' command...")
        cmd = [
            "imgemb",
            "generate",
            "test_images",
            "--output",
            embeddings_file,
            "--method",
            "average_color",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("- Generate command successful")
        else:
            print(f"- Generate command failed: {result.stderr}")

        # Test 'find-similar' command
        if os.path.exists(embeddings_file):
            print("\nTesting 'find-similar' command...")
            query_image = os.path.join("test_images", "red_square.jpg")
            cmd = [
                "imgemb",
                "find-similar",
                query_image,
                "test_images",
                "--method",
                "average_color",
                "--top-k",
                "2",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("- Find-similar command successful")
                print(f"- Output: {result.stdout}")
            else:
                print(f"- Find-similar command failed: {result.stderr}")

        # Test 'compare' command
        print("\nTesting 'compare' command...")
        image1 = os.path.join("test_images", "red_square.jpg")
        image2 = os.path.join("test_images", "blue_circle.jpg")
        cmd = ["imgemb", "compare", image1, image2, "--method", "average_color"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("- Compare command successful")
            print(f"- Output: {result.stdout}")
        else:
            print(f"- Compare command failed: {result.stderr}")

        # Test help command
        print("\nTesting help command...")
        cmd = ["imgemb", "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("- Help command successful")
        else:
            print(f"- Help command failed: {result.stderr}")

    except Exception as e:
        print(f"Error during CLI testing: {str(e)}")
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def cleanup():
    """Clean up test files."""
    print("\nCleaning up test files...")
    try:
        for file in os.listdir("test_images"):
            os.remove(os.path.join("test_images", file))
        os.rmdir("test_images")
        print("- Cleanup successful")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")


def main():
    print("Starting comprehensive functionality test for imgemb package...")

    # Test embedding methods
    test_embedding_methods()

    # Test error handling
    test_error_handling()

    # Test CLI functionality
    test_cli()

    # Clean up
    cleanup()

    print("\nTest completed!")


if __name__ == "__main__":
    main()
