"""
Example script demonstrating semantic image search using CLIP.
This script shows how to:
1. Initialize the semantic searcher
2. Index a directory of images
3. Search using natural language queries
4. Visualize search results
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from imgemb import SemanticSearcher


def display_results(
    query: str, results: list, num_cols: int = 5, figsize: tuple = (15, 3)
) -> None:
    """Display search results in a grid.

    Args:
        query (str): Search query
        results (list): List of (path, score) tuples
        num_cols (int): Number of columns in the grid
        figsize (tuple): Figure size (width, height)
    """
    num_images = len(results)
    num_rows = (num_images + num_cols - 1) // num_cols

    plt.figure(figsize=figsize)
    plt.suptitle(f'Results for: "{query}"', fontsize=14)

    for idx, (path, score) in enumerate(results, 1):
        plt.subplot(num_rows, num_cols, idx)
        # Read and convert image from BGR to RGB
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(f"Score: {score:.3f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def search_examples(image_dir: str) -> None:
    """Run various search examples.

    Args:
        image_dir (str): Directory containing images
    """
    # Initialize searcher
    searcher = SemanticSearcher()

    # Index directory
    searcher.index_directory(image_dir)

    # Example queries to demonstrate capabilities
    queries = [
        "a photo of a dog",
        "a beautiful sunset",
        "people smiling",
        "food on a plate",
        "a car on the road",
    ]

    # Search and display results for each query
    for query in queries:
        print(f'\nSearching for: "{query}"')
        results = searcher.search(query, top_k=5)

        # Print results
        print("\nTop matches:")
        for path, score in results:
            print(f"- {Path(path).name}: {score:.3f}")

        # Display results
        display_results(query, results)


def advanced_usage(image_dir: str) -> None:
    """Demonstrate advanced usage features.

    Args:
        image_dir (str): Directory containing images
    """
    # Initialize with specific model and device
    searcher = SemanticSearcher(
        model_name="ViT-B/32",  # You can also try "ViT-L/14" for better accuracy
        device="cuda",  # Use "cpu" if no GPU available
    )

    # Index with specific file types
    searcher.index_directory(
        image_dir,
        extensions=[".jpg", ".png"],  # Only process jpg and png
    )

    # Search with threshold
    results = searcher.search(
        "a red car",
        top_k=5,
        threshold=0.25,  # Only return results with similarity > 0.25
    )

    # Display results
    display_results("a red car (with threshold)", results)

    # Compositional queries
    complex_queries = [
        "a person wearing a red shirt",
        "a dog playing in the snow",
        "a modern building at night",
    ]

    for query in complex_queries:
        results = searcher.search(query, top_k=3)
        display_results(query, results)


def main():
    """Main function demonstrating semantic search."""
    # Set your image directory
    image_dir = "examples/images"

    if not os.path.exists(image_dir):
        print(f"Error: Directory not found: {image_dir}")
        print("Please create an 'images' directory with some test images.")
        return

    # Run basic examples
    print("\n=== Basic Search Examples ===")
    search_examples(image_dir)

    # Run advanced examples
    print("\n=== Advanced Usage Examples ===")
    advanced_usage(image_dir)


if __name__ == "__main__":
    main()
