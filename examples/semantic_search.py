"""
Example script demonstrating semantic search features of imgemb.
"""

import os
from imgemb import SemanticSearcher, plot_similar_images


def main():
    """Main function demonstrating semantic search features."""
    # Initialize searcher with GPU if available
    searcher = SemanticSearcher(
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    )

    # Example directory containing images
    image_dir = (
        "test_images"  # Use the same directory created by visualization_example.py
    )

    if not os.path.exists(image_dir):
        print(f"Please run visualization_example.py first to create {image_dir}")
        return

    # Index the directory
    print(f"\nIndexing images in {image_dir}...")
    searcher.index_directory(image_dir)

    # Example text queries
    queries = [
        "a red shape",
        "a blue circle",
        "a yellow star",
        "a geometric shape",
        "a purple rectangle",
    ]

    # Search for each query
    for query in queries:
        print(f"\nSearching for: {query}")
        results = searcher.search(
            query, top_k=3, threshold=0.2  # Lower threshold for demonstration
        )

        if not results:
            print("No results found.")
            continue

        # Print results
        print("Results:")
        for path, score in results:
            print(f"  {os.path.basename(path)}: {score:.3f}")

        # Create visualization
        fig = plot_similar_images(
            results[0][0],  # Use first result as query image
            results[1:],  # Remaining results
            title=f'Results for "{query}"',
        )

        # Save plot
        output_file = f"semantic_search_{query.replace(' ', '_')}.html"
        fig.write_html(output_file)
        print(f"Saved visualization to {output_file}")


if __name__ == "__main__":
    main()
