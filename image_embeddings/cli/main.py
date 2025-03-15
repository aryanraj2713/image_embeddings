"""Command-line interface for image embeddings tools."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import json
from ..embedder import ImageEmbedder
from ..semantic_search import SemanticSearcher


def save_embeddings(embeddings: List[np.ndarray], output_file: str) -> None:
    """Save embeddings to a JSON file."""
    # Convert embeddings to list for JSON serialization
    embeddings_list = [emb.tolist() for emb in embeddings]
    with open(output_file, "w") as f:
        json.dump(embeddings_list, f)


def load_embeddings(input_file: str) -> List[np.ndarray]:
    """Load embeddings from a JSON file."""
    with open(input_file, "r") as f:
        embeddings_list = json.load(f)
    return [np.array(emb) for emb in embeddings_list]


def generate_embeddings(
    input_path: str,
    output_file: Optional[str] = None,
    method: str = "grid",
    grid_size: tuple = (4, 4),
    normalize: bool = True,
) -> List[np.ndarray]:
    """Generate embeddings for images in the input path."""
    # Initialize embedder
    embedder = ImageEmbedder(method=method, grid_size=grid_size, normalize=normalize)

    # Handle single image or directory
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"Error: Path does not exist: {input_path}")
        sys.exit(1)

    embeddings = []
    if input_path.is_file():
        try:
            embedding = embedder.embed_image(str(input_path))
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            sys.exit(1)
    else:
        # Process all images in directory
        image_files = [
            f
            for f in input_path.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
        for image_file in image_files:
            try:
                embedding = embedder.embed_image(str(image_file))
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                continue

    # Save embeddings if output file specified
    if output_file:
        save_embeddings(embeddings, output_file)

    return embeddings


def find_similar(
    query_image: str,
    database_path: str,
    top_k: int = 5,
    method: str = "grid",
    grid_size: tuple = (4, 4),
) -> None:
    """Find similar images to the query image."""
    # Generate embedding for query image
    embedder = ImageEmbedder(method=method, grid_size=grid_size)

    # Find similar images
    results = embedder.find_similar_images(query_image, database_path, top_k)

    # Print results
    print(f"\nTop {len(results)} similar images:")
    for path, score in results:
        print(f"{path}: {score:.4f}")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Image embeddings and semantic search tool"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Semantic search command
    search_parser = subparsers.add_parser(
        "search", help="Search for images using text queries"
    )
    search_parser.add_argument(
        "directory", help="Directory containing images to search"
    )
    search_parser.add_argument("query", help="Text query (e.g., 'a photo of a dog')")
    search_parser.add_argument(
        "-k", "--top-k", type=int, default=5, help="Number of results to return"
    )
    search_parser.add_argument(
        "--threshold", type=float, default=0.0, help="Minimum similarity score (0 to 1)"
    )

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare", help="Compare two images for similarity"
    )
    compare_parser.add_argument("image1", help="Path to first image")
    compare_parser.add_argument("image2", help="Path to second image")
    compare_parser.add_argument(
        "--method",
        choices=ImageEmbedder.VALID_METHODS,
        default="grid",
        help="Embedding method to use",
    )
    compare_parser.add_argument(
        "--grid-size",
        nargs=2,
        type=int,
        default=(4, 4),
        help="Grid size for grid method (height width)",
    )

    # Generate embeddings command
    generate_parser = subparsers.add_parser(
        "generate", help="Generate embeddings for images"
    )
    generate_parser.add_argument("input", help="Input image or directory")
    generate_parser.add_argument("--output", help="Output JSON file to save embeddings")
    generate_parser.add_argument(
        "--method",
        choices=ImageEmbedder.VALID_METHODS,
        default="grid",
        help="Embedding method to use",
    )
    generate_parser.add_argument(
        "--grid-size",
        nargs=2,
        type=int,
        default=(4, 4),
        help="Grid size for grid method (height width)",
    )
    generate_parser.add_argument(
        "--no-normalize", action="store_true", help="Disable embedding normalization"
    )

    # Find similar command
    similar_parser = subparsers.add_parser(
        "find-similar", help="Find similar images in a directory"
    )
    similar_parser.add_argument("query_image", help="Path to query image")
    similar_parser.add_argument(
        "image_dir", help="Directory containing images to search"
    )
    similar_parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Number of similar images to return (default: 5)",
    )
    similar_parser.add_argument(
        "--method",
        choices=ImageEmbedder.VALID_METHODS,
        default="grid",
        help="Embedding method to use",
    )
    similar_parser.add_argument(
        "--grid-size",
        nargs=2,
        type=int,
        default=(4, 4),
        help="Grid size for grid method (height width)",
    )

    return parser.parse_args(args)


def search_command(args: argparse.Namespace) -> None:
    """Execute semantic search command."""
    try:
        # Initialize searcher
        searcher = SemanticSearcher()

        # Index directory
        searcher.index_directory(args.directory)

        # Perform search
        results = searcher.search(
            args.query, top_k=args.top_k, threshold=args.threshold
        )

        # Print results
        print("\nSearch Results:")
        print("-" * 50)
        for path, score in results:
            print(f"Score: {score:.3f} - {path}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI."""
    if args is None:
        args = sys.argv[1:]

    parsed_args = parse_args(args)

    if not parsed_args.command:
        print("Error: No command specified. Use --help for usage information.")
        return 1

    try:
        if parsed_args.command == "compare":
            # Validate paths
            if not Path(parsed_args.image1).exists():
                print(f"Error: Image not found: {parsed_args.image1}")
                return 1
            if not Path(parsed_args.image2).exists():
                print(f"Error: Image not found: {parsed_args.image2}")
                return 1

            # Compare images
            embedder = ImageEmbedder(
                method=parsed_args.method, grid_size=tuple(parsed_args.grid_size)
            )
            similarity = embedder.compare_images(parsed_args.image1, parsed_args.image2)
            print(f"Similarity score: {similarity:.4f}")

        elif parsed_args.command == "generate":
            # Generate embeddings
            embeddings = generate_embeddings(
                parsed_args.input,
                parsed_args.output,
                parsed_args.method,
                tuple(parsed_args.grid_size),
                not parsed_args.no_normalize,
            )
            print(f"Generated {len(embeddings)} embeddings")

        elif parsed_args.command == "find-similar":
            # Validate paths
            if not Path(parsed_args.query_image).exists():
                print(f"Error: Query image not found: {parsed_args.query_image}")
                return 1
            if not Path(parsed_args.image_dir).is_dir():
                print(f"Error: Directory not found: {parsed_args.image_dir}")
                return 1

            # Find similar images
            find_similar(
                parsed_args.query_image,
                parsed_args.image_dir,
                parsed_args.top_k,
                parsed_args.method,
                tuple(parsed_args.grid_size),
            )

        elif parsed_args.command == "search":
            search_command(parsed_args)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
