"""Command-line interface for image embeddings tools."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import json
import os
from ..embedder import ImageEmbedder
from ..semantic_search import SemanticSearcher


def save_embeddings(embeddings: List[np.ndarray], output_file: str) -> None:
    """Save embeddings to a JSON file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Convert embeddings to list for JSON serialization
        embeddings_list = [emb.tolist() for emb in embeddings]
        with open(output_file, "w") as f:
            json.dump(embeddings_list, f)
    except Exception as e:
        raise ValueError(f"Failed to save embeddings: {e}")


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


def parse_args(args=None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Image embeddings and semantic search tool"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two images")
    compare_parser.add_argument("image1", help="First image path")
    compare_parser.add_argument("image2", help="Second image path")
    compare_parser.add_argument(
        "--method",
        choices=["average_color", "grid", "edge"],
        default="grid",
        help="Embedding method",
    )
    compare_parser.add_argument(
        "--grid-size",
        nargs=2,
        type=int,
        default=(4, 4),
        help="Grid size for grid method (height width)",
    )

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate", help="Generate embeddings for images"
    )
    generate_parser.add_argument("input", help="Input image or directory path")
    generate_parser.add_argument("--output", help="Output path for embeddings")
    generate_parser.add_argument(
        "--method",
        choices=["average_color", "grid", "edge"],
        default="grid",
        help="Embedding method",
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
    find_similar_parser = subparsers.add_parser(
        "find-similar", help="Find similar images"
    )
    find_similar_parser.add_argument("query_image", help="Query image path")
    find_similar_parser.add_argument("image_dir", help="Directory containing images")
    find_similar_parser.add_argument(
        "-k", "--top-k", type=int, default=5, help="Number of similar images to return"
    )
    find_similar_parser.add_argument(
        "--method",
        choices=["average_color", "grid", "edge"],
        default="grid",
        help="Embedding method",
    )
    find_similar_parser.add_argument(
        "--grid-size",
        nargs=2,
        type=int,
        default=(4, 4),
        help="Grid size for grid method (height width)",
    )

    # Search command
    search_parser = subparsers.add_parser(
        "search", help="Search images using text query"
    )
    search_parser.add_argument("query", help="Text query")
    search_parser.add_argument("directory", help="Directory containing images")
    search_parser.add_argument(
        "-k", "--top-k", type=int, default=5, help="Number of results to return"
    )
    search_parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum similarity score threshold",
    )

    return parser.parse_args(args)


def compare_command(args: argparse.Namespace) -> int:
    """Execute compare command."""
    try:
        # Validate paths
        if not Path(args.image1).exists():
            print(f"Error: Image not found: {args.image1}", file=sys.stderr)
            return 1
        if not Path(args.image2).exists():
            print(f"Error: Image not found: {args.image2}", file=sys.stderr)
            return 1

        # Initialize embedder
        embedder = ImageEmbedder(
            method=args.method, grid_size=tuple(map(int, args.grid_size))
        )

        # Compare images
        similarity = embedder.compare_images(args.image1, args.image2)
        print(f"Similarity score: {similarity:.4f}")
        return 0

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


def generate_command(args: argparse.Namespace) -> int:
    """Execute generate command."""
    try:
        # Validate paths
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input path not found: {args.input}", file=sys.stderr)
            return 1

        # Initialize embedder
        embedder = ImageEmbedder(
            method=args.method, grid_size=tuple(map(int, args.grid_size))
        )

        # Generate embeddings
        if input_path.is_file():
            embedding = embedder.embed_image(str(input_path))
            print(f"Generated embedding shape: {embedding.shape}")
            if args.output:
                np.save(args.output, embedding)
                print(f"Saved embedding to {args.output}")
        else:
            embeddings = []
            for img_path in input_path.glob("**/*.jpg"):
                try:
                    embedding = embedder.embed_image(str(img_path))
                    embeddings.append(embedding)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}", file=sys.stderr)

            if embeddings:
                embeddings = np.array(embeddings)
                print(f"Generated {len(embeddings)} embeddings")
                if args.output:
                    np.save(args.output, embeddings)
                    print(f"Saved embeddings to {args.output}")
            else:
                print("No images processed successfully.")

        return 0

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


def find_similar_command(args: argparse.Namespace) -> int:
    """Execute find-similar command."""
    try:
        # Validate paths
        if not Path(args.query_image).exists():
            print(f"Error: Query image not found: {args.query_image}", file=sys.stderr)
            return 1
        if not Path(args.image_dir).exists():
            print(f"Error: Directory not found: {args.image_dir}", file=sys.stderr)
            return 1

        # Initialize embedder
        embedder = ImageEmbedder(
            method=args.method, grid_size=tuple(map(int, args.grid_size))
        )

        # Find similar images
        similar_images = embedder.find_similar_images(
            args.query_image, args.image_dir, top_k=args.top_k
        )

        # Print results
        print("\nTop similar images:")
        for path, score in similar_images:
            print(f"{path}: {score:.4f}")

        return 0

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


def search_command(args: argparse.Namespace) -> int:
    """Execute semantic search command."""
    try:
        # Validate directory
        if not Path(args.directory).exists():
            print(f"Error: Directory not found: {args.directory}", file=sys.stderr)
            return 1

        # Initialize searcher
        searcher = SemanticSearcher(device="cpu")  # Default to CPU for stability

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

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main(args=None) -> int:
    """Main entry point."""
    try:
        parsed_args = parse_args(args)

        # Execute appropriate command
        if parsed_args.command == "compare":
            return compare_command(parsed_args)
        elif parsed_args.command == "generate":
            return generate_command(parsed_args)
        elif parsed_args.command == "find-similar":
            return find_similar_command(parsed_args)
        elif parsed_args.command == "search":
            return search_command(parsed_args)
        else:
            print(f"Error: Unknown command: {parsed_args.command}", file=sys.stderr)
            return 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
