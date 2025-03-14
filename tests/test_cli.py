"""Tests for the command-line interface."""

import pytest
import numpy as np
import os
import json
import tempfile
from pathlib import Path
from image_embeddings.cli.main import (
    generate_embeddings,
    find_similar,
    save_embeddings,
    load_embeddings,
    main,
    parse_args,
)
import cv2
import pickle
from image_embeddings.embedder import ImageEmbedder


@pytest.fixture
def sample_image_file(tmp_path):
    """Create a temporary sample image file."""
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(image_path), image)
    return str(image_path)


@pytest.fixture
def sample_image_dir(tmp_path):
    """Create a temporary directory with sample images."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    # Create multiple test images
    for i in range(3):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image_path = image_dir / f"test_image_{i}.jpg"
        cv2.imwrite(str(image_path), image)

    return str(image_dir)


def test_generate_embeddings_single_image(sample_image_file):
    """Test generating embeddings for a single image."""
    embeddings = generate_embeddings(sample_image_file)
    assert len(embeddings) == 1
    assert isinstance(embeddings[0], np.ndarray)


def test_generate_embeddings_directory(sample_image_dir):
    """Test generating embeddings for a directory of images."""
    embeddings = generate_embeddings(sample_image_dir)
    assert len(embeddings) == 3  # We created 3 test images
    assert all(isinstance(emb, np.ndarray) for emb in embeddings)


def test_save_and_load_embeddings(tmp_path):
    """Test saving and loading embeddings."""
    # Create sample embeddings
    embeddings = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]

    # Save embeddings
    output_file = tmp_path / "embeddings.json"
    save_embeddings(embeddings, str(output_file))

    # Load embeddings
    loaded_embeddings = load_embeddings(str(output_file))

    # Compare
    assert len(loaded_embeddings) == len(embeddings)
    for orig, loaded in zip(embeddings, loaded_embeddings):
        assert np.allclose(orig, loaded)


def test_find_similar(sample_image_dir, sample_image_file):
    """Test finding similar images."""
    # This should run without errors
    find_similar(query_image=sample_image_file, database_path=sample_image_dir, top_k=2)


def test_generate_embeddings_with_options(sample_image_file):
    """Test generating embeddings with different options."""
    # Test different methods
    methods = ["average_color", "grid", "edge"]
    for method in methods:
        embeddings = generate_embeddings(sample_image_file, method=method)
        assert len(embeddings) == 1

    # Test grid size
    embeddings = generate_embeddings(sample_image_file, method="grid", grid_size=(8, 8))
    assert embeddings[0].shape == (8 * 8 * 3,)

    # Test normalization
    embeddings = generate_embeddings(sample_image_file, normalize=False)
    assert len(embeddings) == 1


def test_generate_embeddings_invalid_path():
    """Test handling of invalid input path."""
    with pytest.raises(SystemExit):
        generate_embeddings("nonexistent_path")


def test_save_embeddings_with_output(sample_image_file, tmp_path):
    """Test generating and saving embeddings."""
    output_file = tmp_path / "embeddings.json"
    embeddings = generate_embeddings(sample_image_file, output_file=str(output_file))
    assert output_file.exists()

    # Verify the saved embeddings
    loaded_embeddings = load_embeddings(str(output_file))
    assert len(loaded_embeddings) == len(embeddings)
    assert np.allclose(loaded_embeddings[0], embeddings[0])


# Reuse the test_images fixture from test_embedder.py
@pytest.fixture
def test_images(tmp_path):
    # Create test images
    img1_path = tmp_path / "test1.jpg"
    img2_path = tmp_path / "test2.jpg"
    img3_path = tmp_path / "test3.jpg"

    # Create dummy image files (1x1 pixel images)
    for path in [img1_path, img2_path, img3_path]:
        img = np.zeros((1, 1, 3), dtype=np.uint8)
        img.fill(255)  # White image
        cv2.imwrite(str(path), img)

    return str(tmp_path), str(img1_path), str(img2_path), str(img3_path)


def test_parse_args():
    """Test argument parsing."""
    # Test compare command
    args = parse_args(["compare", "img1.jpg", "img2.jpg"])
    assert args.command == "compare"
    assert args.image1 == "img1.jpg"
    assert args.image2 == "img2.jpg"
    assert args.method == "grid"
    assert tuple(args.grid_size) == (4, 4)

    # Test generate command
    args = parse_args(
        [
            "generate",
            "images/",
            "--output",
            "embeddings.json",
            "--method",
            "edge",
            "--grid-size",
            "2",
            "2",
        ]
    )
    assert args.command == "generate"
    assert args.input == "images/"
    assert args.output == "embeddings.json"
    assert args.method == "edge"
    assert tuple(args.grid_size) == (2, 2)
    assert not args.no_normalize

    # Test find-similar command
    args = parse_args(
        [
            "find-similar",
            "query.jpg",
            "images/",
            "-k",
            "10",
            "--method",
            "average_color",
        ]
    )
    assert args.command == "find-similar"
    assert args.query_image == "query.jpg"
    assert args.image_dir == "images/"
    assert args.top_k == 10
    assert args.method == "average_color"
    assert tuple(args.grid_size) == (4, 4)


def test_main_no_args():
    """Test main function with no arguments."""
    result = main([])
    assert result == 1  # Should fail without arguments


def test_main_compare(test_images):
    """Test compare command."""
    _, img1_path, img2_path, _ = test_images

    # Test successful comparison
    result = main(["compare", img1_path, img2_path])
    assert result == 0

    # Test with nonexistent image
    result = main(["compare", "nonexistent.jpg", img2_path])
    assert result == 1

    # Test with different methods
    for method in ["average_color", "grid", "edge"]:
        result = main(["compare", img1_path, img2_path, "--method", method])
        assert result == 0


def test_main_generate(test_images, tmp_path):
    """Test generate command."""
    img_dir, _, _, _ = test_images
    output_file = tmp_path / "embeddings.json"

    # Test with directory input
    result = main(["generate", img_dir, "--output", str(output_file)])
    assert result == 0
    assert output_file.exists()

    # Test with different methods
    for method in ["average_color", "grid", "edge"]:
        result = main(["generate", img_dir, "--method", method])
        assert result == 0


def test_main_find_similar(test_images):
    """Test find-similar command."""
    img_dir, img1_path, _, _ = test_images

    # Test successful search
    result = main(["find-similar", img1_path, img_dir])
    assert result == 0

    # Test with nonexistent query image
    result = main(["find-similar", "nonexistent.jpg", img_dir])
    assert result == 1

    # Test with nonexistent directory
    result = main(["find-similar", img1_path, "nonexistent/"])
    assert result == 1

    # Test with custom top-k and method
    result = main(
        ["find-similar", img1_path, img_dir, "-k", "2", "--method", "average_color"]
    )
    assert result == 0


def test_generate_embeddings_error_handling(tmp_path):
    """Test error handling in generate_embeddings function."""
    # Test with non-existent directory
    with pytest.raises(SystemExit):
        generate_embeddings("nonexistent_dir", "output.pkl", method="average_color")

    # Test with invalid method
    with pytest.raises(ValueError) as exc_info:
        embedder = ImageEmbedder(method="invalid_method")
    assert "Invalid method: invalid_method" in str(exc_info.value)


def test_find_similar_error_handling(tmp_path):
    """Test error handling in find_similar function."""
    # Create a test image
    test_img_path = tmp_path / "test.jpg"
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(test_img_path), img)

    # Test with non-existent query image
    with pytest.raises(ValueError) as exc_info:
        embedder = ImageEmbedder()
        embedder.embed_image("nonexistent.jpg")
    assert "Could not load image at" in str(exc_info.value)

    # Test with non-existent directory
    with pytest.raises(ValueError) as exc_info:
        embedder = ImageEmbedder()
        embedder.find_similar_images(str(test_img_path), "nonexistent_dir")
    assert "Directory does not exist" in str(exc_info.value)


def test_help_text(capsys):
    """Test help text display."""
    with pytest.raises(SystemExit):
        main(["--help"])
    captured = capsys.readouterr()
    assert "usage:" in captured.out
    assert "compare" in captured.out
    assert "generate" in captured.out
    assert "find-similar" in captured.out


def test_version_display(capsys):
    """Test version information display."""
    with pytest.raises(SystemExit):
        main(["--version"])
    captured = capsys.readouterr()
    assert "error: unrecognized arguments: --version" in captured.err


def test_main_error_handling(capsys):
    """Test error handling in main function."""
    # Test with invalid command
    with pytest.raises(SystemExit):
        main(["invalid_command"])
    captured = capsys.readouterr()
    assert "invalid choice: 'invalid_command'" in captured.err

    # Test with missing required arguments
    with pytest.raises(SystemExit):
        main(["generate"])
    captured = capsys.readouterr()
    assert "error:" in captured.err
