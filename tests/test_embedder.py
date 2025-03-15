"""Tests for the image_embeddings library."""

import pytest
import numpy as np
import os
from pathlib import Path
import cv2
from image_embeddings.embedder import ImageEmbedder
import tempfile


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


def test_average_color_embedder(sample_image):
    """Test the average color embedding method."""
    embedder = ImageEmbedder(method="average_color")
    embedding = embedder.embed(sample_image)

    # Check shape (should be 3 for RGB)
    assert embedding.shape == (3,)

    # Check if values are normalized
    assert np.abs(np.linalg.norm(embedding) - 1.0) < 1e-6

    # Check if values are in reasonable range
    assert np.all(embedding >= -1) and np.all(embedding <= 1)


def test_grid_embedder(sample_image):
    """Test the grid-based embedding method."""
    grid_size = (4, 4)
    embedder = ImageEmbedder(method="grid", grid_size=grid_size)
    embedding = embedder.embed(sample_image)

    # Check shape (grid_h * grid_w * channels)
    expected_shape = (grid_size[0] * grid_size[1] * 3,)
    assert embedding.shape == expected_shape

    # Check if values are normalized
    assert np.abs(np.linalg.norm(embedding) - 1.0) < 1e-6


def test_edge_embedder(sample_image):
    """Test the edge-based embedding method."""
    embedder = ImageEmbedder(method="edge")
    embedding = embedder.embed(sample_image)

    # Check shape (64 bins for histogram)
    assert embedding.shape == (64,)

    # Check if values are normalized
    assert np.abs(np.linalg.norm(embedding) - 1.0) < 1e-6

    # Check if values are non-negative (it's a histogram)
    assert np.all(embedding >= 0)


def test_invalid_method():
    """Test handling of invalid embedding method."""
    with pytest.raises(ValueError):
        ImageEmbedder(method="invalid_method")


def test_invalid_image_path():
    """Test handling of invalid image path."""
    embedder = ImageEmbedder()
    with pytest.raises(ValueError):
        embedder.embed_image("nonexistent_image.jpg")


def test_normalization():
    """Test embedding normalization."""
    # Create embedder with normalization off
    embedder = ImageEmbedder(normalize=False)
    image = np.ones((100, 100, 3), dtype=np.uint8) * 128

    embedding = embedder.embed(image)
    # Should be all 128s
    assert np.allclose(embedding, 128.0)

    # Create embedder with normalization on
    embedder = ImageEmbedder(normalize=True)
    embedding = embedder.embed(image)
    # Should be normalized to unit length
    assert np.abs(np.linalg.norm(embedding) - 1.0) < 1e-6


def test_grid_size_validation():
    """Test grid size validation."""
    # Valid grid size
    embedder = ImageEmbedder(method="grid", grid_size=(2, 2))
    assert embedder.grid_size == (2, 2)

    # Invalid grid size type
    with pytest.raises(ValueError):
        ImageEmbedder(grid_size=[2, 2])

    # Invalid grid size length
    with pytest.raises(ValueError):
        ImageEmbedder(grid_size=(2,))


# Create a fixture for test images
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


def test_init():
    """Test ImageEmbedder initialization."""
    # Test default initialization
    embedder = ImageEmbedder()
    assert embedder.target_size == (224, 224)
    assert embedder.method == "grid"
    assert embedder.grid_size == (4, 4)
    assert embedder.normalize == True

    # Test custom initialization
    embedder = ImageEmbedder(
        target_size=(128, 128),
        method="average_color",
        grid_size=(2, 2),
        normalize=False,
    )
    assert embedder.target_size == (128, 128)
    assert embedder.method == "average_color"
    assert embedder.grid_size == (2, 2)
    assert embedder.normalize == False


def test_preprocess_image(test_images):
    """Test image preprocessing."""
    _, img_path, _, _ = test_images
    embedder = ImageEmbedder(target_size=(32, 32))

    # Test successful preprocessing
    img = embedder.preprocess_image(img_path)
    assert img.shape == (32, 32, 3)
    assert img.dtype == np.float32
    assert np.all(img >= 0) and np.all(img <= 1)

    # Test invalid image path
    with pytest.raises(ValueError):
        embedder.preprocess_image("nonexistent.jpg")


def test_generate_embedding(test_images):
    """Test embedding generation."""
    _, img_path, _, _ = test_images
    embedder = ImageEmbedder(target_size=(32, 32))

    embedding = embedder.generate_embedding(img_path)

    # Check embedding shape for grid method (4x4 grid, 3 colors per cell)
    expected_size = 4 * 4 * 3  # grid_size * grid_size * RGB
    assert embedding.shape == (expected_size,)

    # Check normalization
    assert np.abs(np.linalg.norm(embedding) - 1.0) < 1e-6


def test_compare_images(test_images):
    """Test image comparison."""
    _, img1_path, img2_path, _ = test_images
    embedder = ImageEmbedder()

    # Compare image with itself
    similarity = embedder.compare_images(img1_path, img1_path)
    assert np.abs(similarity - 1.0) < 1e-6  # Should be perfectly similar

    # Compare with different image
    similarity = embedder.compare_images(img1_path, img2_path)
    assert 0 <= similarity <= 1  # Similarity should be between 0 and 1


def test_find_similar_images(test_images):
    """Test finding similar images."""
    img_dir, img1_path, _, _ = test_images
    embedder = ImageEmbedder()

    # Find similar images
    results = embedder.find_similar_images(img1_path, img_dir, top_k=2)

    # Check results format
    assert len(results) <= 2  # Should return at most top_k results
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    assert all(isinstance(r[0], str) and isinstance(r[1], float) for r in results)

    # Check similarity scores
    assert all(0 <= score <= 1 for _, score in results)

    # Check sorting
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_invalid_grid_size():
    """Test grid size validation."""
    # Invalid grid size type
    with pytest.raises(ValueError):
        ImageEmbedder(grid_size=[2, 2])

    # Invalid grid size length
    with pytest.raises(ValueError):
        ImageEmbedder(grid_size=(2,))


def test_average_color_embedder(test_images):
    """Test average color embedding method."""
    _, img_path, _, _ = test_images
    embedder = ImageEmbedder(method="average_color")

    embedding = embedder.embed_image(img_path)
    assert embedding.shape == (3,)  # RGB values
    assert np.all(embedding >= 0) and np.all(embedding <= 1)


def test_grid_embedder(test_images):
    """Test grid-based embedding method."""
    _, img_path, _, _ = test_images
    grid_size = (4, 4)
    embedder = ImageEmbedder(method="grid", grid_size=grid_size)

    embedding = embedder.embed_image(img_path)
    expected_size = grid_size[0] * grid_size[1] * 3  # grid cells * RGB
    assert embedding.shape == (expected_size,)


def test_edge_embedder(test_images):
    """Test edge-based embedding method."""
    _, img_path, _, _ = test_images
    embedder = ImageEmbedder(method="edge")

    embedding = embedder.embed_image(img_path)
    assert embedding.shape == (32,)  # 32 histogram bins


def test_normalization(test_images):
    """Test embedding normalization."""
    _, img_path, _, _ = test_images

    # Test with normalization
    embedder = ImageEmbedder(normalize=True)
    embedding = embedder.embed_image(img_path)
    assert np.abs(np.linalg.norm(embedding) - 1.0) < 1e-6

    # Test without normalization
    embedder = ImageEmbedder(normalize=False)
    embedding = embedder.embed_image(img_path)
    assert np.linalg.norm(embedding) != 1.0


def test_find_similar_images_empty_directory(tmp_path):
    """Test find_similar_images with empty directory."""
    embedder = ImageEmbedder()
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    # Create a query image
    query_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    query_path = tmp_path / "query.jpg"
    cv2.imwrite(str(query_path), query_img)

    results = embedder.find_similar_images(str(query_path), str(empty_dir))
    assert len(results) == 0


def test_compare_images_same_image(tmp_path):
    """Test comparing an image with itself."""
    embedder = ImageEmbedder()

    # Create a test image
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img_path = tmp_path / "test.jpg"
    cv2.imwrite(str(img_path), img)

    similarity = embedder.compare_images(str(img_path), str(img_path))
    assert similarity == pytest.approx(1.0)


def test_edge_embedder_empty_image(tmp_path):
    """Test edge embedder with an empty (black) image."""
    embedder = ImageEmbedder(method="edge")

    # Create a black image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img_path = tmp_path / "black.jpg"
    cv2.imwrite(str(img_path), img)

    embedding = embedder.embed_image(str(img_path))
    assert embedding is not None
    assert not np.isnan(embedding).any()


def test_grid_embedder_single_cell():
    """Test grid embedder with 1x1 grid."""
    embedder = ImageEmbedder(method="grid", grid_size=(1, 1))
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128

    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        cv2.imwrite(tmp.name, img)
        embedding = embedder.embed_image(tmp.name)
        assert embedding.shape == (3,)  # RGB values for single cell
        # Values should be normalized to [0, 1] range
        expected = np.array([0.5, 0.5, 0.5])  # 128/255 ≈ 0.5
        # Use more lenient tolerance to account for JPEG compression
        np.testing.assert_allclose(embedding, expected, rtol=1e-2)


def test_embedder_color_spaces():
    """Test embedder with different color spaces."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128

    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        cv2.imwrite(tmp.name, img)

        # Test RGB
        embedder_rgb = ImageEmbedder(color_space="rgb")
        emb_rgb = embedder_rgb.embed_image(tmp.name)

        # Test HSV
        embedder_hsv = ImageEmbedder(color_space="hsv")
        emb_hsv = embedder_hsv.embed_image(tmp.name)

        assert emb_rgb.shape == emb_hsv.shape
        assert not np.array_equal(emb_rgb, emb_hsv)
