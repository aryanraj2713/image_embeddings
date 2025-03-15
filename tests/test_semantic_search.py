"""Tests for semantic search functionality."""

import os
import pytest
import torch
from PIL import Image
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from image_embeddings.semantic_search import SemanticSearcher


@pytest.fixture
def test_images_dir(tmp_path):
    """Create temporary test images."""
    # Create test directory
    img_dir = tmp_path / "test_images"
    img_dir.mkdir()

    # Create some test images
    sizes = [(100, 100), (200, 150), (150, 200)]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    image_paths = []
    for i, (size, color) in enumerate(zip(sizes, colors)):
        img = Image.new("RGB", size, color)
        path = img_dir / f"test_image_{i}.jpg"
        img.save(path)
        image_paths.append(path)

    return img_dir


@pytest.fixture
def mock_clip():
    """Mock CLIP model and preprocessing."""
    # Create mock parameters with device property
    mock_param = MagicMock()
    mock_param.device.type = "cpu"

    # Create consistent embeddings for testing
    def get_consistent_embedding(batch_size=1):
        # Create a normalized embedding
        emb = torch.ones(batch_size, 512)
        return emb / emb.norm(dim=-1, keepdim=True)

    # Create mock model
    mock_model = MagicMock()
    mock_model.encode_image.return_value = get_consistent_embedding()
    mock_model.encode_text.return_value = get_consistent_embedding()
    mock_model.parameters.return_value = iter([mock_param])

    # Create mock preprocess function
    mock_preprocess = MagicMock()
    mock_preprocess.return_value = torch.randn(1, 3, 224, 224)

    with patch("clip.load") as mock_load:
        mock_load.return_value = (mock_model, mock_preprocess)
        yield mock_load


def test_initialization(mock_clip):
    """Test SemanticSearcher initialization."""
    # Test default initialization
    searcher = SemanticSearcher()
    assert searcher.device in ["cuda", "cpu"]
    assert searcher.model is not None
    assert searcher.preprocess is not None

    # Test specific device
    searcher = SemanticSearcher(device="cpu")
    assert searcher.device == "cpu"

    # Test model variant
    searcher = SemanticSearcher(model_name="ViT-B/32")
    assert searcher.model is not None


def test_image_embedding(mock_clip, test_images_dir):
    """Test image embedding generation."""
    searcher = SemanticSearcher(device="cpu")

    # Get first test image
    test_image = next(test_images_dir.glob("*.jpg"))

    # Generate embedding
    embedding = searcher._get_image_embedding(str(test_image))

    # Check embedding properties
    assert isinstance(embedding, torch.Tensor)
    assert embedding.dim() == 2  # Should be 2D tensor
    assert embedding.shape[0] == 1  # Batch size 1
    assert embedding.shape[1] == 512  # CLIP feature dimension


def test_text_embedding(mock_clip):
    """Test text embedding generation."""
    searcher = SemanticSearcher(device="cpu")

    # Test queries
    queries = ["a red image", "a photo of a dog", "an abstract painting"]

    for query in queries:
        embedding = searcher._get_text_embedding(query)

        # Check embedding properties
        assert isinstance(embedding, torch.Tensor)
        assert embedding.dim() == 2
        assert embedding.shape[0] == 1
        assert embedding.shape[1] == 512


def test_index_directory(mock_clip, test_images_dir):
    """Test directory indexing."""
    searcher = SemanticSearcher(device="cpu")

    # Index directory
    searcher.index_directory(str(test_images_dir))

    # Check if all images were indexed
    num_images = len(list(test_images_dir.glob("*.jpg")))
    assert len(searcher._image_embeddings) == num_images
    assert len(searcher._image_paths) == num_images

    # Check with specific extensions
    searcher.index_directory(str(test_images_dir), extensions=[".jpg"])
    assert len(searcher._image_embeddings) == num_images

    # Test with non-existent extension
    searcher.index_directory(str(test_images_dir), extensions=[".xyz"])
    assert len(searcher._image_embeddings) == 0


def test_search(mock_clip, test_images_dir):
    """Test image search functionality."""
    searcher = SemanticSearcher(device="cpu")

    # Index directory
    searcher.index_directory(str(test_images_dir))

    # Test basic search
    results = searcher.search("a red image", top_k=2)
    assert len(results) == 2
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    assert all(isinstance(score, float) for _, score in results)
    assert all(score >= 0.0 for _, score in results)  # Scores should be non-negative

    # Test with threshold
    results = searcher.search("a red image", top_k=2, threshold=0.5)
    assert len(results) > 0  # Should return at least one result
    assert all(score >= 0.5 for _, score in results)

    # Test with empty index
    searcher._image_embeddings = {}
    with pytest.raises(ValueError):
        searcher.search("a red image")


def test_error_handling(mock_clip, test_images_dir):
    """Test error handling."""
    searcher = SemanticSearcher(device="cpu")

    # Test non-existent image
    with pytest.raises(FileNotFoundError):
        searcher._get_image_embedding("non_existent.jpg")

    # Test invalid directory
    with pytest.raises(FileNotFoundError):
        searcher.index_directory("non_existent_dir")

    # Test search without indexing
    with pytest.raises(ValueError):
        searcher.search("test query")


def test_device_handling(mock_clip):
    """Test device handling."""
    # Test CPU
    searcher = SemanticSearcher(device="cpu")
    assert searcher.device == "cpu"
    assert next(searcher.model.parameters()).device.type == "cpu"

    # Test CUDA if available
    if torch.cuda.is_available():
        # Create mock parameter with CUDA device
        mock_param = MagicMock()
        mock_param.device.type = "cuda"
        searcher.model.parameters.return_value = iter([mock_param])

        searcher = SemanticSearcher(device="cuda")
        assert searcher.device == "cuda"
        assert next(searcher.model.parameters()).device.type == "cuda"
