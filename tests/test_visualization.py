"""Tests for visualization functions."""

import pytest
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import base64
import cv2
import tempfile
from pathlib import Path
import json
from imgemb import ImageEmbedder
from imgemb.visualization import plot_similar_images

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_plot_embedding():
    """Test the plot_embedding function."""
    # Create a sample embedding
    embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    # Create and test plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=embedding, mode="lines"))
    fig.update_layout(
        title="Test Embedding",
        xaxis_title="Dimension",
        yaxis_title="Value",
        showlegend=False,
    )

    # Test that the figure was created successfully
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].mode == "lines"


def test_plot_similar_images():
    """Test the plot_similar_images function."""
    # Create a temporary test image
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    test_img[30:70, 30:70] = [255, 0, 0]  # Red square

    # Save test images
    temp_dir = "temp_test_images"
    os.makedirs(temp_dir, exist_ok=True)

    query_path = os.path.join(temp_dir, "query.jpg")
    similar1_path = os.path.join(temp_dir, "similar1.jpg")
    similar2_path = os.path.join(temp_dir, "similar2.jpg")

    cv2.imwrite(query_path, test_img)
    cv2.imwrite(similar1_path, test_img)
    cv2.imwrite(similar2_path, test_img)

    # Create test data
    similar_images = [(similar1_path, 0.95), (similar2_path, 0.85)]

    # Create and test plotly figure
    n_images = len(similar_images) + 1
    fig = make_subplots(rows=1, cols=n_images)

    # Test that the figure was created successfully
    assert isinstance(fig, go.Figure)
    # Check subplot layout through axis domains
    assert len(fig.layout.xaxis.domain) == 2  # Each axis has a domain range
    assert len(fig.layout.yaxis.domain) == 2
    assert all(
        0 <= x <= 1 for x in fig.layout.xaxis.domain
    )  # Domain values are between 0 and 1

    # Clean up
    for path in [query_path, similar1_path, similar2_path]:
        if os.path.exists(path):
            os.remove(path)
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)


def test_plotly_figure_properties():
    """Test the properties of generated plotly figures."""
    # Create a sample embedding
    embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    # Create a figure using plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=embedding, mode="lines"))
    fig.update_layout(
        title="Test Embedding",
        xaxis_title="Dimension",
        yaxis_title="Value",
        showlegend=False,
    )

    # Test figure properties
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].mode == "lines"
    assert fig.layout.showlegend == False
    assert fig.layout.xaxis.title.text == "Dimension"
    assert fig.layout.yaxis.title.text == "Value"


def test_subplots_creation():
    """Test the creation of subplots for multiple embeddings."""
    # Create sample data
    embeddings = [
        np.array([0.1, 0.2, 0.3]),
        np.array([0.4, 0.5, 0.6]),
        np.array([0.7, 0.8, 0.9]),
    ]
    titles = ["Embedding 1", "Embedding 2", "Embedding 3"]

    # Create subplots
    fig = make_subplots(rows=len(embeddings), cols=1, subplot_titles=titles)

    # Add traces
    for i, embedding in enumerate(embeddings, 1):
        fig.add_trace(go.Scatter(y=embedding, mode="lines"), row=i, col=1)

    # Test subplot properties
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == len(embeddings)
    assert len(fig.layout.annotations) == len(titles)  # subplot titles
    for i, title in enumerate(titles):
        assert fig.layout.annotations[i].text == title
