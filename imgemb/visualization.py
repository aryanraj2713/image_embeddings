"""Visualization functions for image embeddings."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
import base64
from typing import List, Tuple, Optional
import os


def plot_similar_images(
    query_image_path: str,
    similar_images: List[Tuple[str, float]],
    title: Optional[str] = None,
) -> go.Figure:
    """Plot query image and its similar images with similarity scores.

    Args:
        query_image_path: Path to the query image
        similar_images: List of tuples containing (image_path, similarity_score)
        title: Optional title for the plot

    Returns:
        A plotly Figure object
    """
    # Create subplots
    n_images = len(similar_images) + 1
    fig = make_subplots(rows=1, cols=n_images)

    # Add query image
    query_img = cv2.imread(query_image_path)
    if query_img is None:
        raise ValueError(f"Could not load query image at {query_image_path}")
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode(".jpg", query_img)
    img_base64 = base64.b64encode(buffer).decode()

    fig.add_trace(go.Image(source=f"data:image/jpeg;base64,{img_base64}"), row=1, col=1)

    # Add similar images
    for i, (img_path, score) in enumerate(similar_images, start=2):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image at {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode(".jpg", img)
        img_base64 = base64.b64encode(buffer).decode()

        fig.add_trace(
            go.Image(source=f"data:image/jpeg;base64,{img_base64}"), row=1, col=i
        )

    # Update layout
    fig.update_layout(
        title=title or "Similar Images",
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    # Add similarity scores as annotations
    for i, (_, score) in enumerate(similar_images, start=1):
        fig.add_annotation(
            x=i, y=1.1, text=f"Score: {score:.3f}", showarrow=False, yref="paper"
        )

    return fig
