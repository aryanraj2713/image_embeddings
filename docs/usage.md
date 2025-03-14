# Image Embeddings Usage Guide

This guide provides detailed information about using the Image Embeddings library for generating numerical representations of images.

## Installation

```bash
pip install image-embeddings
```

## Basic Usage

The library provides a simple interface for generating image embeddings:

```python
from image_embeddings import ImageEmbedder

# Create an embedder instance
embedder = ImageEmbedder()

# Generate embedding from an image file
embedding = embedder.embed_image("path/to/image.jpg")

# Or generate embedding from a numpy array
import cv2
image = cv2.imread("path/to/image.jpg")
embedding = embedder.embed(image)
```

## Embedding Methods

### 1. Average Color Embedding

The simplest method that computes the mean color values across the image:

```python
embedder = ImageEmbedder(method='average_color')
```

- Output shape: `(3,)` representing BGR channels
- Best for: Quick color-based similarity comparisons
- Limitations: Loses spatial information

### 2. Grid-based Embedding

Divides the image into a grid and computes local features:

```python
embedder = ImageEmbedder(
    method='grid',
    grid_size=(4, 4)  # Creates a 4x4 grid
)
```

- Output shape: `(grid_height * grid_width * 3,)`
- Best for: Preserving spatial information
- Customization: Adjust `grid_size` for different granularity

### 3. Edge-based Embedding

Uses edge detection to create a histogram-based embedding:

```python
embedder = ImageEmbedder(method='edge')
```

- Output shape: `(64,)` representing edge intensity histogram
- Best for: Shape and texture analysis
- Features: Invariant to color changes

## Configuration Options

### Normalization

By default, embeddings are normalized to unit length. Disable with:

```python
embedder = ImageEmbedder(normalize=False)
```

### Grid Size

For grid-based embeddings, customize the grid dimensions:

```python
embedder = ImageEmbedder(
    method='grid',
    grid_size=(8, 8)  # More detailed grid
)
```

## Best Practices

1. **Choose the Right Method:**
   - Use `average_color` for simple color matching
   - Use `grid` for spatial awareness
   - Use `edge` for shape matching

2. **Normalization:**
   - Keep normalization enabled for similarity comparisons
   - Disable for absolute value analysis

3. **Grid Size Selection:**
   - Larger grids capture more detail but increase dimensionality
   - Start with 4x4 and adjust based on needs

## Example Applications

### Image Similarity

```python
from image_embeddings import ImageEmbedder
import numpy as np

embedder = ImageEmbedder(method='grid')

# Generate embeddings
emb1 = embedder.embed_image("image1.jpg")
emb2 = embedder.embed_image("image2.jpg")

# Compare using cosine similarity
similarity = np.dot(emb1, emb2)
```

### Batch Processing

```python
import glob
from image_embeddings import ImageEmbedder

embedder = ImageEmbedder()
embeddings = []

# Process multiple images
for image_path in glob.glob("images/*.jpg"):
    embedding = embedder.embed_image(image_path)
    embeddings.append(embedding)
```

## Error Handling

The library includes robust error handling:

```python
try:
    embedding = embedder.embed_image("nonexistent.jpg")
except ValueError as e:
    print(f"Error: {e}")
```

Common errors:
- Invalid image path
- Unsupported image format
- Invalid method name
- Invalid grid size

## Performance Considerations

- Edge detection is more computationally intensive
- Grid-based embedding memory usage increases with grid size
- Consider batch processing for multiple images
- Use appropriate grid size for your image dimensions 