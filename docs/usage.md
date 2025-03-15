# Usage Guide

This guide covers the main features and usage patterns of the `imgemb` library.

## Basic Usage

### Installation

```bash
pip install imgemb
```

### Quick Start

```python
from imgemb import ImageEmbedder, plot_similar_images

# Initialize embedder
embedder = ImageEmbedder(method="grid", grid_size=(4, 4))

# Generate embedding for an image
embedding = embedder.embed_image("path/to/image.jpg")

# Find similar images
similar_images = embedder.find_similar_images(
    "query.jpg",
    "images_directory/",
    top_k=5
)

# Visualize results
fig = plot_similar_images("query.jpg", similar_images)
fig.show()
```

## Embedding Methods

### 1. Average Color Method

Simple but effective for color-based similarity.

```python
embedder = ImageEmbedder(method="average_color")
embedding = embedder.embed_image("image.jpg")
print(f"Embedding shape: {embedding.shape}")  # (3,) for RGB
```

### 2. Grid Method

Divides image into grid cells and computes color statistics.

```python
embedder = ImageEmbedder(
    method="grid",
    grid_size=(4, 4),  # 4x4 grid
    normalize=True,     # Normalize embeddings
    color_space="hsv"  # Use HSV color space
)
embedding = embedder.embed_image("image.jpg")
print(f"Embedding shape: {embedding.shape}")  # (48,) for 4x4 grid with RGB
```

### 3. Edge Method

Extracts edge features using Sobel operators.

```python
embedder = ImageEmbedder(method="edge")
embedding = embedder.embed_image("image.jpg")
print(f"Embedding shape: {embedding.shape}")  # (32,) for edge histogram
```

## Finding Similar Images

### Basic Similarity Search

```python
# Initialize embedder
embedder = ImageEmbedder(method="grid")

# Find similar images
similar_images = embedder.find_similar_images(
    "query.jpg",
    "image_directory/",
    top_k=5
)

# Print results
for path, score in similar_images:
    print(f"{path}: {score:.3f}")

# Visualize results
fig = plot_similar_images(
    "query.jpg",
    similar_images,
    title="Similar Images"
)
fig.show()
```

### Semantic Search

```python
from imgemb import SemanticSearcher

# Initialize searcher
searcher = SemanticSearcher(device="cuda")  # Use GPU if available

# Index directory
searcher.index_directory("image_directory/")

# Search with text query
results = searcher.search(
    "a photo of a dog",
    top_k=5,
    threshold=0.7  # Minimum similarity score
)

# Visualize results
fig = plot_similar_images(
    results[0][0],     # First result as query
    results[1:],       # Remaining results
    title="Semantic Search Results"
)
fig.show()
```

## Command Line Interface

### Generate Embeddings

```bash
# Generate embeddings for a single image
imgemb generate input.jpg --output embeddings.json --method grid

# Generate embeddings for a directory
imgemb generate images/ --output embeddings.json --method edge
```

### Compare Images

```bash
# Compare two images
imgemb compare image1.jpg image2.jpg --method average_color

# Compare with custom grid size
imgemb compare image1.jpg image2.jpg --method grid --grid-size 8 8
```

### Find Similar Images

```bash
# Find similar images in a directory
imgemb find-similar query.jpg images/ -k 5 --method grid

# Find similar images with edge features
imgemb find-similar query.jpg images/ -k 10 --method edge
```

### Semantic Search

```bash
# Search images using text query
imgemb search "a red car" images/ -k 5

# Search with similarity threshold
imgemb search "landscape photo" images/ -k 10 --threshold 0.7
```

## Visualization

### Plotting Similar Images

```python
from imgemb import plot_similar_images

# Create visualization
fig = plot_similar_images(
    query_image_path="query.jpg",
    similar_images=[
        ("similar1.jpg", 0.95),
        ("similar2.jpg", 0.85),
        ("similar3.jpg", 0.75)
    ],
    title="Similar Images"
)

# Show interactive plot
fig.show()

# Save plot to HTML
fig.write_html("similar_images.html")
```

## Best Practices

1. **Choose the Right Method**:
   - `average_color`: Quick color-based comparison
   - `grid`: Better spatial awareness
   - `edge`: Focus on shape features
   - Semantic search: Understanding image content

2. **Optimization Tips**:
   - Use appropriate grid sizes (4x4 or 8x8 work well)
   - Enable normalization for better comparisons
   - Consider HSV color space for color-focused tasks

3. **Performance Considerations**:
   - Index directories once and reuse embeddings
   - Use GPU for semantic search when available
   - Batch process images when possible

4. **Visualization Tips**:
   - Keep number of displayed images reasonable (5-10)
   - Use descriptive titles
   - Save interactive plots for sharing

## Error Handling

```python
try:
    embedder = ImageEmbedder(method="grid")
    embedding = embedder.embed_image("image.jpg")
except ValueError as e:
    print(f"Invalid configuration: {e}")
except FileNotFoundError as e:
    print(f"Image not found: {e}")
except Exception as e:
    print(f"Unexpected error: {e}") 