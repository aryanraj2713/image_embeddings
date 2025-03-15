# imgemb: Image Embeddings and Semantic Search Library

[![CI](https://github.com/aryanraj2713/image_embeddings/actions/workflows/ci.yml/badge.svg)](https://github.com/aryanraj2713/image_embeddings/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/imgemb.svg)](https://badge.fury.io/py/imgemb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[!]

## Overview

`imgemb` is a Python library designed for generating image embeddings and performing efficient similarity search operations. It provides multiple embedding methods ranging from simple color-based approaches to advanced semantic understanding using CLIP models.

## Installation

### Basic Installation
```bash
pip install imgemb
```

### Development Installation
```bash
git clone https://github.com/aryanraj2713/image_embeddings
cd image_embeddings
pip install -e ".[dev]"
```

## Core Components

### 1. Embedding Generation

The library supports four distinct embedding methods:

#### a. Average Color Method
- Fastest method for basic color-based similarity
- Generates a compact 3-dimensional embedding (RGB values)
- Suitable for simple color-matching tasks
```python
from imgemb import ImageEmbedder

embedder = ImageEmbedder(method="average")
embedding = embedder.generate_embedding("path/to/image.jpg")
```

#### b. Grid Method
- Captures spatial color distribution
- Configurable grid size (default: 4x4)
- Returns flattened grid of RGB values
```python
embedder = ImageEmbedder(method="grid", grid_size=(4, 4))
embedding = embedder.generate_embedding("path/to/image.jpg")
```

#### c. Edge Detection Method
- Focuses on image structure and shapes
- Uses Canny edge detection
- Returns edge intensity histogram
```python
embedder = ImageEmbedder(method="edge")
embedding = embedder.generate_embedding("path/to/image.jpg")
```

#### d. Semantic Method (CLIP-based)
- Provides high-level semantic understanding
- Uses OpenAI's CLIP model
- Returns 512-dimensional embedding
```python
embedder = ImageEmbedder(method="semantic")
embedding = embedder.generate_embedding("path/to/image.jpg")
```

### 2. Similarity Search

#### Basic Search
```python
from imgemb import ImageEmbedder

# Initialize embedder
embedder = ImageEmbedder(method="semantic")

# Find similar images
results = embedder.find_similar_images(
    query_image="query.jpg",
    image_directory="path/to/images/",
    top_k=5,
    distance_metric="cosine"
)

# Results format: List[Tuple[str, float]]
# [(image_path, similarity_score), ...]
```

#### Batch Processing
```python
# Process multiple queries efficiently
query_images = ["query1.jpg", "query2.jpg", "query3.jpg"]
batch_results = embedder.batch_find_similar_images(
    query_images=query_images,
    image_directory="path/to/images/",
    top_k=5
)
```

### 3. Visualization

#### Basic Visualization
```python
from imgemb import plot_similar_images

# Create interactive plot
fig = plot_similar_images(
    query_image="query.jpg",
    similar_images=results,
    plot_title="Similar Images"
)

# Display in browser
fig.show()

# Save as HTML
fig.write_html("similar_images.html")
```

#### Customized Visualization
```python
fig = plot_similar_images(
    query_image="query.jpg",
    similar_images=results,
    plot_title="Custom Visualization",
    width=1200,
    height=800,
    images_per_row=3
)
```

### 4. Command Line Interface

The library provides a comprehensive CLI for common operations:

```bash
# Generate embeddings for a directory of images
imgemb generate images/ --output embeddings.json --method semantic

# Find similar images
imgemb find-similar query.jpg images/ \
    --method semantic \
    --top-k 5 \
    --metric cosine \
    --output results.json

# Semantic text-to-image search
imgemb search "a photo of a mountain" images/ \
    --top-k 5 \
    --output results.json
```

## Performance Considerations

1. **Memory Usage**
   - Semantic embeddings: ~2KB per image
   - Grid embeddings: Variable (depends on grid size)
   - Edge embeddings: ~1KB per image
   - Average color: 12 bytes per image

2. **Processing Speed** (approximate, on CPU)
   - Average color: ~0.01s per image
   - Grid method: ~0.05s per image
   - Edge detection: ~0.1s per image
   - Semantic (CLIP): ~0.5s per image

3. **GPU Acceleration**
   - Automatic GPU usage for semantic embeddings if available
   - Significant speed improvement (5-10x) for semantic method

## Error Handling

```python
from imgemb.exceptions import EmbeddingError, InvalidMethodError

try:
    embedder = ImageEmbedder(method="semantic")
    embedding = embedder.generate_embedding("image.jpg")
except InvalidMethodError:
    print("Invalid embedding method specified")
except EmbeddingError as e:
    print(f"Error generating embedding: {e}")
```

## Advanced Usage

### Custom Distance Metrics
```python
from imgemb import ImageEmbedder
import numpy as np

def custom_distance(embedding1, embedding2):
    return np.sum(np.abs(embedding1 - embedding2))

embedder = ImageEmbedder(method="grid")
results = embedder.find_similar_images(
    "query.jpg",
    "images/",
    distance_metric=custom_distance
)
```

### Embedding Persistence
```python
# Save embeddings
embedder.save_embeddings("embeddings.json")

# Load embeddings
embedder.load_embeddings("embeddings.json")
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=imgemb

# Generate coverage report
pytest --cov=imgemb --cov-report=html
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@software{imgemb2024,
  author = {Aryan Raj},
  title = {imgemb: Efficient Image Embeddings and Similarity Search},
  year = {2025},
  publisher = {Aryan Raj},
  url = {https://github.com/aryanraj2713/image_embeddings}
}
```
