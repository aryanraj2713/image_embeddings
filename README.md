# Imgemb

[![CI](https://github.com/aryanraj2713/image_embeddings/actions/workflows/ci.yml/badge.svg)](https://github.com/aryanraj2713/image_embeddings/actions/workflows/ci.yml)
[pypi.org](https://pypi.org/project/imgemb/)

A lightweight Python library for generating and comparing image embeddings using various methods. This library provides tools for image similarity search, clustering, and comparison.

## Features

- Multiple embedding methods:
  - **Average Color**: Simple RGB color averaging
  - **Grid-based**: Divides image into grid cells and computes color features
  - **Edge-based**: Uses Sobel edge detection and histogram features
  - **CLIP-based**: Semantic embeddings for natural language search
- Command-line interface (CLI) for easy usage
- Normalization options for embeddings
- Tools for finding similar images in a directory
- Support for batch processing
- **Semantic Search**:
  - Natural language queries for image search
  - Zero-shot image classification
  - Cross-modal understanding between text and images
  - GPU acceleration support

## Installation

### From PyPI (Recommended)

```bash
pip install imgemb
```

### From Source

```bash
git clone https://github.com/aryanraj2713/image_embeddings.git
cd image_embeddings
pip install -e ".[dev]"  # Install with development dependencies
```

## Quick Start

### Using as a Python Library

```python
from imgemb import ImageEmbedder

# Initialize embedder
embedder = ImageEmbedder(
    method='grid',           # 'average_color', 'grid', or 'edge'
    grid_size=(4, 4),       # For grid method
    normalize=True          # Whether to normalize embeddings
)

# Generate embedding for a single image
embedding = embedder.embed_image('path/to/image.jpg')

# Compare two images
similarity = embedder.compare_images('image1.jpg', 'image2.jpg')

# Find similar images in a directory
similar_images = embedder.find_similar_images(
    'query.jpg',
    'path/to/image/directory',
    top_k=5
)
```

### Semantic Search

```python
from imgemb import SemanticSearcher

# Initialize searcher
searcher = SemanticSearcher()

# Index a directory of images
searcher.index_directory("path/to/images")

# Search using natural language
results = searcher.search("a photo of a dog playing in the park", top_k=5)

# Print results
for path, score in results:
    print(f"{path}: {score:.3f}")
```

### Using the Command Line Interface

1. Compare two images:
```bash
imgemb compare image1.jpg image2.jpg --method grid --grid-size 4 4
```

2. Generate embeddings for images:
```bash
imgemb generate path/to/images/ --output embeddings.json --method edge
```

3. Find similar images:
```bash
imgemb find-similar query.jpg image/directory/ -k 5 --method grid
```

## Embedding Methods

### Average Color
Computes the mean RGB values of the entire image. Simple but effective for basic color-based similarity.

```python
embedder = ImageEmbedder(method='average_color')
```

### Grid-based
Divides the image into a grid and computes mean RGB values for each cell. Better for capturing spatial color distribution.

```python
embedder = ImageEmbedder(method='grid', grid_size=(4, 4))
```

### Edge-based
Uses Sobel edge detection and histogram features. Good for capturing structural similarities.

```python
embedder = ImageEmbedder(method='edge')
```

## Development

### Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/ -v
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Requirements

- Python ≥ 3.8
- OpenCV (opencv-python)
- NumPy
- Matplotlib
- scikit-learn

## Citation

If you use this library in your research, please cite:

```bibtex
@software{image_embeddings,
  title = {Image Embeddings: A Lightweight Library for Image Similarity},
  author = {Raj, Aryan},
  year = {2024},
  url = {https://github.com/aryanraj2713/image_embeddings}
}
```
