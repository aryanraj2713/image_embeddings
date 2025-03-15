# imgemb: Efficient Image Embeddings and Similarity Search

[![PyPI version](https://badge.fury.io/py/imgemb.svg)](https://badge.fury.io/py/imgemb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

`imgemb` is a powerful Python library for generating image embeddings and performing similarity search. It offers multiple embedding methods, semantic search capabilities, and interactive visualizations.

## Features

- 🎨 **Multiple Embedding Methods**:
  - Average Color: Fast color-based similarity
  - Grid: Spatial color distribution analysis
  - Edge: Shape and structure detection
  - Semantic: CLIP-based content understanding

- 🔍 **Similarity Search**:
  - Fast nearest neighbor search
  - Multiple distance metrics (cosine, euclidean)
  - Batch processing support
  - Interactive result visualization

- 🖼️ **Interactive Visualization**:
  - Plot similar images with scores
  - Save interactive HTML plots
  - Customizable layouts and titles

- 🛠️ **Command Line Interface**:
  - Generate embeddings
  - Compare images
  - Find similar images
  - Semantic search

- 🚀 **Performance**:
  - Efficient numpy-based computations
  - GPU support for semantic search
  - Optimized for large image collections

## Installation

```bash
pip install imgemb
```

## Quick Start

```python
from imgemb import ImageEmbedder, plot_similar_images

# Initialize embedder
embedder = ImageEmbedder(method="grid")

# Find similar images
similar_images = embedder.find_similar_images(
    "query.jpg",
    "images/",
    top_k=5
)

# Visualize results
fig = plot_similar_images("query.jpg", similar_images)
fig.show()
```

## Command Line Usage

```bash
# Generate embeddings
imgemb generate images/ --output embeddings.json --method grid

# Find similar images
imgemb find-similar query.jpg images/ -k 5 --method edge

# Semantic search
imgemb search "a photo of a dog" images/ -k 5
```

## Documentation

- [Usage Guide](docs/usage.md): Detailed examples and best practices
- [API Reference](docs/api_reference.md): Complete API documentation
- [Examples](examples/): Sample scripts and notebooks

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use `imgemb` in your research, please cite:

```bibtex
@software{imgemb2024,
  author = {Aryan Raj},
  title = {imgemb: Efficient Image Embeddings and Similarity Search},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/aryanraj2713/imgemb}
}
```
