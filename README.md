# Imgemb

[![PyPI version](https://badge.fury.io/py/imgemb.svg)](https://badge.fury.io/py/imgemb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight Python library for generating and comparing image embeddings, featuring semantic search capabilities powered by OpenCLIP.

## Features

- **Multiple Embedding Methods**:
  - Average Color: Simple RGB/HSV color averaging
  - Grid-based: Divide image into grid cells and compute color statistics
  - Edge-based: Extract edge features using Sobel operators
  - Semantic Search: Utilize OpenCLIP for advanced image understanding

- **Command Line Interface (CLI)**: Easy-to-use commands for:
  - Generating embeddings
  - Comparing images
  - Finding similar images
  - Batch processing directories

- **Python API**: Flexible and intuitive API for integration into your projects

- **Efficient Processing**: Optimized for both CPU and GPU environments

## Installation

### From PyPI
```bash
pip install imgemb
```

### From Source
```bash
git clone https://github.com/aryanraj2713/image_embeddings.git
cd image_embeddings
pip install -e .
```

## Quick Start

### Command Line Usage

1. Generate embeddings for an image:
```bash
imgemb generate input.jpg --method grid --grid-size 3 3
```

2. Compare two images:
```bash
imgemb compare image1.jpg image2.jpg --method average
```

3. Find similar images in a directory:
```bash
imgemb find-similar query.jpg images_dir/ --top-k 5
```

### Python API

#### Basic Usage
```python
from imgemb import ImageEmbedder, SemanticSearcher

# Generate embeddings using different methods
embedder = ImageEmbedder(method="grid", grid_size=(3, 3))
embedding = embedder.embed_image("path/to/image.jpg")

# Semantic search
searcher = SemanticSearcher()
searcher.index_directory("path/to/image/directory")
results = searcher.search("a photo of a dog", top_k=5)
```

#### Advanced Usage
```python
# Custom grid size with normalization
embedder = ImageEmbedder(
    method="grid",
    grid_size=(4, 4),
    normalize=True,
    color_space="hsv"
)

# Batch processing with edge detection
embedder = ImageEmbedder(method="edge")
embeddings = embedder.embed_directory(
    "path/to/directory",
    extensions=[".jpg", ".png"]
)

# Semantic search with threshold
searcher = SemanticSearcher(device="cuda")  # Use GPU if available
searcher.index_directory("image_database")
results = searcher.search(
    "abstract art",
    top_k=10,
    threshold=0.7  # Only return results with similarity > 0.7
)
```

## API Documentation

### ImageEmbedder

#### Methods
- `__init__(method="average", grid_size=None, normalize=False, color_space="rgb")`
  - `method`: Embedding method ("average", "grid", "edge")
  - `grid_size`: Tuple of (rows, cols) for grid method
  - `normalize`: Whether to normalize embeddings
  - `color_space`: Color space to use ("rgb" or "hsv")

- `embed_image(image_path: str) -> np.ndarray`
  - Generates embedding for a single image
  - Returns numpy array of embedding values

- `embed_directory(directory: str, extensions: List[str] = None) -> Dict[str, np.ndarray]`
  - Generates embeddings for all images in directory
  - Returns dictionary mapping file paths to embeddings

### SemanticSearcher

#### Methods
- `__init__(device="cuda", model_name="ViT-B-32")`
  - `device`: Computing device ("cuda" or "cpu")
  - `model_name`: OpenCLIP model variant

- `index_directory(directory: str, extensions: List[str] = None)`
  - Indexes all images in specified directory
  - Optional file extension filtering

- `search(query: str, top_k: int = 5, threshold: float = 0.0) -> List[Tuple[str, float]]`
  - Searches for images matching text query
  - Returns list of (image_path, similarity_score) tuples

## Development

### Setup Development Environment
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black image_embeddings/ tests/

# Run linter
flake8 image_embeddings/ tests/
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=image_embeddings

# Generate HTML coverage report
pytest --cov=image_embeddings --cov-report=html
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.5+
- NumPy 1.19+
- Pillow 9.0+
- open-clip-torch 2.20+

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{imgemb,
  author = {Aryan Raj},
  title = {Imgemb: A Lightweight Python Library for Image Embeddings},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/aryanraj2713/image_embeddings}
}
```
