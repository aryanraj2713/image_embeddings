# Semantic Image Search

The `imgemb` library now includes powerful semantic image search capabilities using OpenAI's CLIP model. This feature allows you to search through your image collection using natural language queries.

## Overview

The semantic search functionality is powered by the CLIP (Contrastive Language-Image Pre-training) model, which enables:
- Natural language queries for image search
- Zero-shot image classification
- High-quality image embeddings that capture semantic meaning
- Cross-modal understanding between text and images

## Installation

The semantic search functionality is included in the main package:

```bash
pip install imgemb
```

## Basic Usage

Here's a simple example of how to use semantic search:

```python
from imgemb import SemanticSearcher

# Initialize the searcher
searcher = SemanticSearcher()

# Index a directory of images
searcher.index_directory("path/to/images")

# Search for images
results = searcher.search("a photo of a dog", top_k=5)

# Print results
for path, score in results:
    print(f"{path}: {score:.3f}")
```

## Command Line Interface

You can also use semantic search from the command line:

```bash
# Search for images in a directory
imgemb search path/to/images "a photo of a dog" -k 5

# Search with minimum similarity threshold
imgemb search path/to/images "a red car" -k 5 --threshold 0.25
```

## Advanced Usage

### Initialization Options

```python
searcher = SemanticSearcher(
    model_name="ViT-B/32",  # Model variant (ViT-B/32 or ViT-L/14)
    device="cuda"  # Use GPU if available
)
```

### Indexing Options

```python
searcher.index_directory(
    "path/to/images",
    extensions=[".jpg", ".jpeg", ".png"]  # Specify file types
)
```

### Search Options

```python
results = searcher.search(
    query="a person wearing a red shirt",
    top_k=5,  # Number of results
    threshold=0.25  # Minimum similarity score
)
```

## Query Examples

The semantic search supports a wide range of natural language queries:

- Object descriptions: "a photo of a dog"
- Actions: "people dancing"
- Colors: "a red car"
- Scenes: "sunset over mountains"
- Emotions: "happy people"
- Abstract concepts: "freedom"
- Compositional queries: "a person wearing a red shirt near a blue car"

## Performance Tips

1. **GPU Acceleration**: Use a CUDA-enabled GPU for faster processing
2. **Batch Processing**: Index directories in batches for large collections
3. **Memory Management**: Clear the index when switching between different image sets
4. **Query Optimization**: Use specific and clear queries for better results

## Technical Details

### Model Architecture

The semantic search uses CLIP's vision and text encoders:
- Vision Encoder: Processes images into embeddings
- Text Encoder: Converts queries into compatible embeddings
- Similarity: Computed using cosine similarity between embeddings

### Embedding Dimensions

- ViT-B/32: 512-dimensional embeddings
- ViT-L/14: 768-dimensional embeddings

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- Other formats supported by PIL

## Error Handling

Common errors and solutions:

1. **CUDA Out of Memory**
   - Solution: Use a smaller batch size or switch to CPU
   
2. **File Not Found**
   - Solution: Check file paths and permissions

3. **Invalid Image Format**
   - Solution: Ensure images are in supported formats

## Examples

See the `examples/semantic_search.py` file for complete examples, including:
- Basic search functionality
- Advanced usage patterns
- Visualization of results
- Error handling

## Contributing

To contribute to the semantic search functionality:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

The semantic search functionality is part of the `imgemb` package and is licensed under the MIT License. 