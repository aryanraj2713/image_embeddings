# API Reference

## ImageEmbedder

The main class for generating and comparing image embeddings.

### Constructor

```python
ImageEmbedder(
    method: str = "grid",
    grid_size: Tuple[int, int] = (4, 4),
    normalize: bool = True,
    color_space: str = "rgb"
)
```

**Parameters:**
- `method`: Embedding method to use. Options:
  - `"average_color"`: Simple RGB/HSV color averaging
  - `"grid"`: Grid-based color statistics
  - `"edge"`: Edge-based features
- `grid_size`: Tuple of (rows, cols) for grid method
- `normalize`: Whether to normalize embeddings to unit length
- `color_space`: Color space to use ("rgb" or "hsv")

### Methods

#### embed_image
```python
embed_image(image_path: str) -> np.ndarray
```
Generate embedding for a single image.

**Parameters:**
- `image_path`: Path to the image file

**Returns:**
- Numpy array containing the embedding

#### embed
```python
embed(image: np.ndarray) -> np.ndarray
```
Generate embedding for an image array.

**Parameters:**
- `image`: Image as numpy array (HxWx3)

**Returns:**
- Numpy array containing the embedding

#### compare_images
```python
compare_images(image1_path: str, image2_path: str) -> float
```
Compare two images and return their similarity score.

**Parameters:**
- `image1_path`: Path to first image
- `image2_path`: Path to second image

**Returns:**
- Similarity score between 0 and 1

#### find_similar_images
```python
find_similar_images(
    query_image: str,
    image_dir: str,
    top_k: int = 5
) -> List[Tuple[str, float]]
```
Find similar images to a query image in a directory.

**Parameters:**
- `query_image`: Path to query image
- `image_dir`: Directory containing images to search
- `top_k`: Number of similar images to return

**Returns:**
- List of tuples (image_path, similarity_score)

## SemanticSearcher

Class for semantic image search using OpenCLIP.

### Constructor

```python
SemanticSearcher(
    device: str = "cuda",
    model_name: str = "ViT-B-32"
)
```

**Parameters:**
- `device`: Computing device ("cuda" or "cpu")
- `model_name`: OpenCLIP model variant

### Methods

#### index_directory
```python
index_directory(
    directory: str,
    extensions: List[str] = None
) -> None
```
Index all images in a directory.

**Parameters:**
- `directory`: Directory containing images
- `extensions`: List of file extensions to include

#### search
```python
search(
    query: str,
    top_k: int = 5,
    threshold: float = 0.0
) -> List[Tuple[str, float]]
```
Search for images matching a text query.

**Parameters:**
- `query`: Text query
- `top_k`: Number of results to return
- `threshold`: Minimum similarity score (0 to 1)

**Returns:**
- List of tuples (image_path, similarity_score)

## Visualization

Functions for visualizing embeddings and similar images.

### plot_similar_images

```python
plot_similar_images(
    query_image_path: str,
    similar_images: List[Tuple[str, float]],
    title: Optional[str] = None
) -> plotly.graph_objects.Figure
```

Create an interactive visualization of query image and its similar matches.

**Parameters:**
- `query_image_path`: Path to the query image
- `similar_images`: List of tuples (image_path, similarity_score)
- `title`: Optional title for the plot

**Returns:**
- Plotly Figure object containing the visualization

**Example:**
```python
from imgemb import ImageEmbedder, plot_similar_images

# Find similar images
embedder = ImageEmbedder()
similar_images = embedder.find_similar_images("query.jpg", "images/", top_k=5)

# Create visualization
fig = plot_similar_images("query.jpg", similar_images, title="Similar Images")
fig.show()
```

## Command Line Interface

The package provides a command-line interface for common operations.

### Commands

#### generate
Generate embeddings for images.

```bash
imgemb generate <input> [--output OUTPUT] [--method METHOD] [--grid-size W H]
```

#### compare
Compare two images.

```bash
imgemb compare <image1> <image2> [--method METHOD] [--grid-size W H]
```

#### find-similar
Find similar images in a directory.

```bash
imgemb find-similar <query> <directory> [-k TOP_K] [--method METHOD]
```

#### search
Perform semantic search using text query.

```bash
imgemb search <query> <directory> [-k TOP_K] [--threshold THRESHOLD]
```

## BaseEmbedder

Abstract base class for embedding methods.

### Methods

#### embed

```python
@abstractmethod
embed(image: np.ndarray) -> np.ndarray
```

Generate embedding for the given image.

##### Parameters
- `image` (np.ndarray): Input image in BGR format

##### Returns
- `np.ndarray`: Image embedding

## AverageColorEmbedder

Generates embeddings based on average color values.

### Methods

#### embed

```python
embed(image: np.ndarray) -> np.ndarray
```

Generate embedding by computing average color values.

##### Parameters
- `image` (np.ndarray): Input image in BGR format

##### Returns
- `np.ndarray`: 1D array of average color values (shape: (3,))

## GridEmbedder

Generates embeddings based on grid-wise average colors.

### Constructor

```python
GridEmbedder(grid_size: Tuple[int, int] = (4, 4))
```

#### Parameters
- `grid_size` (Tuple[int, int]): Grid dimensions

### Methods

#### embed

```python
embed(image: np.ndarray) -> np.ndarray
```

Generate embedding by dividing image into grid and computing average colors.

##### Parameters
- `image` (np.ndarray): Input image in BGR format

##### Returns
- `np.ndarray`: Flattened array of grid-wise average colors (shape: (grid_h * grid_w * 3,))

## EdgeEmbedder

Generates embeddings based on edge information.

### Methods

#### embed

```python
embed(image: np.ndarray) -> np.ndarray
```

Generate embedding using edge detection.

##### Parameters
- `image` (np.ndarray): Input image in BGR format

##### Returns
- `np.ndarray`: Edge-based embedding (shape: (64,))

## Return Value Shapes

Different embedding methods produce different output shapes:

| Method | Output Shape | Description |
|--------|--------------|-------------|
| average_color | (3,) | BGR channel means |
| grid | (grid_h * grid_w * 3,) | Grid-wise color features |
| edge | (64,) | Edge intensity histogram |

## Error Types

Common exceptions that may be raised:

- `ValueError`:
  - Invalid method name
  - Invalid image path
  - Invalid image format
  - Invalid grid size dimensions

- `TypeError`:
  - Invalid parameter types
  - Invalid grid size type 