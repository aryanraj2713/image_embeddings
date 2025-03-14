# API Reference

## ImageEmbedder Class

The main class for generating and comparing image embeddings.

### Constructor

```python
ImageEmbedder(
    method: str = 'grid',
    grid_size: Tuple[int, int] = (4, 4),
    normalize: bool = True,
    color_space: str = 'rgb'
)
```

#### Parameters

- `method` (str): The embedding method to use. Options:
  - `'average_color'`: Simple RGB color averaging
  - `'grid'`: Grid-based color features
  - `'edge'`: Edge-based features using Sobel
- `grid_size` (tuple): Grid dimensions for the grid method (rows, cols)
- `normalize` (bool): Whether to normalize embeddings to unit length
- `color_space` (str): Color space to use ('rgb' or 'hsv')

### Methods

#### embed_image

```python
def embed_image(self, image_path: str) -> np.ndarray:
    """Generate embedding for a single image."""
```

**Parameters:**
- `image_path` (str): Path to the image file

**Returns:**
- `np.ndarray`: Image embedding vector

**Example:**
```python
embedder = ImageEmbedder(method='grid')
embedding = embedder.embed_image('path/to/image.jpg')
print(f"Embedding shape: {embedding.shape}")
```

#### compare_images

```python
def compare_images(
    self,
    image1_path: str,
    image2_path: str,
    metric: str = 'cosine'
) -> float:
    """Compare similarity between two images."""
```

**Parameters:**
- `image1_path` (str): Path to first image
- `image2_path` (str): Path to second image
- `metric` (str): Similarity metric ('cosine' or 'euclidean')

**Returns:**
- `float`: Similarity score (higher is more similar for cosine, lower for euclidean)

**Example:**
```python
similarity = embedder.compare_images('image1.jpg', 'image2.jpg', metric='cosine')
print(f"Similarity score: {similarity:.3f}")
```

#### find_similar_images

```python
def find_similar_images(
    self,
    query_path: str,
    image_dir: str,
    top_k: int = 5,
    metric: str = 'cosine'
) -> List[Tuple[str, float]]:
    """Find most similar images to a query image in a directory."""
```

**Parameters:**
- `query_path` (str): Path to query image
- `image_dir` (str): Directory containing images to search
- `top_k` (int): Number of similar images to return
- `metric` (str): Similarity metric ('cosine' or 'euclidean')

**Returns:**
- `List[Tuple[str, float]]`: List of (image_path, similarity_score) pairs

**Example:**
```python
similar_images = embedder.find_similar_images(
    'query.jpg',
    'image/directory/',
    top_k=5
)
for path, score in similar_images:
    print(f"{path}: {score:.3f}")
```

## Command Line Interface

### Compare Images

```bash
imgemb compare [OPTIONS] IMAGE1 IMAGE2
```

**Options:**
- `--method TEXT`: Embedding method [default: grid]
- `--grid-size INT INT`: Grid dimensions [default: 4 4]
- `--metric TEXT`: Similarity metric [default: cosine]
- `--normalize`: Normalize embeddings [default: True]

### Generate Embeddings

```bash
imgemb generate [OPTIONS] INPUT_DIR
```

**Options:**
- `--output TEXT`: Output JSON file [required]
- `--method TEXT`: Embedding method [default: grid]
- `--grid-size INT INT`: Grid dimensions [default: 4 4]
- `--normalize`: Normalize embeddings [default: True]

### Find Similar Images

```bash
imgemb find-similar [OPTIONS] QUERY_IMAGE IMAGE_DIR
```

**Options:**
- `-k, --top-k INT`: Number of similar images [default: 5]
- `--method TEXT`: Embedding method [default: grid]
- `--grid-size INT INT`: Grid dimensions [default: 4 4]
- `--metric TEXT`: Similarity metric [default: cosine]
- `--normalize`: Normalize embeddings [default: True]

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