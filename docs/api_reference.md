# API Reference

## ImageEmbedder

The main class for generating image embeddings.

### Constructor

```python
ImageEmbedder(
    method: str = 'average_color',
    grid_size: Tuple[int, int] = (4, 4),
    normalize: bool = True
)
```

#### Parameters

- `method` (str, optional):
  - Available methods: 'average_color', 'grid', 'edge'
  - Default: 'average_color'

- `grid_size` (Tuple[int, int], optional):
  - Grid dimensions for grid-based embedding
  - Only used when method='grid'
  - Default: (4, 4)

- `normalize` (bool, optional):
  - Whether to normalize embeddings to unit length
  - Default: True

### Methods

#### embed_image

```python
embed_image(image_path: str) -> np.ndarray
```

Generate embedding from an image file.

##### Parameters
- `image_path` (str): Path to the image file

##### Returns
- `np.ndarray`: Image embedding

##### Raises
- `ValueError`: If image cannot be read or path is invalid

#### embed

```python
embed(image: np.ndarray) -> np.ndarray
```

Generate embedding from an image array.

##### Parameters
- `image` (np.ndarray): Input image in BGR format

##### Returns
- `np.ndarray`: Image embedding

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