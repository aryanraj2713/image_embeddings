# Usage Guide

## Installation

### From PyPI

```bash
pip install imgemb
```

### From Source

```bash
git clone https://github.com/aryanraj2713/image_embeddings.git
cd image_embeddings
pip install -e ".[dev]"  # Install with development dependencies
```

## Basic Usage

### 1. Generating Image Embeddings

```python
from imgemb import ImageEmbedder

# Initialize embedder with desired method
embedder = ImageEmbedder(
    method='grid',           # Choose from: 'average_color', 'grid', 'edge'
    grid_size=(4, 4),       # For grid method
    normalize=True,         # Normalize embeddings to unit length
    color_space='rgb'      # Color space for feature extraction
)

# Generate embedding for a single image
embedding = embedder.embed_image('path/to/image.jpg')
print(f"Embedding shape: {embedding.shape}")
```

### 2. Comparing Images

```python
# Compare two images directly
similarity = embedder.compare_images(
    'image1.jpg',
    'image2.jpg',
    metric='cosine'  # or 'euclidean'
)
print(f"Similarity score: {similarity:.3f}")

# Find similar images in a directory
similar_images = embedder.find_similar_images(
    'query.jpg',
    'image/directory/',
    top_k=5,
    metric='cosine'
)

# Print results
for path, score in similar_images:
    print(f"{path}: {score:.3f}")
```

## Advanced Usage

### 1. Batch Processing

```python
import glob
import numpy as np

# Process multiple images
image_paths = glob.glob('images/*.jpg')
embeddings = []

for path in image_paths:
    embedding = embedder.embed_image(path)
    embeddings.append(embedding)

# Convert to numpy array for efficient processing
embeddings = np.array(embeddings)
```

### 2. Image Clustering

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
scaled_embeddings = scaler.fit_transform(embeddings)

# Perform clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(scaled_embeddings)

# Analyze clusters
for cluster in range(n_clusters):
    cluster_size = np.sum(clusters == cluster)
    print(f"\nCluster {cluster}:")
    print(f"Size: {cluster_size} images")
```

### 3. Custom Similarity Metrics

```python
def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Custom similarity metric using Manhattan distance."""
    return np.sum(np.abs(a - b))

# Use custom metric for comparison
def compare_custom(image1_path: str, image2_path: str) -> float:
    emb1 = embedder.embed_image(image1_path)
    emb2 = embedder.embed_image(image2_path)
    return manhattan_distance(emb1, emb2)
```

## Command Line Interface

### 1. Compare Two Images

```bash
# Basic comparison
imgemb compare image1.jpg image2.jpg

# Advanced options
imgemb compare image1.jpg image2.jpg \
    --method grid \
    --grid-size 4 4 \
    --metric cosine \
    --normalize
```

### 2. Generate Embeddings

```bash
# Generate embeddings for a directory of images
imgemb generate images/ \
    --output embeddings.json \
    --method edge \
    --normalize
```

### 3. Find Similar Images

```bash
# Find similar images with custom parameters
imgemb find-similar query.jpg images/ \
    -k 5 \
    --method grid \
    --grid-size 8 8 \
    --metric cosine
```

## Best Practices

1. **Choosing Embedding Methods:**
   - `average_color`: Best for simple color-based similarity
   - `grid`: Good balance between detail and efficiency
   - `edge`: Best for structural similarity

2. **Performance Optimization:**
   - Pre-compute embeddings for large datasets
   - Use batch processing for multiple images
   - Consider dimensionality reduction for large embeddings

3. **Memory Management:**
   - Process images in batches for large datasets
   - Clear unused embeddings from memory
   - Use memory-mapped files for large datasets

4. **Error Handling:**
```python
try:
    embedding = embedder.embed_image(image_path)
except Exception as e:
    print(f"Error processing {image_path}: {e}")
    # Handle error appropriately
```

5. **Visualization Tips:**
```python
import matplotlib.pyplot as plt

def visualize_embedding(embedding):
    plt.figure(figsize=(10, 4))
    plt.plot(embedding)
    plt.title("Embedding Visualization")
    plt.xlabel("Dimension")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()
``` 