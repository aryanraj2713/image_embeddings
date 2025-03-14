# Contributing to imgemb

Thank you for your interest in contributing to imgemb! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Documentation](#documentation)

## Code of Conduct

This project follows a standard code of conduct. Please be respectful and professional in all interactions. We are committed to providing a welcoming and inspiring community for all.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
```bash
git clone https://github.com/YOUR_USERNAME/image_embeddings.git
cd image_embeddings
```

3. Add the upstream repository:
```bash
git remote add upstream https://github.com/aryanraj2713/image_embeddings.git
```

## Development Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Making Changes

1. Create a new branch for your changes:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes, following our [style guidelines](#style-guidelines)

3. Commit your changes:
```bash
git add .
git commit -m "Description of your changes"
```

## Testing

1. Run the test suite:
```bash
pytest tests/ -v
```

2. Check test coverage:
```bash
pytest tests/ -v --cov=image_embeddings --cov-report=term-missing
```

3. Run linting:
```bash
flake8 .
black --check .
```

## Pull Request Process

1. Update documentation for any new features or changes
2. Add or update tests as needed
3. Ensure all tests pass and coverage is maintained
4. Update the README.md if necessary
5. Push your changes to your fork:
```bash
git push origin feature/your-feature-name
```
6. Create a Pull Request through GitHub

## Style Guidelines

1. Code Style:
   - Follow PEP 8 guidelines
   - Use Black for code formatting
   - Use type hints for function parameters and return values
   - Write descriptive variable and function names

2. Docstrings:
   - Use Google-style docstrings
   - Include type information
   - Document parameters, return values, and exceptions
   - Provide examples for complex functionality

Example:
```python
def process_image(
    image_path: str,
    method: str = "grid"
) -> np.ndarray:
    """Process an image and return its features.

    Args:
        image_path (str): Path to the image file
        method (str, optional): Processing method. Defaults to "grid"

    Returns:
        np.ndarray: Processed image features

    Raises:
        ValueError: If the image path is invalid
        TypeError: If parameters are of wrong type
    """
```

## Documentation

1. Code Documentation:
   - Document all public functions, classes, and methods
   - Include inline comments for complex logic
   - Update API reference when adding new features

2. Example Updates:
   - Add examples for new features
   - Update existing examples if behavior changes
   - Include doctest examples where appropriate

3. README Updates:
   - Keep feature list current
   - Update installation instructions if needed
   - Add new usage examples for new features

## Project Structure
```
image_embeddings/
├── docs/
│   ├── api_reference.md
│   └── usage.md
├── examples/
│   ├── basic_usage.py
│   ├── image_clustering.py
│   └── image_similarity.py
├── image_embeddings/
│   ├── __init__.py
│   ├── embedder.py
│   └── cli/
│       ├── __init__.py
│       └── main.py
├── tests/
│   ├── test_cli.py
│   └── test_embedder.py
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── pyproject.toml
```

## Questions or Issues?

If you have questions or run into issues, please:
1. Check existing issues on GitHub
2. Create a new issue if needed
3. Ask questions in the issue tracker
4. Join our discussions

Thank you for contributing to imgemb! 