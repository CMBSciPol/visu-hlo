# Development

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/CMBSciPol/visu-hlo.git
cd visu-hlo
```

2. Install in development mode:
```bash
pip install --group dev .
```

## Running Tests

The project uses pytest for testing:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test categories
pytest -m integration  # Integration tests only
pytest tests/test_platform.py  # Specific test file
```

### Test Environment

Tests automatically configure JAX to use CPU-only mode for reproducibility. The test suite includes:

- **Unit tests**: Individual function testing with mocks
- **Integration tests**: End-to-end testing with real JAX functions
- **Platform tests**: Cross-platform compatibility testing

## Code Quality

The project uses several tools for code quality:

### Ruff (Linting and Formatting)
```bash
# Check code style
ruff check .

# Format code
ruff format .
```

### Type Checking
The project uses type hints throughout. Verify with mypy or your IDE.

## Documentation

Build documentation locally:

```bash
# Install documentation dependencies
pip install --group docs .

# Build HTML documentation
cd docs
sphinx-build -b html source _build/html
```

The documentation uses:
- **Sphinx**: Documentation generator
- **MyST**: Markdown support in Sphinx
- **Read the Docs Theme**: Clean, responsive theme
- **nbsphinx**: Jupyter notebook support

## Project Structure

```
visu-hlo/
├── visu_hlo.py          # Main module
├── tests/               # Test suite
│   ├── test_platform.py    # Platform detection tests
│   ├── test_display.py     # Display functionality tests
│   ├── test_show.py        # Main function tests
│   ├── test_integration.py # Integration tests
│   └── conftest.py         # Test configuration
├── docs/                # Documentation
│   └── source/             # Sphinx source files
├── .github/             # CI/CD workflows
└── pyproject.toml       # Project configuration
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `pytest`
6. Check code style: `ruff check .`
7. Submit a pull request

## Release Process

Releases are automated through GitHub Actions:

1. Create a new tag: `git tag v0.x.x`
2. Push the tag: `git push origin v0.x.x`
3. Create a GitHub release
4. The CI will automatically build and publish to PyPI
