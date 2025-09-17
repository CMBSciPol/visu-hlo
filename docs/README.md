# Documentation

This directory contains the documentation for visu-hlo, built with Sphinx and MyST.

## Building Documentation

### Using uv (recommended)

```bash
uv run --group docs make html
```

### Using pip

```bash
pip install --group docs .
cd docs
make html
```

## Viewing Documentation

After building, open `_build/html/index.html` in your browser.

## Documentation Structure

- `source/index.md`: Main page
- `source/installation.md`: Installation instructions
- `source/quickstart.md`: Quick start guide
- `source/examples/`: Usage examples with SVG outputs
- `source/api/`: API reference documentation
- `source/development.md`: Development guide

## Dependencies

Documentation dependencies are defined in the `docs` dependency group in `pyproject.toml`:

- Sphinx ≥ 7.0.0
- Sphinx RTD Theme ≥ 1.3.0
- MyST Parser ≥ 2.0.0
- nbsphinx ≥ 0.9.0
- matplotlib ≥ 3.7.0
- numpy ≥ 1.24.0

## Read the Docs

The documentation is automatically built and deployed on Read the Docs using the configuration in `.readthedocs.yaml`.
