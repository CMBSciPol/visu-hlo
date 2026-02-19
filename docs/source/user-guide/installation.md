# Installation

## Requirements

- Python ≥ 3.10
- JAX ≥ 0.4.0
- Graphviz ≥ 0.18.2

## Install from PyPI

```bash
pip install visu-hlo
```

## Install from Source

```bash
git clone https://github.com/CMBSciPol/visu-hlo.git
cd visu-hlo
pip install .
```

## Development Installation

For development, install with the `dev` dependency group:

```bash
git clone https://github.com/CMBSciPol/visu-hlo.git
cd visu-hlo
pip install --group dev .
```

This includes additional tools for testing and development.

## System Dependencies

### Graphviz

visu_hlo requires the Graphviz system package for rendering graphs:

#### Ubuntu/Debian
```bash
sudo apt-get install graphviz
```

#### macOS
```bash
brew install graphviz
```

#### Windows
Download and install from [graphviz.org](https://graphviz.org/download/)

## CUDA Support (Optional)

For GPU acceleration with JAX:

```bash
pip install --group cuda12 .
```

This installs JAX with CUDA 12 support.

## Troubleshooting

### GraphvizNotFoundError

If you see the error:
```
GraphvizNotFoundError: Graphviz is not installed or not found in PATH.
```

This means the Graphviz system package is not installed. Install it using the instructions in the [System Dependencies](#system-dependencies) section above.

**Note:** The Python `graphviz` package (installed automatically with visu-hlo) is just a wrapper. You also need the Graphviz system binaries (`dot`, `neato`, etc.) to be installed and available in your PATH.

### JAX Installation Issues

If JAX fails to install, you may need to install it separately first:

```bash
pip install jax jaxlib
```

For specific hardware (GPU/TPU), see the [JAX installation guide](https://github.com/google/jax#installation).

### Import Errors

If you get import errors related to `jaxlib`, ensure you have compatible versions of JAX and jaxlib:

```bash
pip install --upgrade jax jaxlib
```
