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
