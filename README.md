# visu-hlo

**visu-hlo** displays the HLO representation of JAX functions as SVG visualizations.

[![Documentation](https://readthedocs.org/projects/visu-hlo/badge/?version=latest)](https://visu-hlo.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/visu-hlo.svg)](https://badge.fury.io/py/visu-hlo)

## Quick Example

```python
import jax.numpy as jnp
from visu_hlo import show

def func(x):
    return 3 * x * 2

# Visualize the graph of the XLA-optimized function
show(func, jnp.ones(10))

# Visualize the graph of the original function
show(func, jnp.ones(10), jit=False)
```

## Installation

```bash
pip install visu-hlo
```

**System dependency:** Install [Graphviz](https://graphviz.org/download/)

## Features

- 🎯 **Easy Visualization**: Display HLO graphs with a single function call
- ⚡ **JIT Support**: Works with both regular and jitted JAX functions
- 🖼️ **SVG Output**: High-quality vector graphics that scale perfectly
- 🖥️ **Cross-Platform**: Supports Linux, macOS, and Windows
- 📦 **Lightweight**: Minimal dependencies - just JAX and Graphviz

## Documentation

Full documentation: https://visu-hlo.readthedocs.io/

## License

MIT
