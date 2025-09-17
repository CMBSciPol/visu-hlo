# API Reference

## Main Functions

```{eval-rst}
.. automodule:: visu_hlo
   :members:
   :undoc-members:
   :show-inheritance:
```

## Function Reference

### show()

```{eval-rst}
.. autofunction:: visu_hlo.show
```

The main interface for visualizing JAX functions. Automatically detects whether the function is jitted and uses the appropriate visualization method.

**Parameters:**
- `f`: The JAX function to visualize
- `*args`: Positional arguments to pass to the function
- `**keywords`: Keyword arguments to pass to the function

**Returns:**
- None (displays the visualization)

**Example:**
```python
import jax.numpy as jnp
from visu_hlo import show

def my_function(x, y, scale=1.0):
    return (x + y) * scale

show(my_function, jnp.ones(5), jnp.zeros(5), scale=2.0)
```

## Internal Functions

The following functions are used internally but may be useful for advanced usage:

### _get_original_dot_graph()

```{eval-rst}
.. autofunction:: visu_hlo._get_original_dot_graph
```

Generates DOT graph representation for non-jitted functions.

### _get_compiled_dot_graph()

```{eval-rst}
.. autofunction:: visu_hlo._get_compiled_dot_graph
```

Generates DOT graph representation for jitted functions.

### _display_svg()

```{eval-rst}
.. autofunction:: visu_hlo._display_svg
```

Handles the display of SVG content using the system's default viewer.

## Constants

### DISPLAY_PROGRAM

The system command used to open SVG files:
- Linux: `'xdg-open'`
- macOS: `'open'`
- Windows: `'start'`

This is automatically detected based on the platform.
