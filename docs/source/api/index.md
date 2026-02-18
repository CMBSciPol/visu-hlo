# API Reference

## show()

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
