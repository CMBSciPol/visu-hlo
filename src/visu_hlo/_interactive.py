"""Displays the HLO representation of (un-)jitted functions as SVG."""

from typing import Any

from ._display import HLOViewer
from ._hlo import from_compiled_function, from_lowered_function, from_stable_hlo

__all__ = ['show']


def show(f, *args: Any, **keywords: Any) -> None:
    """Displays the HLO representation of (un-)jitted functions as SVG.

    Args:
        f: Function to be displayed. It can be a callable (jitted or not), or it can a HLO or
            StableHLO representation as string.
        *args: Arguments to be passed to f.
        **keywords: Keyword arguments to be passed to f.

    Usage:
        >>> import jax.numpy as jnp
        >>> from jax import jit
        >>> from visu_hlo import show
        >>> def func(x):
        ...     return 3 * x * 2
        >>> show(func, jnp.ones(10))  # Display HLO for original function
        >>> show(jit(func), jnp.ones(10))  # Display HLO for jitted function
    """
    if isinstance(f, str):
        if f.startswith('HloModule '):
            hlo = f
        else:
            hlo = from_stable_hlo(f)
    else:
        if hasattr(f, 'lower'):
            hlo = from_compiled_function(f, *args, **keywords)
        else:
            hlo = from_lowered_function(f, *args, **keywords)

    viewer = HLOViewer(hlo)
    viewer.show()
