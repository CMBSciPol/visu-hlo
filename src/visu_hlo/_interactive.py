"""Displays the HLO representation of (un-)jitted functions as SVG."""

from typing import Any

import jax

from ._display import HLOViewer
from ._hlo import from_compiled_function, from_lowered_function, from_stable_hlo

__all__ = ['show']


def _unwrap(f):
    """Unwrap jitted functions to get the original function."""
    while hasattr(f, 'lower') and hasattr(f, '__wrapped__'):
        f = f.__wrapped__
    return f


def show(f, *args: Any, jit: bool = True, **keywords: Any) -> None:
    """Displays the HLO representation of functions as SVG.

    Args:
        f: Function to be displayed. It can be a callable (jitted or not), or it can be a HLO or
            StableHLO representation as string.
        *args: Arguments to be passed to f.
        jit: If True, display the optimized HLO (after XLA compilation). If False, display the
            non-optimized HLO.
        **keywords: Keyword arguments to be passed to f.

    Usage:
        >>> import jax.numpy as jnp
        >>> from visu_hlo import show
        >>> def func(x):
        ...     return 3 * x * 2
        >>> show(func, jnp.ones(10))  # Display optimized HLO (default)
        >>> show(func, jnp.ones(10), jit=False)  # Display non-optimized HLO
    """
    if isinstance(f, str):
        if f.startswith('HloModule '):
            hlo = f
        else:
            hlo = from_stable_hlo(f)
    else:
        if jit:
            if not hasattr(f, 'lower'):
                f = jax.jit(f)
            hlo = from_compiled_function(f, *args, **keywords)
        else:
            f = _unwrap(f)
            hlo = from_lowered_function(f, *args, **keywords)

    viewer = HLOViewer(hlo)
    viewer.show()
