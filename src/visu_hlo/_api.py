"""Public API for visu_hlo."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax

from ._display import HLOViewer
from ._hlo import from_compiled_function, from_lowered_function, from_stable_hlo

__all__ = ['show', 'write_dot', 'write_svg']

# Type alias for function or HLO/StableHLO string
FunctionOrHLO = Callable[..., Any] | str


def _unwrap(f: Callable[..., Any]) -> Callable[..., Any]:
    """Unwrap jitted functions to get the original function."""
    while hasattr(f, 'lower') and hasattr(f, '__wrapped__'):
        f = f.__wrapped__
    return f


def _get_viewer(f: FunctionOrHLO, *args: Any, jit: bool = True, **keywords: Any) -> HLOViewer:
    """Create an HLOViewer from a function or HLO string."""
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

    return HLOViewer(hlo)


def show(f: FunctionOrHLO, *args: Any, jit: bool = True, **keywords: Any) -> None:
    """Display the HLO representation of functions as SVG.

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
    viewer = _get_viewer(f, *args, jit=jit, **keywords)
    viewer.show()


def write_dot(
    path: str | Path, f: FunctionOrHLO, *args: Any, jit: bool = True, **keywords: Any
) -> None:
    """Write the HLO representation of functions as a DOT file.

    Args:
        path: Path to the DOT file.
        f: Function to be written. It can be a callable (jitted or not), or it can be a HLO or
            StableHLO representation as string.
        *args: Arguments to be passed to f.
        jit: If True, write the optimized HLO (after XLA compilation). If False, write the
            non-optimized HLO.
        **keywords: Keyword arguments to be passed to f.

    Usage:
        >>> import jax.numpy as jnp
        >>> from visu_hlo import write_dot
        >>> def func(x):
        ...     return 3 * x * 2
        >>> write_dot('graph.dot', func, jnp.ones(10))
    """
    viewer = _get_viewer(f, *args, jit=jit, **keywords)
    viewer.write_dot(path)


def write_svg(
    path: str | Path, f: FunctionOrHLO, *args: Any, jit: bool = True, **keywords: Any
) -> None:
    """Write the HLO representation of functions as an SVG file.

    Args:
        path: Path to the SVG file.
        f: Function to be written. It can be a callable (jitted or not), or it can be a HLO or
            StableHLO representation as string.
        *args: Arguments to be passed to f.
        jit: If True, write the optimized HLO (after XLA compilation). If False, write the
            non-optimized HLO.
        **keywords: Keyword arguments to be passed to f.

    Usage:
        >>> import jax.numpy as jnp
        >>> from visu_hlo import write_svg
        >>> def func(x):
        ...     return 3 * x * 2
        >>> write_svg('graph.svg', func, jnp.ones(10))
    """
    viewer = _get_viewer(f, *args, jit=jit, **keywords)
    viewer.write_svg(path)
