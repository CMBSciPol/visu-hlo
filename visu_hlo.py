"""Displays the HLO representation of (un-)jitted functions as SVG."""

import platform
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

try:
    import graphviz
except ImportError:
    raise ImportError('Please install graphviz to use this function.')

import jax
from jaxlib.xla_client import _xla as xla

if platform.platform().startswith('Linux'):
    DISPLAY_PROGRAM = 'xdg-open'
elif platform.platform() == 'Darwin':
    DISPLAY_PROGRAM = 'open'
elif platform.platform().startswith('Windows'):
    DISPLAY_PROGRAM = 'start'
else:
    raise RuntimeError('Unsupported platform')


def show(f, *args, **keywords) -> None:
    """Displays the HLO representation of (un-)jitted functions as SVG.

    Args:
        f: Function to be displayed.
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
    if hasattr(f, 'lower'):
        get_dot_graph = _get_compiled_dot_graph
    else:
        get_dot_graph = _get_original_dot_graph
    dot_graph = get_dot_graph(f, *args, **keywords)
    svg_graph = graphviz.pipe_string('dot', 'svg', dot_graph, encoding='utf-8')
    _display_svg(svg_graph)


def _get_original_dot_graph(f, *args, **keywords) -> str:
    lowered = jax.jit(f).lower(*args, **keywords)
    hlo_text = lowered.as_text('hlo')
    return _get_dot_graph_from_hlo(hlo_text)


def _get_compiled_dot_graph(f, *args, **keywords) -> str:
    lowered = jax.jit(f).lower(*args, **keywords)
    hlo_text = lowered.compile().as_text()
    return _get_dot_graph_from_hlo(hlo_text)


def _get_dot_graph_from_hlo(hlo_text: str) -> str:
    hlo_module = xla.hlo_module_from_text(hlo_text)
    return xla.hlo_module_to_dot_graph(hlo_module)


def _display_svg(svg_graph: str) -> None:
    if _in_notebook():
        from IPython.display import SVG, display

        display(SVG(svg_graph))
    else:
        with NamedTemporaryFile(suffix='.svg', delete=False) as file:
            Path(file.name).write_text(svg_graph)
            subprocess.run([DISPLAY_PROGRAM, file.name])


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True
