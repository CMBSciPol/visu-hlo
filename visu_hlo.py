"""Displays the HLO representation of (un-)jitted functions as SVG."""

import platform
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

try:
    import graphviz
except ImportError:
    raise ImportError('Please install graphviz to use this function.')

import jax
import jaxlib
from jaxlib.xla_client import _xla as xla

if platform.platform().startswith('Linux'):
    DISPLAY_PROGRAM = 'xdg-open'
elif platform.platform() == 'Darwin':
    DISPLAY_PROGRAM = 'open'
elif platform.platform().startswith('Windows'):
    DISPLAY_PROGRAM = 'start'
else:
    raise RuntimeError('Unsupported platform')


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
            hlo = _get_hlo_from_stable_hlo(f)
    else:
        if hasattr(f, 'lower'):
            hlo = _get_compiled_hlo(f, *args, **keywords)
        else:
            hlo = _get_lowered_hlo(f, *args, **keywords)
    dot_graph = _get_dot_graph_from_hlo(hlo)
    svg_graph = graphviz.pipe_string('dot', 'svg', dot_graph, encoding='utf-8')
    _display_svg(svg_graph)


def _get_lowered_hlo(f, *args: Any, **keywords: Any) -> str:
    lowered = jax.jit(f).lower(*args, **keywords)
    return lowered.as_text('hlo')


def _get_compiled_hlo(f, *args: Any, **keywords: Any) -> str:
    lowered = f.lower(*args, **keywords)
    return lowered.compile().as_text()


def _get_hlo_from_stable_hlo(stable_hlo_str: str) -> str:
    """StableHLO to HLO conversion.

    The only easy way produce a DOT (Graphviz) representation is from HLO
    and is no easy way in JAX to convert StableHLO to HLO."""

    from jax._src.interpreters.mlir import make_ir_context
    from jax._src.lib.mlir import ir

    # Parse StableHLO in MLIR bytecode
    ctx = make_ir_context()
    with ctx:
        module = ir.Module.parse(stable_hlo_str)
        bytecode = module.operation.get_asm(binary=True)

    computation = jaxlib._jax.mlir.mlir_module_to_xla_computation(bytecode, use_tuple_args=False)

    return computation.as_hlo_text()


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
