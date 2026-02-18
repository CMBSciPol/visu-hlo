"""HLO extraction utilities."""

from collections.abc import Callable
from typing import Any

import jax
import jaxlib


def from_lowered_function(f: Callable[..., Any], *args: Any, **keywords: Any) -> str:
    """Extract non-optimized HLO from a function.

    Args:
        f: Function to extract HLO from (must not be jitted).
        *args: Arguments to be passed to f.
        **keywords: Keyword arguments to be passed to f.

    Returns:
        HLO text representation.
    """
    lowered = jax.jit(f).lower(*args, **keywords)
    result: str = lowered.as_text('hlo')
    return result


def from_compiled_function(f: Callable[..., Any], *args: Any, **keywords: Any) -> str:
    """Extract optimized HLO from a jitted function.

    Args:
        f: Jitted function to extract HLO from (must have a `lower` method).
        *args: Arguments to be passed to f.
        **keywords: Keyword arguments to be passed to f.

    Returns:
        HLO text representation after XLA compilation.
    """
    lowered = f.lower(*args, **keywords)  # type: ignore[attr-defined]
    result: str = lowered.compile().as_text()
    return result


def from_stable_hlo(stable_hlo_str: str) -> str:
    """Convert StableHLO to HLO.

    Args:
        stable_hlo_str: StableHLO text representation (MLIR format).

    Returns:
        HLO text representation.

    Note:
        The only easy way to produce a DOT (Graphviz) representation is from HLO,
        and there is no easy way in JAX to convert StableHLO to HLO directly.
    """

    from jax._src.interpreters.mlir import make_ir_context
    from jax._src.lib.mlir import ir  # type: ignore[attr-defined]

    # Parse StableHLO in MLIR bytecode
    ctx = make_ir_context()
    with ctx:
        module = ir.Module.parse(stable_hlo_str)  # type: ignore[attr-defined]
        bytecode: bytes = module.operation.get_asm(binary=True)

    computation = jaxlib._jax.mlir.mlir_module_to_xla_computation(bytecode, use_tuple_args=False)

    result: str = computation.as_hlo_text()
    return result
