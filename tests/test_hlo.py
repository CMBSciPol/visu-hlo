"""Tests for HLO dot graph generation functions."""

import jax.numpy as jnp
import visu_hlo
from jax import jit


def test_lowered_function():
    """Test HLO generation from function."""

    def func(x):
        return x * 2

    hlo = visu_hlo._get_lowered_hlo(func, jnp.ones(3))

    assert hlo.startswith('HloModule jit_func')


def test_compiled_function():
    """Test HLO generation from jitted function."""

    @jit
    def func(x):
        return x * 3

    hlo = visu_hlo._get_compiled_hlo(func, jnp.ones(3))

    assert hlo.startswith('HloModule jit_func')
