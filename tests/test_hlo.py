"""Tests for HLO dot graph generation functions."""

import jax.numpy as jnp
from jax import jit

from visu_hlo._hlo import from_compiled_function, from_lowered_function, from_stable_hlo


def test_from_lowered_function():
    """Test HLO generation from function."""

    def func(x):
        return x * 2

    hlo = from_lowered_function(func, jnp.ones(3))

    assert hlo.startswith('HloModule jit_func')


def test_from_compiled_function():
    """Test HLO generation from jitted function."""

    @jit
    def func(x):
        return x * 3

    hlo = from_compiled_function(func, jnp.ones(3))

    assert hlo.startswith('HloModule jit_func')


def test_from_stable_hlo() -> None:
    stable_hlo = """
    module @my_module attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
      func.func public @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
        %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
        return %0 : tensor<4xf32>
      }
    }
    """

    hlo = from_stable_hlo(stable_hlo)

    assert type(hlo) is str
    assert hlo.startswith('HloModule my_module')
