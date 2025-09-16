"""Tests for HLO dot graph generation functions."""

import jax.numpy as jnp
import visu_hlo
from jax import jit


def test_get_original_dot_graph():
    """Test dot graph generation for original (non-jitted) functions."""

    def simple_func(x):
        return x * 2

    dot_graph = visu_hlo._get_original_dot_graph(simple_func, jnp.ones(3))

    assert isinstance(dot_graph, str)
    assert 'digraph' in dot_graph.lower()
    assert len(dot_graph) > 0


def test_get_compiled_dot_graph():
    """Test dot graph generation for compiled (jitted) functions."""

    @jit
    def jitted_func(x):
        return x * 3

    dot_graph = visu_hlo._get_compiled_dot_graph(jitted_func, jnp.ones(3))

    assert isinstance(dot_graph, str)
    assert 'digraph' in dot_graph.lower()
    assert len(dot_graph) > 0
