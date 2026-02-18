"""Tests for HLO dot graph generation functions."""

import jax
import visu_hlo


def test_lowered_function():
    """Test dot graph generation for a non-jitted functions."""

    def func(x):
        return x * 2

    hlo = jax.jit(func).lower(2).as_text('hlo')
    dot_graph = visu_hlo._get_dot_graph_from_hlo(hlo)

    assert isinstance(dot_graph, str)
    assert 'digraph' in dot_graph.lower()
    assert len(dot_graph) > 0


def test_compiled_function():
    """Test dot graph generation for a jitted functions."""

    def func(x):
        return x * 2

    hlo = jax.jit(func).lower(2).compile().as_text()
    dot_graph = visu_hlo._get_dot_graph_from_hlo(hlo)

    assert isinstance(dot_graph, str)
    assert 'digraph' in dot_graph.lower()
    assert len(dot_graph) > 0
