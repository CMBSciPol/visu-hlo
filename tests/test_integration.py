"""Integration tests with real JAX functions."""

import subprocess

import jax.numpy as jnp
import pytest
import pytest_mock
import visu_hlo
from jax import jit

pytestmark = pytest.mark.integration


def test_original_function(mocker: pytest_mock.MockerFixture) -> None:
    """Integration test with real JAX function (mocking only display)."""
    mocker.patch('visu_hlo.DISPLAY_PROGRAM', 'touch')
    spied_run = mocker.spy(subprocess, 'run')

    def real_func(x):
        return jnp.sum(x**2) + jnp.mean(x)

    # This should work without errors
    visu_hlo.show(real_func, jnp.array([1.0, 2.0, 3.0]))

    # Verify that subprocess.run was called twice: once for graphviz, once for display
    assert spied_run.call_count == 2

    # Check the display call (second call)
    display_call = spied_run.call_args_list[1]
    args, kwargs = display_call
    assert len(args) == 1
    assert isinstance(args[0], list)
    assert len(args[0]) == 2  # ['touch', '/path/to/file.svg']
    assert args[0][0] == 'touch'
    assert args[0][1].endswith('.svg')


def test_jitted_function(mocker: pytest_mock.MockerFixture) -> None:
    """Integration test with jitted function (mocking only display)."""
    mocker.patch('visu_hlo.DISPLAY_PROGRAM', 'touch')
    spied_run = mocker.spy(subprocess, 'run')

    @jit
    def jitted_real_func(x):
        return jnp.dot(x, x) + jnp.sin(jnp.sum(x))

    # This should work without errors
    visu_hlo.show(jitted_real_func, jnp.array([1.0, 2.0, 3.0, 4.0]))

    # Verify that subprocess.run was called twice: once for graphviz, once for display
    assert spied_run.call_count == 2

    # Check the display call (second call)
    display_call = spied_run.call_args_list[1]
    args, kwargs = display_call
    assert len(args) == 1
    assert isinstance(args[0], list)
    assert len(args[0]) == 2  # ['touch', '/path/to/file.svg']
    assert args[0][0] == 'touch'
    assert args[0][1].endswith('.svg')
