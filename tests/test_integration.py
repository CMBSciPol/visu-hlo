"""Integration tests with real JAX functions."""

import subprocess

import jax
import jax.numpy as jnp
import pytest
import pytest_mock
import visu_hlo

pytestmark = pytest.mark.integration


@pytest.mark.parametrize('do_jit', [False, True])
def test_function(mocker: pytest_mock.MockerFixture, do_jit: bool) -> None:
    """Integration test with jitted function (mocking only display)."""
    mocker.patch('visu_hlo.DISPLAY_PROGRAM', 'touch')
    spied_run = mocker.spy(subprocess, 'run')

    def func(x):
        return jnp.dot(x, x) + jnp.sin(jnp.sum(x))

    if do_jit:
        func = jax.jit(func)

    # This should work without errors
    visu_hlo.show(func, jnp.array([1.0, 2.0, 3.0, 4.0]))

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
