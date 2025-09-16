"""Tests for the main show function."""

import jax.numpy as jnp
import pytest
import pytest_mock
import visu_hlo
from jax import jit


def test_show_original_function(mocker: pytest_mock.MockerFixture) -> None:
    """Test show function with original (non-jitted) function."""
    mocked_pipe_string = mocker.patch('graphviz.pipe_string', return_value='<svg>test</svg>')
    mocked_display_svg = mocker.patch('visu_hlo._display_svg')
    spied_get_graph = mocker.spy(visu_hlo, '_get_original_dot_graph')

    def test_func(x):
        return x + 1

    visu_hlo.show(test_func, jnp.ones(3))

    assert spied_get_graph.call_count == 1
    mocked_pipe_string.assert_called_once()
    args, kwargs = mocked_pipe_string.call_args
    assert args[0] == 'dot'
    assert args[1] == 'svg'
    assert isinstance(args[2], str)  # DOT graph string
    assert kwargs.get('encoding') == 'utf-8'

    mocked_display_svg.assert_called_once_with('<svg>test</svg>')


def test_show_jitted_function(mocker: pytest_mock.MockerFixture) -> None:
    """Test show function with jitted function."""
    mocked_pipe_string = mocker.patch('graphviz.pipe_string', return_value='<svg>jitted_test</svg>')
    mocked_display_svg = mocker.patch('visu_hlo._display_svg')
    spied_get_graph = mocker.spy(visu_hlo, '_get_compiled_dot_graph')

    @jit
    def jitted_func(x):
        return x * 2

    visu_hlo.show(jitted_func, jnp.ones(3))

    assert spied_get_graph.call_count == 1
    mocked_pipe_string.assert_called_once()
    args, kwargs = mocked_pipe_string.call_args
    assert args[0] == 'dot'
    assert args[1] == 'svg'
    assert isinstance(args[2], str)  # DOT graph string
    assert kwargs.get('encoding') == 'utf-8'

    mocked_display_svg.assert_called_once_with('<svg>jitted_test</svg>')


def test_show_with_args_and_kwargs(mocker: pytest_mock.MockerFixture) -> None:
    """Test show function with positional and keyword arguments."""
    mocked_pipe_string = mocker.patch(
        'graphviz.pipe_string', return_value='<svg>args_kwargs_test</svg>'
    )
    mocked_display_svg = mocker.patch('visu_hlo._display_svg')

    def func_with_kwargs(x, y, multiplier=1):
        return x * y * multiplier

    visu_hlo.show(func_with_kwargs, jnp.ones(3), jnp.zeros(3), multiplier=5)

    mocked_pipe_string.assert_called_once()
    mocked_display_svg.assert_called_once_with('<svg>args_kwargs_test</svg>')


def test_show_detects_jitted_function_correctly(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test that show function correctly detects jitted vs non-jitted functions."""
    mocked_pipe_string = mocker.patch(
        'graphviz.pipe_string', return_value='<svg>detection_test</svg>'
    )
    mocked_display_svg = mocker.patch('visu_hlo._display_svg')

    def regular_func(x):
        return x

    @jit
    def jitted_func(x):
        return x

    # Test regular function (should not have 'lower' attribute initially)
    assert not hasattr(regular_func, 'lower')
    visu_hlo.show(regular_func, jnp.ones(3))

    # Test jitted function (should have 'lower' attribute)
    assert hasattr(jitted_func, 'lower')
    visu_hlo.show(jitted_func, jnp.ones(3))

    # Both should have been called
    assert mocked_pipe_string.call_count == 2
    assert mocked_display_svg.call_count == 2


def test_show_propagates_graphviz_errors(mocker: pytest_mock.MockerFixture) -> None:
    """Test that show function propagates graphviz errors."""
    mocker.patch('graphviz.pipe_string', side_effect=Exception('Graphviz error'))

    def test_func(x):
        return x

    with pytest.raises(Exception, match='Graphviz error'):
        visu_hlo.show(test_func, jnp.ones(3))


def test_show_with_invalid_function_arguments() -> None:
    """Test show function behavior with invalid function arguments."""

    def test_func(x):
        return x * 2

    # This should raise an error from JAX/XLA
    with pytest.raises((TypeError, ValueError)):
        visu_hlo.show(test_func, 'invalid_input')
