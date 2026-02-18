"""Tests for the main show function."""

import jax
import jax.numpy as jnp
import pytest_mock

from visu_hlo import show


def test_show_dispatch_hlo(mocker: pytest_mock.MockerFixture) -> None:
    """Test that HLO strings are passed directly."""
    mocked_from_lowered_function = mocker.patch('visu_hlo._interactive.from_lowered_function')
    mocked_from_compiled_function = mocker.patch('visu_hlo._interactive.from_compiled_function')
    mocked_from_stable_hlo = mocker.patch('visu_hlo._interactive.from_stable_hlo')
    mock_viewer = mocker.patch('visu_hlo._interactive.HLOViewer')
    hlo_string = 'HloModule test_module'

    show(hlo_string)

    mocked_from_lowered_function.assert_not_called()
    mocked_from_compiled_function.assert_not_called()
    mocked_from_stable_hlo.assert_not_called()
    mock_viewer.assert_called_once_with(hlo_string)


def test_show_dispatch_stable_hlo(mocker: pytest_mock.MockerFixture) -> None:
    """Test that non-jitted functions dispatch to from_lowered_function."""
    mocked_from_stable_hlo = mocker.patch(
        'visu_hlo._interactive.from_stable_hlo', return_value='HloModule stable_hlo_test'
    )
    mock_viewer = mocker.patch('visu_hlo._interactive.HLOViewer')
    stablehlo_string = 'module @test { func.func @main() {} }'

    show(stablehlo_string)

    mocked_from_stable_hlo.assert_called_once_with(stablehlo_string)
    mock_viewer.assert_called_once_with('HloModule stable_hlo_test')


def test_show_dispatch_lowered_function(mocker: pytest_mock.MockerFixture) -> None:
    """Test that non-jitted functions dispatch to from_lowered_function."""
    mocked_from_lowered_function = mocker.patch(
        'visu_hlo._interactive.from_lowered_function', return_value='HloModule test'
    )
    mock_viewer = mocker.patch('visu_hlo._interactive.HLOViewer')

    def func(x):
        return x + 1

    show(func, jnp.ones(3))

    mocked_from_lowered_function.assert_called_once()
    args = mocked_from_lowered_function.call_args[0]
    assert args[0] is func
    mock_viewer.assert_called_once_with('HloModule test')


def test_show_dispatch_compiled_function(mocker: pytest_mock.MockerFixture) -> None:
    """Test that non-jitted functions dispatch to from_lowered_function."""
    mocked_from_compiled_function = mocker.patch(
        'visu_hlo._interactive.from_compiled_function', return_value='HloModule jitted_test'
    )
    mock_viewer = mocker.patch('visu_hlo._interactive.HLOViewer')

    @jax.jit
    def func(x):
        return x + 1

    show(func, jnp.ones(3))

    mocked_from_compiled_function.assert_called_once()
    args = mocked_from_compiled_function.call_args[0]
    assert args[0] is func
    mock_viewer.assert_called_once_with('HloModule jitted_test')


def test_args_passed_to_from_lowered(mocker: pytest_mock.MockerFixture) -> None:
    """Test that args and kwargs are passed correctly."""
    mock_from_lowered = mocker.patch(
        'visu_hlo._interactive.from_lowered_function', return_value='HloModule test'
    )
    mocker.patch('visu_hlo._interactive.HLOViewer')

    def func(x, y, scale=1.0):
        return (x + y) * scale

    arr1 = jnp.ones(3)
    arr2 = jnp.zeros(3)

    show(func, arr1, arr2, scale=2.0)

    mock_from_lowered.assert_called_once()
    args, kwargs = mock_from_lowered.call_args
    assert args[0] is func
    assert jnp.array_equal(args[1], arr1)
    assert jnp.array_equal(args[2], arr2)
    assert kwargs == {'scale': 2.0}
