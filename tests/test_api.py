"""Tests for the public API functions."""

import jax
import jax.numpy as jnp
import pytest_mock

from visu_hlo import show, write_dot, write_svg
from visu_hlo._api import _get_viewer, _unwrap


class TestGetViewerDispatch:
    """Tests for _get_viewer() dispatch logic."""

    def test_hlo_string(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test that HLO strings are passed directly."""
        mocked_from_lowered_function = mocker.patch('visu_hlo._api.from_lowered_function')
        mocked_from_compiled_function = mocker.patch('visu_hlo._api.from_compiled_function')
        mocked_from_stable_hlo = mocker.patch('visu_hlo._api.from_stable_hlo')
        mock_viewer = mocker.patch('visu_hlo._api.HLOViewer')
        hlo_string = 'HloModule test_module'

        _get_viewer(hlo_string)

        mocked_from_lowered_function.assert_not_called()
        mocked_from_compiled_function.assert_not_called()
        mocked_from_stable_hlo.assert_not_called()
        mock_viewer.assert_called_once_with(hlo_string)

    def test_stable_hlo_string(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test that StableHLO strings dispatch to from_stable_hlo."""
        mocked_from_stable_hlo = mocker.patch(
            'visu_hlo._api.from_stable_hlo', return_value='HloModule stable_hlo_test'
        )
        mock_viewer = mocker.patch('visu_hlo._api.HLOViewer')
        stablehlo_string = 'module @test { func.func @main() {} }'

        _get_viewer(stablehlo_string)

        mocked_from_stable_hlo.assert_called_once_with(stablehlo_string)
        mock_viewer.assert_called_once_with('HloModule stable_hlo_test')

    def test_non_jitted_function_gets_jitted(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test that non-jitted functions are jitted when jit=True."""
        mocked_jit = mocker.patch('visu_hlo._api.jax.jit')
        mocked_jitted_func = mocked_jit.return_value
        mocked_from_compiled = mocker.patch(
            'visu_hlo._api.from_compiled_function', return_value='HloModule test'
        )
        mocker.patch('visu_hlo._api.HLOViewer')

        def func(x):
            return x + 1

        _get_viewer(func, jnp.ones(3), jit=True)

        mocked_jit.assert_called_once_with(func)
        mocked_from_compiled.assert_called_once()
        assert mocked_from_compiled.call_args[0][0] is mocked_jitted_func

    def test_already_jitted_function_not_re_jitted(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test that already jitted functions are not re-jitted."""
        mocked_from_compiled = mocker.patch(
            'visu_hlo._api.from_compiled_function', return_value='HloModule test'
        )
        mocker.patch('visu_hlo._api.HLOViewer')

        def func(x):
            return x + 1

        jitted_func = jax.jit(func)
        mocked_jit = mocker.patch('visu_hlo._api.jax.jit')

        _get_viewer(jitted_func, jnp.ones(3), jit=True)

        mocked_jit.assert_not_called()
        mocked_from_compiled.assert_called_once()
        assert mocked_from_compiled.call_args[0][0] is jitted_func

    def test_default_is_jit_true(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test that jit=True is the default."""
        mocked_jit = mocker.patch('visu_hlo._api.jax.jit')
        mocker.patch('visu_hlo._api.from_compiled_function', return_value='HloModule test')
        mocker.patch('visu_hlo._api.HLOViewer')

        def func(x):
            return x + 1

        _get_viewer(func, jnp.ones(3))

        mocked_jit.assert_called_once()

    def test_jit_false_uses_lowered(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test that jit=False dispatches to from_lowered_function."""
        mocked_from_lowered = mocker.patch(
            'visu_hlo._api.from_lowered_function', return_value='HloModule test'
        )
        mocker.patch('visu_hlo._api.HLOViewer')

        def func(x):
            return x + 1

        _get_viewer(func, jnp.ones(3), jit=False)

        mocked_from_lowered.assert_called_once()
        assert mocked_from_lowered.call_args[0][0] is func

    def test_jit_false_unwraps_jitted_function(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test that jitted functions are unwrapped when jit=False."""
        mocked_from_lowered = mocker.patch(
            'visu_hlo._api.from_lowered_function', return_value='HloModule test'
        )
        mocker.patch('visu_hlo._api.HLOViewer')

        def func(x):
            return x + 1

        jitted_func = jax.jit(func)

        _get_viewer(jitted_func, jnp.ones(3), jit=False)

        mocked_from_lowered.assert_called_once()
        assert mocked_from_lowered.call_args[0][0] is func

    def test_args_and_kwargs_passed(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test that args and kwargs are passed correctly."""
        mocked_from_compiled = mocker.patch(
            'visu_hlo._api.from_compiled_function', return_value='HloModule test'
        )
        mocker.patch('visu_hlo._api.HLOViewer')

        def func(x, y, scale=1.0):
            return (x + y) * scale

        arr1 = jnp.ones(3)
        arr2 = jnp.zeros(3)

        _get_viewer(jax.jit(func), arr1, arr2, scale=2.0)

        mocked_from_compiled.assert_called_once()
        args, kwargs = mocked_from_compiled.call_args
        assert jnp.array_equal(args[1], arr1)
        assert jnp.array_equal(args[2], arr2)
        assert kwargs == {'scale': 2.0}


class TestShow:
    """Tests for show() function."""

    def test_calls_viewer_show(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test that show() calls viewer.show()."""
        mock_viewer_instance = mocker.MagicMock()
        mocker.patch('visu_hlo._api._get_viewer', return_value=mock_viewer_instance)

        show('HloModule test')

        mock_viewer_instance.show.assert_called_once()


class TestWriteDot:
    """Tests for write_dot() function."""

    def test_calls_viewer_write_dot(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test that write_dot() calls viewer.write_dot() with the path."""
        mock_viewer_instance = mocker.MagicMock()
        mocker.patch('visu_hlo._api._get_viewer', return_value=mock_viewer_instance)

        write_dot('/path/to/file.dot', 'HloModule test')

        mock_viewer_instance.write_dot.assert_called_once_with('/path/to/file.dot')

    def test_passes_args_to_get_viewer(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test that write_dot() passes arguments to _get_viewer."""
        mock_get_viewer = mocker.patch('visu_hlo._api._get_viewer')
        mock_get_viewer.return_value = mocker.MagicMock()

        def func(x):
            return x

        write_dot('/path/to/file.dot', func, jnp.ones(3), jit=False)

        mock_get_viewer.assert_called_once()
        args, kwargs = mock_get_viewer.call_args
        assert args[0] is func
        assert jnp.array_equal(args[1], jnp.ones(3))
        assert kwargs == {'jit': False}


class TestWriteSvg:
    """Tests for write_svg() function."""

    def test_calls_viewer_write_svg(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test that write_svg() calls viewer.write_svg() with the path."""
        mock_viewer_instance = mocker.MagicMock()
        mocker.patch('visu_hlo._api._get_viewer', return_value=mock_viewer_instance)

        write_svg('/path/to/file.svg', 'HloModule test')

        mock_viewer_instance.write_svg.assert_called_once_with('/path/to/file.svg')

    def test_passes_args_to_get_viewer(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test that write_svg() passes arguments to _get_viewer."""
        mock_get_viewer = mocker.patch('visu_hlo._api._get_viewer')
        mock_get_viewer.return_value = mocker.MagicMock()

        def func(x):
            return x

        write_svg('/path/to/file.svg', func, jnp.ones(3), jit=False)

        mock_get_viewer.assert_called_once()
        args, kwargs = mock_get_viewer.call_args
        assert args[0] is func
        assert jnp.array_equal(args[1], jnp.ones(3))
        assert kwargs == {'jit': False}


class TestUnwrap:
    """Tests for _unwrap function."""

    def test_unwrap_non_jitted(self) -> None:
        """Test that non-jitted functions are returned as-is."""

        def func(x):
            return x

        assert _unwrap(func) is func

    def test_unwrap_jitted(self) -> None:
        """Test that jitted functions are unwrapped."""

        def func(x):
            return x

        jitted = jax.jit(func)

        assert _unwrap(jitted) is func

    def test_unwrap_double_jitted(self) -> None:
        """Test that double-jitted functions are fully unwrapped."""

        def func(x):
            return x

        jitted = jax.jit(jax.jit(func))

        assert _unwrap(jitted) is func

    def test_unwrap_stops_at_non_jitted_wrapper(self) -> None:
        """Test that unwrap stops when reaching a non-jitted wrapper."""
        from functools import wraps

        def func(x):
            return x

        @wraps(func)
        def wrapper(x):
            return func(x)

        assert hasattr(wrapper, '__wrapped__')
        assert not hasattr(wrapper, 'lower')
        assert _unwrap(wrapper) is wrapper
