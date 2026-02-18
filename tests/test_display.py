"""Tests for SVG display functionality."""

import subprocess

import pytest
import pytest_mock

from visu_hlo._display import DISPLAY_PROGRAM, DotGraphViewer


class TestInNotebook:
    """Tests for _in_notebook detection."""

    def test_in_notebook(self, mocker: pytest_mock.MockFixture) -> None:
        mocked_get_ipython = mocker.patch('IPython.get_ipython')
        mocked_get_ipython().config = {'IPKernelApp': None}

        assert DotGraphViewer._in_notebook() is True

    def test_in_notebook_no_ipython(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test detection when IPython is not installed."""
        # Force reimport to trigger ImportError path
        mocker.patch('builtins.__import__', side_effect=ImportError)

        assert DotGraphViewer._in_notebook() is False

    def test_in_notebook_no_kernel(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test detection when IPython exists but no kernel."""
        mocked_get_ipython = mocker.patch('IPython.get_ipython')
        mocked_get_ipython().config.return_value = {}

        assert DotGraphViewer._in_notebook() is False

    def test_in_notebook_get_ipython_none(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test detection when get_ipython returns None."""
        mocker.patch('IPython.get_ipython', return_value=None)

        assert DotGraphViewer._in_notebook() is False


class TestDotGraphViewerShowInProgram:
    """Tests for terminal display mode."""

    def test_show_in_program(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test SVG display in terminal mode."""
        mock_tempfile = mocker.patch('visu_hlo._display.NamedTemporaryFile')
        mock_tempfile.return_value.__enter__.return_value.name = '/tmp/test.svg'
        mock_write_svg = mocker.patch('visu_hlo._display.DotGraphViewer.write_svg')
        mock_subprocess = mocker.patch('subprocess.run')

        viewer = DotGraphViewer('digraph {}')
        viewer._show_in_program()

        mock_write_svg.assert_called_once_with('/tmp/test.svg')
        mock_subprocess.assert_called_once_with([DISPLAY_PROGRAM, '/tmp/test.svg'])

    def test_show_in_program_subprocess_error(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test handling of subprocess errors."""
        mocker.patch('visu_hlo._display.DotGraphViewer.write_svg')
        mocker.patch(
            'subprocess.run',
            side_effect=subprocess.CalledProcessError(1, 'cmd'),
        )

        viewer = DotGraphViewer('digraph {}')
        with pytest.raises(subprocess.CalledProcessError):
            viewer._show_in_program()


class TestDotGraphViewerShowInNotebook:
    """Tests for notebook display mode."""

    def test_show_in_notebook(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test SVG display in notebook mode."""
        mock_svg = mocker.patch('IPython.display.SVG')
        mock_display = mocker.patch('IPython.display.display')
        mocker.patch('visu_hlo._display.DotGraphViewer._as_svg', return_value='<svg></svg>')

        viewer = DotGraphViewer('digraph {}')
        viewer._show_in_notebook()

        mock_svg.assert_called_once_with('<svg></svg>')
        mock_display.assert_called_once()

    def test_show_in_notebook_no_file_operations(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test that notebook mode doesn't use file operations."""
        mocker.patch('IPython.display.SVG')
        mocker.patch('IPython.display.display')
        mocker.patch('visu_hlo._display.DotGraphViewer._as_svg', return_value='<svg></svg>')
        mock_tempfile = mocker.patch('visu_hlo._display.NamedTemporaryFile')
        mock_subprocess = mocker.patch('subprocess.run')

        viewer = DotGraphViewer('digraph {}')
        viewer._show_in_notebook()

        mock_tempfile.assert_not_called()
        mock_subprocess.assert_not_called()


class TestDotGraphViewerWrite:
    """Tests for file writing methods."""

    def test_write_dot(self, mocker: pytest_mock.MockerFixture, tmp_path) -> None:
        """Test writing DOT file."""
        dot_content = 'digraph { a -> b }'
        viewer = DotGraphViewer(dot_content)

        output_path = tmp_path / 'test.dot'
        viewer.write_dot(output_path)

        assert output_path.read_text() == dot_content

    def test_write_svg(self, mocker: pytest_mock.MockerFixture, tmp_path) -> None:
        """Test writing SVG file."""
        mocker.patch('visu_hlo._display.DotGraphViewer._as_svg', return_value='<svg>test</svg>')

        viewer = DotGraphViewer('digraph {}')
        output_path = tmp_path / 'test.svg'
        viewer.write_svg(output_path)

        assert output_path.read_text() == '<svg>test</svg>'

    def test_write_svg_error(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test handling of file write errors."""
        mocker.patch('visu_hlo._display.DotGraphViewer._as_svg', return_value='<svg></svg>')
        mocker.patch('visu_hlo._display.Path.write_text', side_effect=OSError('Cannot write'))

        viewer = DotGraphViewer('digraph {}')
        with pytest.raises(OSError, match='Cannot write'):
            viewer.write_svg('/invalid/path.svg')


class TestDotGraphViewerAsSvg:
    """Tests for SVG conversion."""

    def test_as_svg_calls_graphviz(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test that _as_svg calls graphviz correctly."""
        mock_pipe = mocker.patch('graphviz.pipe_string', return_value='<svg></svg>')

        viewer = DotGraphViewer('digraph { a -> b }')
        result = viewer._as_svg()

        mock_pipe.assert_called_once_with('dot', 'svg', 'digraph { a -> b }', encoding='utf-8')
        assert result == '<svg></svg>'
