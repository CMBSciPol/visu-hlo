"""Tests for SVG display functionality."""

import graphviz
import pytest
import pytest_mock

from visu_hlo._display import DISPLAY_PROGRAM, DotGraphViewer, GraphvizNotFoundError, HLOViewer


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


class TestDotGraphViewerShow:
    """Tests for DotGraphViewer.show() dispatch."""

    def test_show_dispatches_to_notebook(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test that show() dispatches to _show_in_notebook when in notebook."""
        mocker.patch.object(DotGraphViewer, '_in_notebook', return_value=True)
        mocked_show_in_notebook = mocker.patch.object(DotGraphViewer, '_show_in_notebook')
        mocked_show_in_program = mocker.patch.object(DotGraphViewer, '_show_in_program')

        viewer = DotGraphViewer('digraph {}')
        viewer.show()

        mocked_show_in_notebook.assert_called_once()
        mocked_show_in_program.assert_not_called()

    def test_show_dispatches_to_program(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test that show() dispatches to _show_in_program when not in notebook."""
        mocker.patch.object(DotGraphViewer, '_in_notebook', return_value=False)
        mocked_show_in_notebook = mocker.patch.object(DotGraphViewer, '_show_in_notebook')
        mocked_show_in_program = mocker.patch.object(DotGraphViewer, '_show_in_program')

        viewer = DotGraphViewer('digraph {}')
        viewer.show()

        mocked_show_in_program.assert_called_once()
        mocked_show_in_notebook.assert_not_called()


class TestDotGraphViewerShowInProgram:
    """Tests for terminal display mode."""

    def test_show_in_program(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test SVG display in terminal mode."""
        mocked_tempfile = mocker.patch('visu_hlo._display.NamedTemporaryFile')
        mocked_tempfile.return_value.__enter__.return_value.name = '/tmp/test.svg'
        mocked_write_svg = mocker.patch('visu_hlo._display.DotGraphViewer.write_svg')
        mocked_subprocess = mocker.patch('subprocess.run')
        mocked_atexit = mocker.patch('visu_hlo._display.atexit.register')

        viewer = DotGraphViewer('digraph {}')
        viewer._show_in_program()

        mocked_write_svg.assert_called_once()
        mocked_subprocess.assert_called_once_with([DISPLAY_PROGRAM, '/tmp/test.svg'], check=False)
        mocked_atexit.assert_called_once()

    def test_show_in_program_registers_cleanup(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test that a cleanup function is registered with atexit."""
        mocker.patch('visu_hlo._display.NamedTemporaryFile')
        mocker.patch('visu_hlo._display.DotGraphViewer.write_svg')
        mocker.patch('subprocess.run')
        mocked_atexit = mocker.patch('visu_hlo._display.atexit.register')

        viewer = DotGraphViewer('digraph {}')
        viewer._show_in_program()

        # Verify atexit.register was called with a callable
        mocked_atexit.assert_called_once()
        cleanup_func = mocked_atexit.call_args[0][0]
        assert callable(cleanup_func)


class TestDotGraphViewerShowInNotebook:
    """Tests for notebook display mode."""

    def test_show_in_notebook(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test SVG display in notebook mode."""
        mocked_svg = mocker.patch('IPython.display.SVG')
        mocked_display = mocker.patch('IPython.display.display')
        mocker.patch('visu_hlo._display.DotGraphViewer._as_svg', return_value='<svg></svg>')

        viewer = DotGraphViewer('digraph {}')
        viewer._show_in_notebook()

        mocked_svg.assert_called_once_with('<svg></svg>')
        mocked_display.assert_called_once()

    def test_show_in_notebook_no_file_operations(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test that notebook mode doesn't use file operations."""
        mocker.patch('IPython.display.SVG')
        mocker.patch('IPython.display.display')
        mocker.patch('visu_hlo._display.DotGraphViewer._as_svg', return_value='<svg></svg>')
        mocked_tempfile = mocker.patch('visu_hlo._display.NamedTemporaryFile')
        mocked_subprocess = mocker.patch('subprocess.run')

        viewer = DotGraphViewer('digraph {}')
        viewer._show_in_notebook()

        mocked_tempfile.assert_not_called()
        mocked_subprocess.assert_not_called()


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
        mocked_pipe = mocker.patch('graphviz.pipe_string', return_value='<svg></svg>')

        viewer = DotGraphViewer('digraph { a -> b }')
        result = viewer._as_svg()

        mocked_pipe.assert_called_once_with('dot', 'svg', 'digraph { a -> b }', encoding='utf-8')
        assert result == '<svg></svg>'

    def test_as_svg_graphviz_not_found(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test error handling when Graphviz is not installed."""
        mocker.patch(
            'graphviz.pipe_string',
            side_effect=graphviz.ExecutableNotFound(['dot']),
        )

        viewer = DotGraphViewer('digraph {}')
        with pytest.raises(GraphvizNotFoundError, match='Graphviz is not installed'):
            viewer._as_svg()


class TestGraphvizNotFoundError:
    """Tests for GraphvizNotFoundError exception."""

    def test_exception_is_importable(self) -> None:
        """Test that the exception can be imported from the package."""
        from visu_hlo import GraphvizNotFoundError as ImportedError

        assert ImportedError is GraphvizNotFoundError

    def test_exception_message(self) -> None:
        """Test exception message content."""
        err = GraphvizNotFoundError('test message')
        assert 'test message' in str(err)


class TestHLOViewer:
    """Tests for HLOViewer class."""

    def test_init(self) -> None:
        """Test HLOViewer initialization."""
        hlo = 'HloModule test'
        viewer = HLOViewer(hlo)
        assert viewer.hlo == hlo

    def test_show(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test that show() delegates to DotGraphViewer.show()."""
        mocked_dot_viewer = mocker.MagicMock()
        mocker.patch.object(HLOViewer, '_dot_graph_viewer', mocked_dot_viewer)

        viewer = HLOViewer('HloModule test')
        viewer.show()

        mocked_dot_viewer.show.assert_called_once()

    def test_write_dot(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test that write_dot() delegates to DotGraphViewer.write_dot()."""
        mocked_dot_viewer = mocker.MagicMock()
        mocker.patch.object(HLOViewer, '_dot_graph_viewer', mocked_dot_viewer)

        viewer = HLOViewer('HloModule test')
        viewer.write_dot('/path/to/file.dot')

        mocked_dot_viewer.write_dot.assert_called_once_with('/path/to/file.dot')

    def test_write_svg(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test that write_svg() delegates to DotGraphViewer.write_svg()."""
        mocked_dot_viewer = mocker.MagicMock()
        mocker.patch.object(HLOViewer, '_dot_graph_viewer', mocked_dot_viewer)

        viewer = HLOViewer('HloModule test')
        viewer.write_svg('/path/to/file.svg')

        mocked_dot_viewer.write_svg.assert_called_once_with('/path/to/file.svg')

    def test_dot_graph_viewer_cached_property(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test that _dot_graph_viewer creates DotGraphViewer from HLO."""
        mocked_hlo_module = mocker.MagicMock()
        mocked_xla = mocker.patch('visu_hlo._display.xla')
        mocked_xla.hlo_module_from_text.return_value = mocked_hlo_module
        mocked_xla.hlo_module_to_dot_graph.return_value = 'digraph { a -> b }'

        viewer = HLOViewer('HloModule test')
        dot_viewer = viewer._dot_graph_viewer

        mocked_xla.hlo_module_from_text.assert_called_once_with('HloModule test')
        mocked_xla.hlo_module_to_dot_graph.assert_called_once_with(mocked_hlo_module)
        assert isinstance(dot_viewer, DotGraphViewer)
        assert dot_viewer.dot_graph == 'digraph { a -> b }'
