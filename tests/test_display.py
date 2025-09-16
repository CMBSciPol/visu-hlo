"""Tests for SVG display functionality."""

import subprocess

import pytest
import pytest_mock
import visu_hlo


def test_display_svg_with_content(mocker: pytest_mock.MockerFixture) -> None:
    """Test SVG display with complex SVG content."""
    mocked_tempfile = mocker.patch('visu_hlo.NamedTemporaryFile')
    mocked_tempfile.return_value.__enter__.return_value.name = '/tmp/complex.svg'
    mocked_write_text = mocker.patch('visu_hlo.Path.write_text')
    mocked_subprocess = mocker.patch('visu_hlo.subprocess.run')

    svg_content = """
    <svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
        <circle cx="100" cy="100" r="50" fill="red"/>
        <text x="100" y="105" text-anchor="middle">Test</text>
    </svg>
    """

    visu_hlo._display_svg(svg_content)

    mocked_tempfile.assert_called_once_with(suffix='.svg', delete=False)
    mocked_write_text.assert_called_once_with(svg_content)
    mocked_subprocess.assert_called_once_with([visu_hlo.DISPLAY_PROGRAM, '/tmp/complex.svg'])


def test_display_svg_handles_subprocess_error(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test handling of subprocess errors when opening SVG file."""
    mocked_tempfile = mocker.patch('visu_hlo.NamedTemporaryFile')
    mocked_tempfile.return_value.__enter__.return_value.name = '/tmp/error.svg'
    mocker.patch('visu_hlo.Path.write_text')
    mocker.patch('visu_hlo.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'cmd'))

    svg_content = '<svg>error test</svg>'

    # Should raise the subprocess error
    with pytest.raises(subprocess.CalledProcessError):
        visu_hlo._display_svg(svg_content)


def test_display_svg_handles_file_write_error(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test handling of file write errors."""
    mocked_tempfile = mocker.patch('visu_hlo.NamedTemporaryFile')
    mocked_tempfile.return_value.__enter__.return_value.name = '/tmp/write_error.svg'
    mocker.patch('visu_hlo.Path.write_text', side_effect=OSError('Cannot write file'))
    mocker.patch('visu_hlo.subprocess.run')

    svg_content = '<svg>error test</svg>'

    # Should raise the IOError
    with pytest.raises(IOError, match='Cannot write file'):
        visu_hlo._display_svg(svg_content)
