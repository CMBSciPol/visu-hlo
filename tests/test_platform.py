"""Tests for platform detection functionality."""

import importlib
from collections.abc import Iterator

import pytest
import pytest_mock

import visu_hlo._display as display_module


@pytest.fixture(autouse=True, scope='module')
def cleanup_module() -> Iterator[None]:
    yield
    importlib.reload(display_module)


def test_linux_platform(mocker: pytest_mock.MockerFixture) -> None:
    """Test Linux platform detection."""
    mocker.patch('platform.system', return_value='Linux-5.4.0-42-generic-x86_64-with-glibc2.31')
    importlib.reload(display_module)
    assert display_module.DISPLAY_PROGRAM == 'xdg-open'


def test_darwin_platform(mocker: pytest_mock.MockerFixture) -> None:
    """Test macOS platform detection."""
    mocker.patch('platform.platform', return_value='Darwin')
    importlib.reload(display_module)
    assert display_module.DISPLAY_PROGRAM == 'open'


def test_windows_platform(mocker: pytest_mock.MockerFixture) -> None:
    """Test Windows platform detection."""
    mocker.patch('platform.platform', return_value='Windows-10-10.0.19041-SP0')
    importlib.reload(display_module)
    assert display_module.DISPLAY_PROGRAM == 'start'


def test_unsupported_platform(mocker: pytest_mock.MockerFixture) -> None:
    """Test handling of unsupported platforms."""
    mocker.patch('platform.platform', return_value='UnsupportedOS')
    with pytest.raises(RuntimeError, match='Unsupported platform'):
        importlib.reload(display_module)
