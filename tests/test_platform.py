"""Tests for platform detection functionality."""

import importlib

import pytest
import pytest_mock
import visu_hlo


def test_linux_platform(mocker: pytest_mock.MockerFixture) -> None:
    """Test Linux platform detection."""
    mocker.patch('platform.platform', return_value='Linux-5.4.0-42-generic-x86_64-with-glibc2.31')
    importlib.reload(visu_hlo)
    assert visu_hlo.DISPLAY_PROGRAM == 'xdg-open'


def test_darwin_platform(mocker: pytest_mock.MockerFixture) -> None:
    """Test macOS platform detection."""
    mocker.patch('platform.platform', return_value='Darwin')
    importlib.reload(visu_hlo)
    assert visu_hlo.DISPLAY_PROGRAM == 'open'


def test_windows_platform(mocker: pytest_mock.MockerFixture) -> None:
    """Test Windows platform detection."""
    mocker.patch('platform.platform', return_value='Windows-10-10.0.19041-SP0')
    importlib.reload(visu_hlo)
    assert visu_hlo.DISPLAY_PROGRAM == 'start'


def test_unsupported_platform(mocker: pytest_mock.MockerFixture) -> None:
    """Test handling of unsupported platforms."""
    mocker.patch('platform.platform', return_value='UnsupportedOS')
    with pytest.raises(RuntimeError, match='Unsupported platform'):
        importlib.reload(visu_hlo)
