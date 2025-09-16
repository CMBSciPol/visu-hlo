"""Shared test configuration and fixtures."""

import jax
import pytest


@pytest.fixture(autouse=True, scope='session')
def setup_jax():
    """Setup JAX for testing: ensure JAX uses CPU for reproducible tests"""
    import os

    os.environ['JAX_PLATFORMS'] = 'cpu'
    jax.config.update('jax_platform_name', 'cpu')
