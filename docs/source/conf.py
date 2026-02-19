"""Sphinx configuration file for visu_hlo documentation."""

import sys
from importlib.metadata import version as get_version
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Project information
project = 'visu-hlo'
copyright = '2025, Pierre Chanial'
author = 'Pierre Chanial'
release = get_version('visu-hlo')
version = '.'.join(release.split('.')[:2])  # Major.minor

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'myst_parser',
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = []

# HTML output configuration
html_theme = 'furo'
html_static_path = ['_static']
html_title = f'{project} v{version}'
html_theme_options = {
    'light_css_variables': {
        'color-brand-primary': '#2EBF4F',
        'color-brand-content': '#2EBF4F',
    },
    'dark_css_variables': {
        'color-brand-primary': '#34D058',
        'color-brand-content': '#34D058',
    },
}

# Extension configuration
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
}
autodoc_typehints = 'description'  # Show type hints in parameter descriptions

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# MyST parser configuration
myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'dollarmath',
    'html_admonition',
    'html_image',
    'linkify',
    'replacements',
    'smartquotes',
    'substitution',
    'tasklist',
]

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# nbsphinx configuration
nbsphinx_execute = 'always'  # Execute notebooks during build
nbsphinx_allow_errors = True  # Continue building even if there are errors
nbsphinx_timeout = 300  # Timeout for notebook execution in seconds

# Additional nbsphinx settings - removed unrecognized options
nbsphinx_execute_arguments = []
