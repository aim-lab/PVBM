# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import jupyter_core.paths

sys.path.insert(0, os.path.abspath('..'))

project = 'PVBM'
copyright = '2023, JF'
author = 'JF'
release = '3.0.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'sphinx.ext.mathjax',
]

nbsphinx_execute = 'never'
source_dir = 'docs'

# Add Jupyter's default template path
template_paths = jupyter_core.paths.jupyter_path('nbconvert/templates')

if not any('rst' in os.listdir(path) for path in template_paths if os.path.exists(path)):
    raise ValueError("No template sub-directory with name 'rst' found in Jupyter's template paths.")

# Now add these template paths to Sphinx
templates_path = ['_templates'] + template_paths

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
modindex_common_prefix = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
