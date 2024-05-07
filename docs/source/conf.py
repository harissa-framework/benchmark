# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from pathlib import Path

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Benchmark'
copyright = '2024, Ulysee Herbach'
author = 'Ulysee Herbach'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinxcontrib.collections',
    'myst_nb',
    'sphinx_copybutton',
]

myst_enable_extensions = [
    # 'amsmath',
    'dollarmath',
    'colon_fence',
    'strikethrough',
    'tasklist',
]

templates_path = ['_templates']
exclude_patterns = []

collections = {
        'notebooks' : {
            'driver': 'copy_folder',
            'source': str(Path(__file__).parent.parent.parent / 'notebooks'),
            'ignore': ['.ipynb_checkpoints'],
        }
    }

autodoc_typehints = 'description'
autosummary_generate = True
autosummary_ignore_module_all = False
napoleon_google_docstring = False
napoleon_numpy_docstring = True

nb_execution_mode = 'off'
copybutton_exclude = '.linenos, .gp, .go, .o'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_copy_source = False
