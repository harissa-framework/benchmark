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
    'sphinxcontrib.mermaid',
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
myst_fence_as_directive = {'mermaid'}
mermaid_version ="10.9.1"
mermaid_init_js = "mermaid.initialize({startOnLoad:true, theme: 'neutral', legacyMathML: true});"
# mermaid_d3_zoom = True

templates_path = ['_templates']
exclude_patterns = []

intersphinx_mapping = {
    'harissa-framework': ('https://github.com/harissa-framework/', None),
}

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

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
  'icon_links': [
        {
            'name': 'GitHub',
            'url': "https://github.com/harissa-framework/benchmark",
            'icon': 'fa-brands fa-github',
            'type': 'fontawesome',
        },
        {
            'name': 'Harissa',
            'url': 'https://harissa-framework.github.io/harissa/',
            'icon': 'fa-solid fa-pepper-hot',
            'type': 'fontawesome',
        }
   ],
   "pygment_light_style": "default",
   "pygment_dark_style": "material",
}
html_static_path = ['_static']
html_css_files = ['custom.css']
html_copy_source = False
