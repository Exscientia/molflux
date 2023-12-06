# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "molflux"
# copyright = "2023, Exscientia"
# author = "Exscientia"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx_toolbox.confval",
    "sphinx_togglebutton",
    "myst_nb",
    "sphinx_design",
    "sphinx_sitemap",
    "sphinx_inline_tabs",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_baseurl = "https://exscientia.github.io/molflux/"
html_favicon = "_static/MolFlux_logo_final_MolFlux logo mark.svg"

html_title = "MolFlux"

html_theme_options = {
    "logo": {
        "image_dark": "_static/MolFlux_logo_final_MolFlux Logo mark with name 2 colours.svg",
        "image_light": "_static/MolFlux_logo_final_MolFlux Logo mark with name grey.svg",
    },
    "repository_url": "https://github.com/Exscientia/molflux",
    "use_repository_button": True,
    "use_sidenotes": True,
    "show_nav_level": 0,
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/molflux/",
            "icon": "https://img.shields.io/pypi/v/molflux",
            "type": "url",
        },
    ],
}
