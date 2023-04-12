"""Configuration file for the Sphinx documentation builder."""
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PyFlyt"
html_title = "PyFlyt"
copyright = "2023, Jet"
author = "Jet"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
]

source_suffix = [".rst", ".md"]
templates_path = ["_templates"]
exclude_patterns = ["readme.md"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_toc_level": 5,
    "show_nav_level": 5,
    "show_prev_next": False,
    "announcement": "This repo is still under development. We are also actively looking for users and developers. If this sounds like you, get in touch!",
}
