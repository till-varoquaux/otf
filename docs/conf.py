"""Sphinx configuration."""
import otf

project = "otf"
author = otf.__author__
copyright = otf.__copyright__
release = version = otf.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]
