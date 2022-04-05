"""Sphinx configuration."""
import os
import sys

import otf

sys.path.append(os.path.abspath("ext"))

project = "otf"
author = otf.__author__
copyright = otf.__copyright__
release = version = otf.__version__

extensions = [
    "sphinxotf",
    "sphinx.ext.intersphinx",
    "sphinx.ext.graphviz",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

graphviz_output_format = "svg"
