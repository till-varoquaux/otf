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
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.graphviz",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinxotf",
]

autodoc_typehints = "none"
autodoc_member_order = "bysource"

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

graphviz_output_format = "svg"
