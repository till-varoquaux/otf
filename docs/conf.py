"""Sphinx configuration."""
from __future__ import annotations

import os
import sys

import otf

sys.path.append(os.path.abspath("ext"))

project = "otf"
author = otf.__author__
copyright = otf.__copyright__
release = version = otf.__version__

extensions = [
    "asdl_highlight",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.graphviz",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinxotf",
]

autodoc_typehints = "none"
autodoc_member_order = "bysource"

doctest_test_doctest_blocks = ""

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

graphviz_output_format = "svg"

html_theme = "furo"

todo_include_todos = True
