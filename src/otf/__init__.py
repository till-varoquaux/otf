"""On-the-fly distributed workflows"""
from __future__ import annotations

from importlib import metadata

# This import is used to get short names from utils.get_locate_name
from .compiler import Function  # noqa: F401
from .compiler import Closure, Environment, Suspension, Workflow
from .decorators import environment, function
from .pack import COMPACT, EXECUTABLE, PRETTY, dump_text, load_text, register
from .runtime import NamedReference

# http://epydoc.sourceforge.net/manual-fields.html#module-metadata-variables

# https://packaging.python.org/en/latest/guides/single-sourcing-package-version/
__version__ = metadata.version(__name__)
__author__ = "Till Varoquaux <till.varoquaux@gmail.com>"
# https://www.copyrightlaws.com/copyright-symbol-notice-year
__copyright__ = f"2022, {__author__}"

__all__ = (
    "Closure",
    "Environment",
    "NamedReference",
    "Suspension",
    "Workflow",
    "environment",
    "function",
    "load_text",
    "dump_text",
    "register",
    "COMPACT",
    "PRETTY",
    "EXECUTABLE",
)
