"""On-the-fly distributed workflows"""
from importlib import metadata

from .compiler import Closure, Environment, Suspension, Workflow
from .decorators import environment, function

# http://epydoc.sourceforge.net/manual-fields.html#module-metadata-variables

# https://packaging.python.org/en/latest/guides/single-sourcing-package-version/
__version__ = metadata.version(__name__)
__author__ = "Till Varoquaux <till.varoquaux@gmail.com>"
# https://www.copyrightlaws.com/copyright-symbol-notice-year
__copyright__ = f"2022, {__author__}"

__all__ = (
    "Closure",
    "Environment",
    "Suspension",
    "Workflow",
    "environment",
    "function",
)
