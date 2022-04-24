"""

:mod:`~otf.pack` is a layer that is intended to be used on top of existing
serialisation libraries. It converts arbitrary python values to a simple human
readable format.

What differentiate :mod:`~otf.pack` from other serialisation libraries is that
it is very explicit. Out of the box :mod:`~otf.pack` only handle a small set of
types; we don't rely on inheritance or the runtime structure of values to handle
anything that isn't of a supported type. Support for new types can be added via
:func:`register`.

Supported types:
----------------

Out of the box, the types that are supported are:

+ :class:`str`, :class:`bytes`, :class:`int`, :class:`float`, :class:`bool`, \
    :const:`None`: Basic python primitives
+ :class:`list`: where all the elements are serialisable
+ :class:`dict`: where all the keys and values are serialisable
+ :class:`tuple`: where all the elements are serialisable
+ **shared references**: but not recursive values

API:
----

"""
from __future__ import annotations

from typing import Final

from .base import copy, register
from .text import Format, dumps, loads

#:
COMPACT: Final = Format.COMPACT

#:
PRETTY: Final = Format.PRETTY

#:
EXECUTABLE: Final = Format.EXECUTABLE

__all__ = (
    "dumps",
    "loads",
    "copy",
    "register",
    "COMPACT",
    "PRETTY",
    "EXECUTABLE",
)
