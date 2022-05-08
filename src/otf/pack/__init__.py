"""

:mod:`~otf.pack` is a layer that is intended to be used on top of existing
serialisation libraries. It converts arbitrary python values to a simple human
readable format.

What differentiate :mod:`~otf.pack` from other serialisation libraries is that
it is very explicit. Out of the box :mod:`~otf.pack` only handle a small set of
types; we don't rely on inheritance or the runtime structure of values to handle
anything that isn't of a supported type. Support for new types can be added via
:func:`register`.

Supported types
---------------

Out of the box, the types that are supported are:

+ :class:`str`, :class:`bytes`, :class:`int`, :class:`float`, :class:`bool`, \
    :const:`None`: Basic python primitives
+ :class:`list`: where all the elements are serialisable
+ :class:`dict`: where all the keys and values are serialisable
+ :class:`tuple`: where all the elements are serialisable
+ :class:`set`: where all the elements are serialisable
+ **shared references**: but not recursive values

"""
from __future__ import annotations

from typing import Final

from .base import RuntimeValueBuilder, copy, reduce_runtime_value, register
from .bin import BinPacker, dis, dump_bin, load_bin, reduce_bin
from .text import (
    CompactPrinter,
    ExecutablePrinter,
    Format,
    PrettyPrinter,
    dump_text,
    load_text,
    reduce_text,
)

#: Print the value on one line
COMPACT: Final = Format.COMPACT

#: Pretty print the value
PRETTY: Final = Format.PRETTY

#: Print the value as python code where the last statement is the value.
EXECUTABLE: Final = Format.EXECUTABLE

__all__ = (
    "COMPACT",
    "PRETTY",
    "EXECUTABLE",
    "dump_text",
    "load_text",
    "dump_bin",
    "load_bin",
    "register",
    "reduce_runtime_value",
    "reduce_bin",
    "reduce_text",
    "RuntimeValueBuilder",
    "CompactPrinter",
    "PrettyPrinter",
    "ExecutablePrinter",
    "BinPacker",
    "copy",
    "dis",
)
