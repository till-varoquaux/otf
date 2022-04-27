from __future__ import annotations

import ast
import inspect
import linecache
import pydoc
import types
import typing
from typing import Protocol, TypeVar

locate = pydoc.locate
cram = pydoc.cram


@typing.runtime_checkable
class QualnameAddressable(Protocol):

    __name__: str
    __qualname__: str
    __module__: str


Addressable = TypeVar(
    "Addressable", bound=types.ModuleType | QualnameAddressable
)


def get_locate_name(v: Addressable) -> str:
    """Get a name that can be used with `locate` to reload the given argument"""
    if inspect.ismodule(v):
        name = v.__name__
    elif isinstance(v, QualnameAddressable):
        import otf

        if v.__name__ == "<lambda>":
            raise TypeError("lambdas are not supported")
        if v.__module__ == "builtins":
            name = v.__qualname__
        # Shorten the names of otf's native types and functions if possible
        elif (
            v.__module__.startswith("otf.")
            and otf.__dict__.get(v.__name__, None) == v
        ):
            return f"otf.{v.__name__}"
        else:
            name = v.__module__ + "." + v.__qualname__
        if ".<locals>." in v.__qualname__:
            raise ValueError(
                "values defined inside of functions are not supported."
            )
    else:
        raise TypeError(f"Type {type(v).__name__!r} not supported")

    elt = locate(name)
    if elt is None:
        raise ValueError(
            f"Argument {v} cannot be reloaded via its name: {name!r}"
        )
    elif elt != v:
        raise ValueError(f"Can't use {v}, it's overridden by {elt} as {name!r}")
    return name


def syntax_error(msg: str, filename: str, node: ast.AST) -> typing.NoReturn:
    """Raise a syntax error on a given ast position"""
    # https://github.com/python/cpython/blob/3.10/Objects/exceptions.c#L1474
    raise SyntaxError(
        msg,
        (
            filename,
            node.lineno,
            node.col_offset + 1,
            linecache.getline(filename, node.lineno),
            node.end_lineno,
            node.end_col_offset,
        ),
    )
