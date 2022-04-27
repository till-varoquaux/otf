from __future__ import annotations

import ast
import hashlib
import io
import linecache
import math
import typing
from typing import Protocol


class Located(Protocol):  # pragma: no cover
    @property
    def lineno(self) -> int:
        ...

    @property
    def col_offset(self) -> int:
        ...

    @property
    def end_lineno(self) -> int | None:
        ...

    @property
    def end_col_offset(self) -> int | None:
        ...


# We use "anchors" to generate code: anchor are node whose position we use in
# the code generation. Their only purpose is to make sure that errors get
# reported at a sensible location. Sometimes we do not care about positions
# either because there is no risk of an error being thrown in the generated code
# or because we do not have a node to tie the generated code too. This is the
# default anchor we use for these cases.
ANCHOR = ast.Pass(
    lineno=0,
    col_offset=0,
    end_lineno=0,
    end_col_offset=0,
)


def _const(
    value: int | float | None | bool | str | bytes, anchor: Located = ANCHOR
) -> ast.expr:
    return ast.Constant(
        kind=None,
        value=value,
        lineno=anchor.lineno,
        col_offset=anchor.col_offset,
        end_lineno=anchor.end_lineno,
        end_col_offset=anchor.col_offset,
    )


def neg(value: ast.expr) -> ast.expr:
    return ast.UnaryOp(
        op=ast.USub(),
        operand=value,
        lineno=value.lineno,
        col_offset=value.col_offset,
        end_lineno=value.end_lineno,
        end_col_offset=value.col_offset,
    )


def constant(
    value: int | float | None | bool | str | bytes, anchor: Located = ANCHOR
) -> ast.expr:
    "Smart constructor for ast.Constant"
    if isinstance(value, bool | str | bytes | None):
        return _const(value, anchor=anchor)
    assert isinstance(value, int | float)
    assert math.isfinite(value)
    # Handle -0. and 0. properly
    if math.copysign(1, value) == 1:
        return _const(value, anchor=anchor)
    # python 3.10 parses negative int constants (e.g.: in ``case -1:...``) as
    # the ``-`` operator applied to a positive int. We want our generated ast to
    # match what would have been produced by the parser.
    return neg(_const(-value, anchor=anchor))


def name(
    id: str,
    ctx: ast.Load | ast.Store | ast.Del = ast.Load(),
    anchor: Located = ANCHOR,
) -> ast.Name:
    "smart constructor for ast.Name"
    assert id.isidentifier(), id
    return ast.Name(
        id=id,
        ctx=ctx,
        lineno=anchor.lineno,
        col_offset=anchor.col_offset,
        end_lineno=anchor.end_lineno,
        end_col_offset=anchor.end_col_offset,
    )


def dotted_path(
    path: str,
    ctx: ast.Load | ast.Store | ast.Del = ast.Load(),
    anchor: Located = ANCHOR,
) -> ast.Name | ast.Attribute:
    """Takes a dotted path and compile it to a python expression"""
    root, *rest = path.split(".")
    res: ast.Name | ast.Attribute = name(root, ctx=ast.Load(), anchor=anchor)
    for x in rest:
        res = ast.Attribute(
            value=res,
            attr=x,
            ctx=ast.Load(),
            lineno=anchor.lineno,
            col_offset=anchor.col_offset,
            end_lineno=anchor.end_lineno,
            end_col_offset=anchor.end_col_offset,
        )
    res.ctx = ctx
    return res


def _arg_to_expr(
    value: int | float | None | bool | str | bytes | ast.expr,
    anchor: Located = ANCHOR,
) -> ast.expr:
    if isinstance(value, ast.expr):
        return value
    return constant(value, anchor=anchor)


def call(
    func: str | ast.expr, *args: ast.expr, anchor: Located = ANCHOR
) -> ast.expr:
    """Smart constructor for ast.Call"""
    return ast.Call(
        func=dotted_path(func, anchor=anchor)
        if isinstance(func, str)
        else func,
        args=[_arg_to_expr(arg) for arg in args],
        keywords=[],
        lineno=anchor.lineno,
        col_offset=anchor.col_offset,
        end_lineno=anchor.end_lineno,
        end_col_offset=anchor.end_col_offset,
    )


# We fill the linecache with the content of the functions to make backtrace work
# well.
#
# Both doctest and ipython patch linecache to handle "fake files":
# + https://github.com/python/cpython/blob/26fa25a9a73/Lib/doctest.py#L1427
# + https://github.com/ipython/ipython/blob/b9c1adb1119/IPython/core
#   /compilerop.py#L189
#
# We probably don't want to do the same: this would cause conflicts between our
# code and theirs.
# Instead we use a mtime of None, which means our entries won't be purged by
# linecache.checkcache.
#
# If this gets used a lot we'll experience memory leak. We might want to use
# some handle with a weak references to make sure we clear the cache of entries
# we do not need


def fill_linecache(data: str) -> str:
    "Fill the linecache with a fake file containing the content of ``data``."
    digest = hashlib.sha1(data.encode("utf8")).hexdigest()
    filename = f"<otf-{digest}>"
    if filename not in linecache.cache:
        size = len(data)
        lines = list(io.StringIO(data))
        # An mtime == None means that this won't be purged by
        # linecache.checkcache
        mtime = None
        linecache.cache[filename] = (
            size,
            mtime,
            lines,
            filename,
        )
    return filename


def raise_at(
    exc: Exception, node: Located, content: str, *, filename: str | None = None
) -> typing.NoReturn:
    """Raise an exception for a given location in a string.

    This is a helper function to make sure we get traceback that go up all the
    way to their real source.
    """
    code = compile(
        ast.Module(
            body=[
                ast.Raise(
                    exc=name("e", anchor=node),
                    cause=None,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                    end_lineno=node.end_lineno,
                    end_col_offset=node.end_col_offset,
                )
            ],
            type_ignores=[],
        ),
        filename=fill_linecache(content) if filename is None else filename,
        mode="exec",
    )
    exec(code, {"e": exc})
    # Mypy fails to infer that this is unreachable
    assert False  # pragma: no cover
