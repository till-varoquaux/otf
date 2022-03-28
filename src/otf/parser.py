import ast
import atexit
import contextlib
import dataclasses
import hashlib
import inspect
import linecache
import os
import re
import tempfile
import typing
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    ParamSpec,
    TypedDict,
    TypeVar,
)

#
# This module contains code adapted from cpython's inspect.py. This code was
# marked as public domain code.

__all__ = ("Function",)


T = TypeVar("T")
P = ParamSpec("P")


@dataclasses.dataclass(frozen=True, slots=True)
class Position:
    lineno: int
    col_offset: int


# Adapted from inspect.getsourcelines


class FnFoundException(Exception):
    pass


class _FnFinder(ast.NodeVisitor):
    lineno: int  # The function should be defined at or below that line
    qualname: str
    stack: list[str]

    def __init__(self, qualname: str, lineno: int) -> None:
        self.stack = []
        self.lineno = lineno
        self.qualname = qualname

    def visit_FunctionDef(
        self, node: ast.AsyncFunctionDef | ast.FunctionDef
    ) -> None:
        self.stack.append(node.name)
        if (
            self.qualname == ".".join(self.stack)
            and node.body[0].lineno >= self.lineno
        ):
            raise FnFoundException(node)

        self.stack.append("<locals>")
        self.generic_visit(node)
        self.stack.pop()
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()


def _get_lines(fn: Callable[..., Any]) -> tuple[str, list[str]]:
    filename = inspect.getsourcefile(fn)
    if not filename:
        raise OSError("source code not available")
    linecache.checkcache(filename)

    module = inspect.getmodule(fn, filename)
    lines = linecache.getlines(
        filename, module.__dict__ if module is not None else None
    )
    if not lines:
        raise OSError("could not get source code")
    return filename, lines


def _get_signature(fn: Callable[..., Any]) -> inspect.Signature:
    """Get the signature of a given function

    Since OTF doesn't currently do anything smart with the annotations we strip
    them out.

    """
    sig = inspect.signature(fn)
    return inspect.Signature(
        [
            param.replace(annotation=param.empty)
            for param in sig.parameters.values()
        ]
    )


class _ExplodedSignatureDictBase(TypedDict, total=True):
    """The mandatory arguments for ExplodedSignatureDict"""

    # Once PEP 655 gets accepted we'll be able to merge this and
    # ExplodedSignatureDict in one class

    args: list[str]


class ExplodedSignatureDict(_ExplodedSignatureDictBase, total=False):
    defaults: list[Any]
    kwdefaults: dict[str, Any]


ExplodedSignature = ExplodedSignatureDict | list[str]


# adapted from Signature.__str__
def _explode_signature(sig: inspect.Signature) -> ExplodedSignature:
    """Convert a signature into a serializable object"""
    # A description of the syntax can be found in:
    # https://www.python.org/dev/peps/pep-0570/
    defaults = []
    kwdefaults = {}
    args = []
    render_pos_only_separator = False
    render_kw_only_separator = True
    assert sig.return_annotation == inspect.Signature.empty
    for param in sig.parameters.values():
        formatted = param.name
        kind = param.kind
        assert param.annotation == param.empty, param

        if kind == inspect.Parameter.POSITIONAL_ONLY:
            render_pos_only_separator = True
        elif render_pos_only_separator:
            # It's not a positional-only parameter, and the flag
            # is set to 'True' (there were pos-only params before.)
            args.append("/")
            render_pos_only_separator = False

        if kind == inspect.Parameter.VAR_POSITIONAL:
            # OK, we have an '*args'-like parameter, so we won't need
            # a '*' to separate keyword-only arguments
            formatted = "*" + param.name
            render_kw_only_separator = False
        elif kind == inspect.Parameter.VAR_KEYWORD:
            # This should be the last keyword: we don't care about
            # render_kw_only_separator
            formatted = "**" + param.name
        elif (
            kind == inspect.Parameter.KEYWORD_ONLY and render_kw_only_separator
        ):
            # We have a keyword-only parameter to render and we haven't
            # rendered an '*args'-like parameter before, so add a '*'
            # separator to the parameters list ("foo(arg1, *, arg2)" case)
            args.append("*")
            # This condition should be only triggered once, so
            # reset the flag
            render_kw_only_separator = False

        if param.default is not param.empty:
            if kind == inspect.Parameter.KEYWORD_ONLY:
                kwdefaults[param.name] = param.default
            else:
                defaults.append(param.default)
        args.append(formatted)

    if render_pos_only_separator:
        # There were only positional-only parameters, hence the
        # flag was not reset to 'False'
        args.append("/")

    if kwdefaults == {} and defaults == []:
        return args
    result: ExplodedSignatureDict = {
        "args": args,
    }

    if kwdefaults:
        result["kwdefaults"] = kwdefaults

    if defaults:
        result["defaults"] = defaults

    return result


def _implode_signature(
    exploded: ExplodedSignature,
) -> inspect.Signature:
    """Turn a serializable object into a Signature"""
    if isinstance(exploded, list):
        args = exploded
        defaults = []
        kwdefaults = {}
    else:
        args = exploded["args"]
        defaults = exploded.get("defaults", [])
        kwdefaults = exploded.get("kwdefaults", {})
    acc = []
    num_pos = 0
    # Sadly this is a private type..
    kind: inspect._ParameterKind
    if "/" in args:
        kind = inspect.Parameter.POSITIONAL_ONLY
    else:
        kind = inspect.Parameter.POSITIONAL_OR_KEYWORD

    for arg in args:
        assert kind < inspect.Parameter.VAR_KEYWORD
        if arg == "/":
            assert kind == inspect.Parameter.POSITIONAL_ONLY
            kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
        elif arg == "*":
            assert kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            kind = inspect.Parameter.KEYWORD_ONLY
        elif arg.startswith("**"):
            kind = inspect.Parameter.VAR_KEYWORD
            name = arg[2:]
            assert name.isidentifier()
            acc.append(inspect.Parameter(name=name, kind=kind))
        elif arg.startswith("*"):
            assert kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            name = arg[1:]
            assert name.isidentifier()
            acc.append(
                inspect.Parameter(
                    name=name, kind=inspect.Parameter.VAR_POSITIONAL
                )
            )
            kind = inspect.Parameter.KEYWORD_ONLY
        else:
            assert arg.isidentifier()
            default = inspect.Parameter.empty
            if kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                num_pos += 1
            else:
                # assert kind == inspect.Parameter.KEYWORD_ONLY, kind
                default = kwdefaults.get(arg, inspect.Parameter.empty)
            acc.append(inspect.Parameter(name=arg, kind=kind, default=default))

    val: Any
    for offset, val in enumerate(reversed(defaults)):
        idx = num_pos - offset - 1
        acc[idx] = acc[idx].replace(default=val)

    return inspect.Signature(acc)


def _not_none(x: Optional[T]) -> T:
    assert x is not None
    return x


class ExplodedFunction(TypedDict, total=True):
    name: str
    signature: ExplodedSignature
    body: str


@dataclasses.dataclass(frozen=True)
class Function(Generic[P, T]):
    """The string representation of the function in body is constructed to keep all
    the preserve the positions between the node from the original source. It is
    not necessarily valid python::

    >>> def f(): return 5
    >>> ff = Function.from_function(f)
    >>> ff.body
    '...: return 5'
    """

    name: str
    filename: str
    signature: inspect.Signature
    statements: tuple[ast.stmt, ...]

    @property
    def lineno(self) -> int:
        """First line of the body of the function in the original file."""
        return _not_none(self.statements[0].lineno)

    @property
    def col_offset(self) -> int:
        return _not_none(self.statements[0].col_offset)

    @property
    def end_lineno(self) -> int:
        """Last line of the body of the function in the original file."""

        return _not_none(self.statements[-1].end_lineno)

    @property
    def end_col_offset(self) -> int:
        return _not_none(self.statements[-1].end_col_offset)

    @property
    def body(self) -> str:
        """A string representation of the function.

        The return value is constructed to preserve the positions between the
        node from the original source and make it easy to map an error back to
        its source in the original file. It is not necessarily valid python:

        >>> def f(): return 5
        >>> ff = Function.from_function(f)
        >>> ff.body
        '  ...: return 5'

        """

        lines = linecache.getlines(self.filename)

        # Can we use the info from the AST in statements?
        fn_lines = lines[self.lineno - 1 : self.end_lineno]
        fn_lines[-1] = fn_lines[-1][: self.end_col_offset]

        prelude = fn_lines[0][: self.col_offset]
        if not prelude.isspace():
            fn_lines[0] = (
                "...: ".rjust(len(prelude)) + fn_lines[0][self.col_offset :]
            )
        return "".join(fn_lines).rstrip()

    @classmethod
    def from_function(cls, fn: Callable[P, T]) -> "Function[P, T]":
        """Construct from an existing python function"""
        # We could use a naive:
        #
        # ast.parse(textwrap.dedent("\n".join(inspect.getsourcelines(...)))
        #
        # but we'd mess indentation for multiline strings in nested function.
        if not inspect.isfunction(fn):
            raise TypeError(
                f"Argument is not a function: {fn} of type {type(fn)}"
            )
        if fn.__name__ == "<lambda>":
            raise TypeError("lambdas not supported")
        filename, lines = _get_lines(fn)
        qualname = fn.__qualname__
        source = "".join(lines)
        tree = ast.parse(source)
        fn_finder = _FnFinder(qualname, lineno=fn.__code__.co_firstlineno)
        try:
            fn_finder.visit(tree)
        except FnFoundException as e:
            node = e.args[0]
        else:
            raise ValueError(
                f"Could not find function definition for: {qualname!r}"
            )

        return cls(
            name=fn.__name__,
            statements=tuple(node.body),
            filename=filename,
            signature=_get_signature(fn),
        )

    # Make sure that pickle uses our explode/implode code.
    def __getstate__(self) -> ExplodedFunction:
        return _explode_function(self)

    def __setstate__(self, exploded: "ExplodedFunction") -> None:
        other = _implode_function(exploded)
        # Getting around "Frozen"... we could use a __reduce__ instead. Either
        # way, things get a bit messy when pickle is involved.
        object.__setattr__(self, "__dict__", other.__dict__)


class _DotDotDot:
    """A class that prints out as ...

    Sadly python's ... does not behave how we'd want it to:

    >>> str(...)
    'Ellipsis'


    So we have to go and create our own class:

    >>> str(_DotDotDot())
    '...'

    """

    def __str__(self) -> str:
        return "..."

    __repr__ = __str__


DOT_DOT_DOT = _DotDotDot()


def _explode_function(fn: Function[Any, Any]) -> ExplodedFunction:
    return {
        "name": fn.name,
        "signature": _explode_signature(fn.signature),
        "body": fn.body,
    }


def _gen_imploded_function_str(
    body: str, name: str, signature: inspect.Signature
) -> str:
    simplified_sig = inspect.Signature(
        [
            param.replace(default=_DotDotDot())
            if param.default is not param.empty
            else param
            for param in signature.parameters.values()
        ]
    )
    sig_str = str(simplified_sig)
    assert sig_str[0] == "(" and sig_str[-1] == ")"
    args_str = sig_str[1:-1]
    m = re.match(r" *\.\.\.:", body)
    header = f"def {name}({args_str}):"
    if m is None:
        return f"{header}\n{body}\n"
    matchlen = len(m.group(0))
    return f"{header}{body[matchlen:]}\n"


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

_SOURCE_DIR: Optional[tempfile.TemporaryDirectory[str]] = None


def _cleanup() -> None:
    global _SOURCE_DIR
    if _SOURCE_DIR is not None:
        _SOURCE_DIR.cleanup()
        _SOURCE_DIR = None


def _fill_linecache(data: str) -> str:
    global _SOURCE_DIR
    if _SOURCE_DIR is None:
        _SOURCE_DIR = tempfile.TemporaryDirectory(prefix="otf_py_srcs")
        atexit.register(_cleanup)
    digest = hashlib.sha1(data.encode("utf8")).hexdigest()
    filename = f"{digest}.py"
    path = os.path.join(_SOURCE_DIR.name, filename)
    with contextlib.suppress(FileExistsError), open(path, mode="xt") as fp:
        fp.write(data)
    return path


def _gen_function(
    name: str, body: str, signature: inspect.Signature
) -> Function[Any, Any]:
    fun_str = _gen_imploded_function_str(
        name=name, body=body, signature=signature
    )
    filename = _fill_linecache(fun_str)
    tree = ast.parse(fun_str)
    statements = typing.cast(ast.FunctionDef, tree.body[0]).body
    return Function(
        name=name,
        filename=filename,
        signature=signature,
        statements=tuple(statements),
    )


def _implode_function(exploded: ExplodedFunction) -> Function[Any, Any]:
    signature = _implode_signature(exploded["signature"])
    return _gen_function(
        name=exploded["name"], body=exploded["body"], signature=signature
    )
