import ast
import dataclasses
import inspect
import linecache
from typing import Any, Callable, TypedDict

#
# This module contains code adapted from cpython's inspect.py. This code was
# marked as public domain code.

__all__ = ("Function",)


@dataclasses.dataclass(frozen=True, slots=True)
class Position:
    lineno: int
    col_offset: int


# Adapted from inspect.getsourcelines


class FnFoundException(Exception):
    pass


class _FnFinder(ast.NodeVisitor):
    qualname: str
    stack: list[str]

    def __init__(self, qualname: str) -> None:
        self.stack = []
        self.qualname = qualname

    def visit_FunctionDef(
        self, node: ast.AsyncFunctionDef | ast.FunctionDef
    ) -> None:
        self.stack.append(node.name)
        if self.qualname == ".".join(self.stack):
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


def get_end_pos(node: ast.stmt) -> Position:
    assert node.end_lineno is not None and node.end_col_offset is not None
    return Position(node.end_lineno, node.end_col_offset)


def get_start_pos(node: ast.stmt) -> Position:
    assert node.lineno is not None and node.col_offset is not None
    return Position(node.lineno, node.col_offset)


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
    """The mandatory argumments for ExplodedSignatureDict"""

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

    # There's a bug in coverage that prevents it from detecting that those lines
    # are covered
    if kwdefaults:  # pragma: no cover
        result["kwdefaults"] = kwdefaults

    if defaults:  # pragma: no cover
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


@dataclasses.dataclass(frozen=True)
class Function:
    name: str
    filename: str
    lineno: int
    col_offset: int
    lines: tuple[str, ...]
    signature: inspect.Signature
    statements: tuple[ast.stmt, ...]

    @classmethod
    def from_function(cls, fn: Callable[..., Any]) -> "Function":
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
        fn_finder = _FnFinder(qualname)
        try:
            fn_finder.visit(tree)
        except FnFoundException as e:
            node = e.args[0]
        else:
            raise ValueError(
                f"Could not find function definition for: {qualname!r}"
            )

        end_pos = get_end_pos(node)
        start_pos = get_start_pos(node.body[0])
        fn_lines = lines[start_pos.lineno - 1 : end_pos.lineno]
        # TODO: look backward for comments?
        # Trim the col_offsets
        fn_lines[-1] = fn_lines[-1][: end_pos.col_offset]
        fn_lines[0] = fn_lines[0][start_pos.col_offset :]
        return cls(
            name=fn.__name__,
            statements=tuple(node.body),
            filename=filename,
            lineno=start_pos.lineno,
            col_offset=start_pos.col_offset,
            lines=tuple(fn_lines),
            signature=_get_signature(fn),
        )
