import ast
import inspect
import linecache
from typing import Any, Callable, NamedTuple

__all__ = ("Function",)


class Position(NamedTuple):
    lineno: int
    col_offset: int


# Adapted from inspect.getsourcelines


class FnFoundException(Exception):
    pass


class _FnFinder(ast.NodeVisitor):
    def __init__(self, qualname):
        self.stack = []
        self.qualname = qualname

    def visit_FunctionDef(self, node):
        self.stack.append(node.name)
        if self.qualname == ".".join(self.stack):
            raise FnFoundException(node)

        self.stack.append("<locals>")
        self.generic_visit(node)
        self.stack.pop()
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()


def _get_lines(object) -> tuple[str, list[str]]:
    filename = inspect.getsourcefile(object)
    if not filename:
        raise OSError("source code not available")
    linecache.checkcache(filename)

    module = inspect.getmodule(object, filename)
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


class Function(NamedTuple):
    name: str
    statements: tuple[ast.stmt, ...]
    filename: str
    lines: tuple[str, ...]
    signature: inspect.Signature
    lineno: int
    col_offset: int

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
            signature=inspect.signature(fn),
        )
