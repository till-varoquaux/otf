"""
``otf.pack.text``: Human readable serialisation
================================================


"""

from __future__ import annotations

import ast
import enum
import heapq
import inspect
import math
import typing
from typing import (
    Any,
    Callable,
    ClassVar,
    Final,
    Iterable,
    Iterator,
    ParamSpec,
    TypeVar,
)

from otf import ast_utils, pretty, utils

from . import base

P = ParamSpec("P")
T = TypeVar("T")

__all__ = (
    "dumps",
    "loads",
    "reduce",
    "Simple",
    "Prettyfier",
    "ModulePrinter",
    "Format",
)


class Format(enum.Enum):
    """Which format to use for :func:`dumps`"""

    #: Print the value on one line with no breaks.
    COMPACT = enum.auto()

    #: Pretty print the value.
    PRETTY = enum.auto()

    #: Print the value as python code where the last statement is the value.
    EXECUTABLE = enum.auto()


COMPACT: Final = Format.COMPACT
PRETTY: Final = Format.PRETTY
EXECUTABLE: Final = Format.EXECUTABLE


COL_SEP = pretty.text(",") + pretty.BREAK
NULL_BREAK = pretty.break_with("")


class Simple(base.Accumulator[pretty.Doc]):
    "Serialize a value as a human readable text."

    NAN: ClassVar[pretty.Doc] = pretty.text("nan")
    INFINITY: ClassVar[pretty.Doc] = pretty.text("inf")

    def __init__(self, indent: int = 4) -> None:
        self.indent = indent

    def format_list(
        self,
        docs: Iterable[pretty.Doc],
        *,
        sep: pretty.Doc = COL_SEP,
        opar: str = "(",
        cpar: str = ")",
        gmode: pretty.Mode = pretty.Mode.AUTO,
    ) -> pretty.Doc:
        acc = NULL_BREAK
        first = True
        for doc in docs:
            if not first:
                acc += sep
            else:
                first = False
            acc += doc
        if first:
            return pretty.text(opar + cpar)
        body = pretty.nest(self.indent, acc) + NULL_BREAK
        return pretty.DocGroup(
            gmode, pretty.text(opar) + body + pretty.text(cpar)
        )

    def constant(
        self, constant: int | float | None | str | bytes | bool
    ) -> pretty.Doc:
        if isinstance(constant, float) and not math.isfinite(constant):
            if math.isnan(constant):
                return self.NAN
            if constant == math.inf:
                return self.INFINITY
            assert constant == -math.inf
            return pretty.text("-") + self.INFINITY
        return pretty.text(repr(constant))

    def mapping(
        self, items: Iterator[tuple[pretty.Doc, pretty.Doc]]
    ) -> pretty.Doc:
        return self.format_list(
            (k + pretty.text(": ") + v for k, v in items), opar="{", cpar="}"
        )

    def sequence(self, items: Iterator[pretty.Doc]) -> pretty.Doc:
        return self.format_list(items, opar="[", cpar="]")

    def reference(self, offset: int) -> pretty.Doc:
        return pretty.text(f"ref({offset:_d})")

    def custom(
        self, constructor: str, value: Iterator[pretty.Doc]
    ) -> pretty.Doc:
        (arg,) = value
        return pretty.agrp(
            pretty.text(f"{constructor}(")
            + pretty.nest(self.indent, NULL_BREAK + arg)
            + NULL_BREAK
            + pretty.text(")")
        )


class Prettyfier(Simple):
    "Convert a value into a multiline document"

    def constant(
        self, constant: int | float | None | str | bytes | bool
    ) -> pretty.Doc:
        if isinstance(constant, int):
            return pretty.text(f"{constant:_d}")
        if (
            isinstance(constant, str)
            and len(constant) > 60
            and "\n" in constant
        ):
            return self.format_list(
                (
                    pretty.text(repr(line))
                    for line in constant.splitlines(keepends=True)
                ),
                sep=pretty.BREAK,
                gmode=pretty.Mode.BREAK,
            )
        return super().constant(constant)


class Boxed(pretty.DocCons):
    """A mutable node in a document tree."""

    __slots__ = ("priority",)
    priority: int

    def __init__(self, doc: pretty.Doc, priority: int) -> None:
        super().__init__(doc, pretty.EMPTY)
        self.priority = priority

    @property
    def value(self) -> pretty.Doc:
        return self.left

    @value.setter
    def value(self, doc: pretty.Doc) -> None:
        self.left = doc


class Alias(pretty.DocText):
    __slots__ = ()


def _get_import(path: str) -> str | None | Exception:
    try:
        while True:
            obj = utils.locate(path)
            if obj is None:
                return ImportError("Failed to find object")
            if inspect.ismodule(obj):
                return path
            if "." not in path:
                return None
            if path.startswith(getattr(obj, "__module__", "\000") + "."):
                return obj.__module__
            path, _ = path.rsplit(".", 1)
    except Exception as e:
        # Clear out all the fields set by `raise ...` that might leak large
        # amounts of memory
        e.__cause__ = e.__context__ = e.__traceback__ = None
        return e


class ModulePrinter(Prettyfier):

    INFINITY: ClassVar[pretty.Doc] = pretty.text('float("infinity")')
    NAN: ClassVar[pretty.Doc] = pretty.text('float("nan")')

    decls: list[tuple[int, pretty.Doc]]
    targets: dict[int, Boxed | Alias]
    imports: dict[str, str | Exception | None] | None
    pos: int

    def __init__(
        self,
        indent: int,
        add_imports: bool = True,
    ) -> None:
        super().__init__(indent=indent)
        self.targets = {}
        self.decls = []
        self.imports = {} if add_imports else None
        self.pos = -1

    def box(
        self, fn: Callable[P, pretty.Doc], *args: P.args, **kwargs: P.kwargs
    ) -> pretty.Doc:
        self.pos += 1
        idx = self.pos
        doc = fn(*args, **kwargs)
        box = self.targets[idx] = Boxed(doc, priority=self.pos)
        return box

    def mapping(
        self, items: Iterator[tuple[pretty.Doc, pretty.Doc]]
    ) -> pretty.Doc:
        return self.box(super().mapping, items)

    def constant(
        self, constant: int | float | None | str | bytes | bool
    ) -> pretty.Doc:
        self.pos += 1
        return super().constant(constant)

    def sequence(self, items: Iterator[pretty.Doc]) -> pretty.Doc:
        return self.box(super().sequence, items)

    def reference(self, offset: int) -> pretty.Doc:
        self.pos += 1
        pos = self.pos
        targets = self.targets
        idx = pos - offset
        assert idx in targets, targets
        orig = targets[idx]
        if isinstance(orig, Alias):
            targets[pos] = orig
            return orig
        doc = orig.value
        alias = targets[pos] = targets[idx] = orig.value = Alias(
            f"_{len(self.decls)}"
        )
        heapq.heappush(
            self.decls, (orig.priority, alias + pretty.text(" = ") + doc)
        )
        return alias

    def custom(
        self, constructor: str, value: Iterator[pretty.Doc]
    ) -> pretty.Doc:
        if self.imports is not None and constructor not in self.imports:
            self.imports[constructor] = _get_import(constructor)
        return self.box(super().custom, constructor, value)

    def root(self, doc: pretty.Doc) -> pretty.Doc:
        prelude = pretty.EMPTY

        import_errors: list[tuple[str, Exception]] = []
        imports = set[str]()

        if self.imports is not None:
            for k, v in self.imports.items():
                if v is None:
                    continue
                if isinstance(v, Exception):
                    import_errors.append((k, v))
                else:
                    imports.add(v)

            if import_errors:
                import_errors.sort()
                prelude += (
                    pretty.text(
                        "# There were errors trying to import the following "
                        "constructors"
                    )
                    + NULL_BREAK
                    + pretty.text("#")
                    + NULL_BREAK
                )
                for k, v in import_errors:
                    prelude += (
                        pretty.text(f"# + {k!r}: {type(v).__name__} {v}")
                        + NULL_BREAK
                    )
                prelude += NULL_BREAK

            if imports:
                for i in sorted(imports):
                    prelude += pretty.text(f"import {i}") + NULL_BREAK
                prelude += NULL_BREAK

        while self.decls:
            _, decl = heapq.heappop(self.decls)
            prelude += decl + NULL_BREAK + NULL_BREAK
        return pretty.hgrp(prelude + doc + NULL_BREAK)


def reduce_ast_expr(expr: ast.expr, orig: str, acc: base.Accumulator[T]) -> T:
    constant = acc.constant
    reference = acc.reference
    sequence = acc.sequence
    mapping = acc.mapping
    custom = acc.custom

    def reduce(expr: ast.expr) -> T:
        # We need a smattering of `type: ignore` statements because version
        # 0.942 of mypy.
        #
        # Resolve in: https://github.com/python/mypy/pull/12321
        match expr:
            # Primitives
            case ast.Constant(value) | ast.UnaryOp(  # type: ignore[misc]
                ast.UAdd(), ast.Constant(float(value) | int(value))
            ):
                return constant(value)
            case ast.UnaryOp(  # type: ignore[misc]
                ast.USub(),
                ast.Constant(float(num) | int(num)),
            ):
                return constant(-num)
            case ast.Name("nan"):  # type: ignore[misc]
                return constant(math.nan)
            case (
                ast.Name("inf")  # type: ignore[misc]
                | ast.UnaryOp(ast.UAdd(), ast.Name("inf"))  # type: ignore[misc]
            ):
                return constant(math.inf)
            case ast.UnaryOp(ast.USub(), ast.Name("inf")):  # type: ignore[misc]
                return constant(-math.inf)
            # /Primitives
            case ast.List(l):  # type: ignore[misc]
                return sequence((reduce(x) for x in l))
            case ast.Dict(k, v):  # type: ignore[misc]
                return mapping(_gen_kv(zip(k, v)))
            case ast.Call(  # type: ignore[misc]
                ast.Name("ref"), [ast.Constant(int(offset))]
            ):
                return reference(offset)
            case ast.Call(constructor, [arg]):  # type: ignore[misc]
                path = []
                while True:
                    match constructor:
                        case ast.Name(elt):
                            path.append(elt)
                            break
                        case ast.Attribute(constructor, elt):
                            path.append(elt)
                        case _:
                            error(constructor)
                return custom(".".join(reversed(path)), _custom_arg(arg))
        error(expr)

    _gen_kv = base._mk_kv_reducer(reduce)

    def _custom_arg(arg: Any) -> Iterator[T]:
        yield reduce(arg)

    def error(node: ast.expr) -> typing.NoReturn:
        ast_utils.raise_at(
            ValueError("Malformed node or string"),
            node=node,
            content=orig,
        )

    return acc.root(reduce(expr))


def dumps(
    obj: Any,
    indent: int | None = None,
    width: int = 60,
    format: Format | None = None,
) -> str:
    """Serialise *obj*


    Dumps supports several formats. Let's take a sample value with shared
    reference:

      >>> v = {'nan': math.nan, '1_5':[1,2,3,4,5]}
      >>> v2 = [v, v]

    + :py:const:`~otf.pack.COMPACT` means it will all be printed on one line.

      >>> print(dumps(v2, format = COMPACT))
      [{'nan': nan, '1_5': [1, 2, 3, 4, 5]}, ref(10)]

    + :py:const:`~otf.pack.PRETTY` will use the *width* and *indent* argument to
      pretty print the output.

      >>> print(dumps(v2, format = PRETTY, width=20))
      [
          {
              'nan': nan,
              '1_5': [
                  1,
                  2,
                  3,
                  4,
                  5
              ]
          },
          ref(10)
      ]

    + :py:data:`~otf.pack.EXECUTABLE` will print code that can run in a python
      environment where the last statement is the value we're building:

      >>> print(dumps(v2, format = EXECUTABLE, width=40))
      _0 = {
          'nan': float("nan"),
          '1_5': [1, 2, 3, 4, 5]
      }
      <BLANKLINE>
      [_0, _0]
      <BLANKLINE>


    Args:
      obj: The value to serialise
      indent(int | None): indentation (for the :const:`~otf.pack.PRETTY` and
         :const:`~otf.pack.EXECUTABLE` formats)
      width(int): Maximum line length (for the :const:`~otf.pack.PRETTY` and
         :const:`~otf.pack.EXECUTABLE` formats).
      format: One of :const:`None`, :const:`~otf.pack.COMPACT`,
         :const:`~otf.pack.PRETTY`, :const:`~otf.pack.EXECUTABLE`. If the value
         is :const:`None` then the format will be :const:`~otf.pack.COMPACT` if
         *indent* wasn't specified and :const:`~otf.pack.PRETTY` otherwise.

    """
    if format is None:
        format = Format.COMPACT if indent is None else Format.PRETTY
    if format == Format.COMPACT:
        reducer = Simple()
        doc = base.reduce(obj, reducer)
        return pretty.single_line(doc)
    if indent is None:
        indent = 4
    assert indent >= 0
    assert width > indent
    if format == Format.PRETTY:
        reducer = Prettyfier(indent=indent)
    else:
        assert format == Format.EXECUTABLE, format
        reducer = ModulePrinter(indent=indent, add_imports=True)
    doc = base.reduce(obj, reducer)
    return doc.to_string(width)


# TODO: allow taking in a typing.TextIO and use its name (if it has one)
def reduce(s: str, acc: base.Accumulator[T]) -> T:
    """TODO: Document me please"""
    e = ast.parse(s, mode="eval")
    assert isinstance(e, ast.Expression), type(e)
    return reduce_ast_expr(e.body, orig=s, acc=acc)


def loads(s: str) -> Any:
    """
    Args:
      s (str):
    """
    return reduce(s, acc=base.RuntimeValueBuilder())
