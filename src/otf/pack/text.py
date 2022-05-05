"""
``otf.pack.text``: Human readable serialisation
================================================


"""

from __future__ import annotations

import ast
import dataclasses
import enum
import heapq
import inspect
import io
import math
import typing
from typing import (
    Any,
    Callable,
    ClassVar,
    Collection,
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
V = TypeVar("V")

__all__ = (
    "dump_text",
    "load_text",
    "reduce_text",
    "CompactPrinter",
    "PrettyPrinter",
    "ExecutablePrinter",
    "Format",
)


class Format(enum.Enum):
    """Which format to use for :func:`dump_text`"""

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


class CompactPrinter(base.Accumulator[pretty.Doc, str]):
    "Serialize a value as a human readable text."

    NAN: ClassVar[pretty.Doc] = pretty.text("nan")
    INFINITY: ClassVar[pretty.Doc] = pretty.text("inf")

    def format_list(
        self,
        docs: Iterable[pretty.Doc],
        *,
        sep: pretty.Doc = pretty.text(", "),
        opar: str = "(",
        cpar: str = ")",
    ) -> pretty.Doc:
        acc = pretty.EMPTY
        first = True
        for doc in docs:
            if not first:
                acc += sep
            else:
                first = False
            acc += doc
        return pretty.text(opar) + acc + pretty.text(cpar)

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
        self, size: int, items: Iterator[tuple[pretty.Doc, pretty.Doc]]
    ) -> pretty.Doc:
        return self.format_list(
            (k + pretty.text(": ") + v for k, v in items), opar="{", cpar="}"
        )

    def sequence(self, size: int, items: Iterator[pretty.Doc]) -> pretty.Doc:
        return self.format_list(items, opar="[", cpar="]")

    def reference(self, offset: int) -> pretty.Doc:
        return pretty.text(f"ref({offset:_d})")

    def custom(
        self,
        constructor: str,
        size: int,
        kwnames: Collection[str],
        values: Iterator[pretty.Doc],
    ) -> pretty.Doc:
        return self.format_list(
            (
                value if name is None else pretty.text(f"{name}=") + value
                for name, value in base.label_args(size, kwnames, values)
            ),
            opar=f"{constructor}(",
            cpar=")",
        )

    def root(self, doc: pretty.Doc) -> str:
        # We only use a very restricted subset of pretty.Doc that always prints
        # out to one line.
        out = io.StringIO()
        docs = [doc]
        while docs:
            match docs.pop():
                case pretty.DocNil():
                    continue
                case pretty.DocText(s):
                    out.write(s)
                case pretty.DocCons(left=l, right=r):
                    docs.append(r)
                    docs.append(l)
                case _:  # pragma: no cover
                    assert False
        return out.getvalue()


class PrettyPrinter(CompactPrinter):
    "Convert a value into a multiline document"
    indent: int
    width: int

    def __init__(self, indent: int = 4, width: int = 80) -> None:
        self.indent = indent
        self.width = width

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

    def root(self, doc: pretty.Doc) -> str:
        res = doc.to_string(self.width)
        assert isinstance(res, str)
        return res


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


class ExecutablePrinter(PrettyPrinter):

    INFINITY: ClassVar[pretty.Doc] = pretty.text('float("inf")')
    NAN: ClassVar[pretty.Doc] = pretty.text('float("nan")')

    decls: list[tuple[int, pretty.Doc]]
    targets: dict[int, Boxed | Alias]
    imports: dict[str, str | Exception | None] | None
    pos: int

    def __init__(
        self,
        indent: int = 4,
        width: int = 80,
        add_imports: bool = True,
    ) -> None:
        super().__init__(indent=indent, width=width)
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
        self, size: int, items: Iterator[tuple[pretty.Doc, pretty.Doc]]
    ) -> pretty.Doc:
        return self.box(super().mapping, size, items)

    def constant(
        self, constant: int | float | None | str | bytes | bool
    ) -> pretty.Doc:
        self.pos += 1
        return super().constant(constant)

    def sequence(self, size: int, items: Iterator[pretty.Doc]) -> pretty.Doc:
        return self.box(super().sequence, size, items)

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
        self,
        constructor: str,
        size: int,
        kwnames: Collection[str],
        values: Iterator[pretty.Doc],
    ) -> pretty.Doc:
        if self.imports is not None and constructor not in self.imports:
            self.imports[constructor] = _get_import(constructor)
        return self.box(
            super().custom, constructor, size, kwnames, values=values
        )

    def root(self, doc: pretty.Doc) -> str:
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
        return super().root(pretty.hgrp(prelude + doc + NULL_BREAK))


@dataclasses.dataclass(slots=True)
class EnvBinding:
    expr: ast.expr
    # this number increases as the ref are defined in the
    # environment. This is used to check that we visit the refs in
    # a subset of the order in which they are defined in the source
    # document.
    src_idx: int
    dst_idx: int | None = None


ENV_ELEMENT = ast.alias | EnvBinding


# TODO: allow taking in a typing.TextIO and use its name (if it has one)
def reduce_text(orig: str, acc: base.Accumulator[T, V]) -> V:
    assert isinstance(orig, str), orig
    module = ast.parse(orig, mode="exec")

    constant = acc.constant
    reference = acc.reference
    sequence = acc.sequence
    mapping = acc.mapping
    custom = acc.custom
    cnt = -1

    env: dict[str, ENV_ELEMENT] = {}

    # This value is used along with the src_idx on the EnvBinding to make sure
    # we don't have cycles or binding defined in the wrong order.
    visiting_ref = len(module.body)

    def seq_arg(size: int, elts: Iterable[ast.expr]) -> Iterator[T]:
        """Generate the argument for the set and tuple custom block"""
        yield sequence(size, (reduce(e) for e in elts))

    def reduce(expr: ast.expr) -> T:
        nonlocal cnt
        cnt += 1
        match expr:
            # Primitives
            case ast.Constant(value) | ast.UnaryOp(
                ast.UAdd(), ast.Constant(float(value) | int(value))
            ):
                return constant(value)
            case ast.UnaryOp(
                ast.USub(),
                ast.Constant(float(num) | int(num)),
            ):
                return constant(-num)
            case (
                ast.Name("nan")
                | ast.Call(ast.Name("float"), [ast.Constant("nan")], [])
            ):
                return constant(math.nan)
            case (
                ast.Name("inf")
                | ast.Call(ast.Name("float"), [ast.Constant("inf")], [])
                | ast.UnaryOp(
                    ast.UAdd(),
                    ast.Name("inf")
                    | ast.Call(ast.Name("float"), [ast.Constant("inf")], []),
                )
            ):
                return constant(math.inf)
            case ast.UnaryOp(
                ast.USub(),
                ast.Name("inf")
                | ast.Call(ast.Name("float"), [ast.Constant("inf")], []),
            ):
                return constant(-math.inf)
            # /Primitives
            case (ast.Name(ref_name)):
                bound = env.get(ref_name, None)
                if not isinstance(bound, EnvBinding):
                    if isinstance(bound, ast.alias) and bound.name == ref_name:
                        error(
                            expr,
                            f"{ref_name!r}: cannot reference an imported "
                            "module",
                        )
                    assert isinstance(bound, ast.alias | None)
                    error(expr, f"Unbound variable {ref_name!r}")
                previous_dst_idx = bound.dst_idx
                bound.dst_idx = cnt
                nonlocal visiting_ref
                current_src_idx = visiting_ref
                if current_src_idx == bound.src_idx:
                    error(expr, f"circular reference for {ref_name!r}")
                if current_src_idx < bound.src_idx:
                    error(
                        expr,
                        "referencing value that isn't defined yet: "
                        f"{ref_name!r}",
                    )
                if previous_dst_idx is None:
                    # Decrease this to inline the source document
                    cnt -= 1
                    visiting_ref = bound.src_idx
                    res = reduce(bound.expr)
                    visiting_ref = current_src_idx
                else:
                    res = reference(cnt - previous_dst_idx)
                return res
            case ast.List(l):
                return sequence(len(l), (reduce(x) for x in l))
            case ast.Dict(k, v):
                return mapping(len(k), _dict_fields(k, v))
            case ast.Set(elts):
                return custom("set", 1, (), seq_arg(len(elts), elts))
            case ast.Tuple(elts):
                return custom("tuple", 1, (), seq_arg(len(elts), elts))
            case ast.Call(ast.Name("ref"), [ast.Constant(int(offset))], []):
                return reference(offset)
            case ast.Call(constructor, args, kwargs):
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
                values = args[:]
                kwnames = []
                for keyword in kwargs:
                    if keyword.arg is None:
                        error(keyword)
                    values.append(keyword.value)
                    kwnames.append(keyword.arg)
                return custom(
                    ".".join(reversed(path)),
                    size=len(values),
                    kwnames=kwnames,
                    values=(reduce(value) for value in values),
                )
        error(
            expr, f"Don't know how to reduce: {utils.cram(ast.dump(expr), 60)}"
        )

    def _dict_fields(
        keys: Iterable[ast.expr | None], values: Iterable[ast.expr]
    ) -> Iterator[tuple[T, T]]:
        for key, value in zip(keys, values):
            if key is None:
                error(value, "Dictionary expansion not supported")
            ek = reduce(key)
            ev = reduce(value)
            yield (ek, ev)

    def error(
        node: ast.AST, message: str = "Malformed node"
    ) -> typing.NoReturn:
        ast_utils.raise_at(
            ValueError(message),
            node=node,
            content=orig,
        )

    match module.body:
        case *head, ast.Expr(expr):
            pass
        case *_, last:
            error(
                last,
                "The last node of the document should be a python expression",
            )
        case _:
            raise ValueError("Empty document")

    has_assigns = False
    for stmt in head:
        match stmt:
            case ast.Import(aliases):
                if has_assigns:
                    error(stmt, "Cannot import after declaring variables")
                for alias in aliases:
                    if alias.asname is not None:
                        error(alias, "`import ... as` are not supported")
                    root_name = alias.name.split(".", 1)[0]
                    if root_name not in env:
                        env[root_name] = alias
            case ast.Assign(
                targets=[ast.Name(name)], value=value
            ) | ast.AnnAssign(target=ast.Name(name), value=ast.expr() as value):
                has_assigns = True
                if name in ("ref", "nan", "inf"):
                    error(stmt, f"{name!r} is a reserved keyword")
                previous = env.get(name, None)
                if isinstance(previous, ast.alias):
                    error(
                        stmt,
                        "Cannot redefine a name already used by an import.",
                    )
                elif previous is not None:
                    assert isinstance(previous, EnvBinding)
                    error(
                        stmt,
                        "Cannot rebind a variable.",
                    )
                env[name] = EnvBinding(value, src_idx=len(env))

            case ast.Assign(targets=[_]):
                error(
                    stmt, "Only assigning to top level variables is supported"
                )

            case ast.Assign(targets=_):
                error(stmt, "Assigning to multiple targets is not supported")
            case ast.ImportFrom():
                error(stmt, "`from ... import` are not supported.")
            case ast.Expr():
                error(
                    stmt,
                    "Expressions are only allowed as the last statement of a "
                    "document",
                )
            case _:
                error(
                    stmt,
                    "Only bindings and imports are supported in the prelude",
                )

    return acc.root(reduce(expr))


def accumulator(
    indent: int | None = None,
    width: int = 60,
    format: Format | None = None,
) -> base.Accumulator[Any, str]:
    if format is None:
        format = Format.COMPACT if indent is None else Format.PRETTY
    if format == Format.COMPACT:
        return CompactPrinter()
    if indent is None:
        indent = 4
    assert indent >= 0
    assert width > indent
    if format == Format.PRETTY:
        return PrettyPrinter(indent=indent, width=width)
    assert format == Format.EXECUTABLE, format
    return ExecutablePrinter(indent=indent, width=width, add_imports=True)


def dump_text(
    obj: Any,
    indent: int | None = None,
    width: int = 60,
    format: Format | None = None,
) -> str:
    """Serialise *obj*


    Dumps supports several formats. Let's take a sample value with a shared
    reference:

      >>> v = {'nan': math.nan, '1_5':[1,2,3,4,5]}
      >>> v2 = [v, v]

    + :py:const:`~otf.pack.COMPACT` means it will all be printed on one line.

      >>> print(dump_text(v2, format = COMPACT))
      [{'nan': nan, '1_5': [1, 2, 3, 4, 5]}, ref(10)]

    + :py:const:`~otf.pack.PRETTY` will use the *width* and *indent* argument to
      pretty print the output.

      >>> print(dump_text(v2, format = PRETTY, width=20))
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

      >>> print(dump_text(v2, format = EXECUTABLE, width=40))
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
    return base.reduce_runtime_value(
        obj, accumulator(indent=indent, width=width, format=format)
    )


def load_text(s: str) -> Any:
    """
    Load a value encoded as a string

    Args:
      s (str):
    """
    return reduce_text(s, acc=base.RuntimeValueBuilder())
