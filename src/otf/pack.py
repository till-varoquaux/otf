"""``otf.pack``: Serialisation library
===================================

:mod:`~otf.pack` is a layer that is intended to be used on top of existing
serialisation libraries. It converts arbitrary python values to a simple
intermediate representation (:any:`Value`).

What differentiate :mod:`~otf.pack` from other serialisation libraries is that
it is very explicit. Out of the box :mod:`~otf.pack` only handle a small set of
types; we don't rely on inheritance or the runtime structure of values to handle
anything that isn't of a supported type. Support for new types can be added via
:func:`register`.

Supported types:
----------------

Out of the box, the types that are supported are:

+ :class:`str`, :class:`bytes`, :class:`int`, :class:`float`, :class:`bool`, \
    ``None``: Basic python primitives
+ :class:`list`: where all the elements are serialisable
+ :class:`dict`: where all the keys and values are serialisable
+ :class:`tuple`: where all the elements are serialisable
+ :class:`shared references`: but not recursive values

API:
----

"""
from __future__ import annotations

import abc
import ast
import copyreg
import dataclasses
import functools
import heapq
import inspect
import math
import types
import typing
import weakref
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterable,
    Iterator,
    ParamSpec,
    Type,
    TypeVar,
)

from . import ast_utils, pretty, utils

__all__ = (
    "dumps",
    "loads",
    "explode",
    "implode",
    "copy",
    "register",
    "Value",
    "Custom",
    "Reference",
)

P = ParamSpec("P")
T = TypeVar("T")
T_Co = TypeVar("T_Co", covariant=True)
V = TypeVar("V")

CustomImplodeFn = Callable[[Any], T]


@dataclasses.dataclass(slots=True, frozen=True)
class Reference:
    """Denotes a shared reference.

    This is a reference to the object that appeared offset nodes ago in a
    depth first traversal of the tree.

    Parameters:
      offset(int):
    """

    offset: int


@dataclasses.dataclass(slots=True, frozen=True)
class Custom:
    """Denotes a type with a custom de-serialisation function

    Args:

      constructor(str): The name of the function used to do the implosion of
        custom types

      value(Value): A serialisable value that can be passed as an argument to
        *constructor* to recreate the original value

    """

    constructor: str
    value: Value


@dataclasses.dataclass(slots=True, frozen=True)
class Mapping:
    data: list[tuple[Value, Value]]

    def items(self) -> Iterator[tuple[Value, Value]]:
        yield from self.data


# Mypy doesn't support recursive types
# (https://github.com/python/mypy/issues/731)

# We could use Protocol like in:
# https://github.com/python/typing/issues/182#issuecomment-893657366 but, in
# practice, this turned out to be a bigger headache than the problem it was
# trying to solve.

#:
Value = (
    int
    | float
    | None
    | str
    | bytes
    | bool
    | Custom
    | Reference
    | Mapping
    | dict[Any, Any]
    | list[Any]
)


@functools.lru_cache()
def _get_named_imploder(name: str) -> CustomImplodeFn[Any]:
    return typing.cast(CustomImplodeFn[Any], utils.locate(name))


Reduced = tuple[Callable[[Any], T], Any]
Reducer = Callable[[T], Reduced[T]]


DISPATCH_TABLE = weakref.WeakKeyDictionary[Type[Any], Reducer[Any]]()


def _infer_reducer_type(f: Reducer[T]) -> Type[T]:
    values = list(inspect.signature(f, eval_str=True).parameters.values())
    if len(values) != 1:
        raise ValueError(
            "The registered function should take only one argument."
        )
    [arg] = values
    ty: Type[T] | None = arg.annotation
    origin = typing.get_origin(ty)
    if origin is not None:
        ty = origin
    assert ty is not None
    return ty


@typing.overload
def register(function: Reducer[T], /) -> Reducer[T]:  # pragma: no cover
    ...


@typing.overload
def register(
    *, type: Type[T] | None = None, pickle: bool = False
) -> Callable[[Reducer[T]], Reducer[T]]:  # pragma: no cover
    ...


def register(
    function: Reducer[T] | None = None,
    /,
    *,
    type: Type[T] | None = None,
    pickle: bool = False,
) -> Reducer[T] | Callable[[Reducer[T]], Reducer[T]]:
    """Register a function to use while packing object of a given type.

    *function* is expected to take objects of type *T* and to return a tuple
    describing how to recreate the object: a function and serialisable
    value.

    If *type* is not specified, :func:`register` uses the type annotation on
    the first argument to deduce which type register *function* for.

    If :func:`register` is used as a simple decorator (with no arguments) it
    acts as though the default values for all of it parameters.

    Here are three equivalent ways to add support for the :class:`complex`
    type::

        >>> def mk_complex(l):
        ...    real, imag = l
        ...    return complex(real, imag)

        >>> @register
        ... def _reduce_complex(c: complex):
        ...   return mk_complex, [c.real, c.imag]

        >>> @register()
        ... def _reduce_complex(c: complex):
        ...   return mk_complex, [c.real, c.imag]

        >>> @register(type=complex)
        ... def _reduce_complex(c: complex):
        ...   return mk_complex, [c.real, c.imag]

    Args:

      function: The reduction we are registering

      type: The type we are registering the function for

      pickle: If set to ``True``, *function* is registered via
       :func:`copyreg.pickle` to be used in :mod:`pickle`.

    """

    def wrapper(function: Reducer[T]) -> Reducer[T]:
        cls = _infer_reducer_type(function) if type is None else type
        if pickle:

            @functools.wraps(function)
            def reduce(v: T) -> tuple[Callable[[V], T], tuple[V]]:
                imploder, arg = function(v)
                return imploder, (arg,)

            copyreg.pickle(cls, reduce)  # type: ignore[arg-type]
        DISPATCH_TABLE[cls] = function
        return function

    if function is None:
        return wrapper
    return wrapper(function)


@register
def _explode_tuple(t: tuple[T, ...]) -> Reduced[tuple[T, ...]]:
    return tuple, list(t)


MISSING = object()


def _get_custom(ty: Type[T]) -> Reducer[T]:
    """Get the custom reducer for a given type."""
    reducer = DISPATCH_TABLE.get(ty)
    if reducer is None:
        raise TypeError(
            f"Object of type {ty.__name__} cannot be serialised by OTF"
        )
    return reducer


def shallow_reduce(v: Any) -> Any:
    """Private function used to pull apart a custom type"""
    reducer = _get_custom(type(v))
    _fn, res = reducer(v)
    return res


class Accumulator(Generic[T], abc.ABC):
    @abc.abstractmethod
    def constant(
        self, constant: int | float | None | str | bytes | bool
    ) -> T:  # pragma: no cover
        ...

    # We want to make sure we pass in iterators because that gives the
    # `sequence`, `mapping` and `custom` constructors a chance to do something
    # both before and after the sub-nodes are visited.
    @abc.abstractmethod
    def mapping(self, items: Iterator[tuple[T, T]]) -> T:  # pragma: no cover
        ...

    @abc.abstractmethod
    def sequence(self, items: Iterator[T]) -> T:  # pragma: no cover
        ...

    @abc.abstractmethod
    def reference(self, offset: int) -> T:  # pragma: no cover
        ...

    # Takes an iterator that yields only one value (the argument to pass to the
    # constructor)
    @abc.abstractmethod
    def custom(
        self, constructor: str, value: Iterator[T]
    ) -> T:  # pragma: no cover
        ...

    def root(self, value: T) -> T:
        # wrap the root value
        return value


def _mk_kv_reducer(
    reduce: Callable[[T], V]
) -> Callable[[Iterable[tuple[T, T]]], Iterator[tuple[V, V]]]:
    def res(items: Iterable[tuple[T, T]]) -> Iterator[tuple[V, V]]:
        for key, value in items:
            # Note that the order is important here for references...
            ek = reduce(key)
            ev = reduce(value)
            yield (ek, ev)

    return res


def reduce_runtime(
    obj: Any, prod: Accumulator[T], string_hashcon_length: int = 32
) -> T:
    cnt: int = -1
    # id -> position
    memo: dict[int, int | None] = {}
    # Since we rely on `id` to detect duplicates we have to hold on to all the
    # intermediate values to make sure addresses do not get reused
    transient: list[Any] = []

    constant = prod.constant
    reference = prod.reference
    sequence = prod.sequence
    mapping = prod.mapping
    custom = prod.custom

    def _custom_arg(arg: Any) -> Iterator[T]:
        # Avoid the `id` being reused by other transient values
        transient.append(arg)
        yield reduce(arg)

    def reduce(v: Any) -> T:
        nonlocal cnt
        cnt += 1
        current = cnt
        ty = type(v)
        # We do exact type comparisons instead of calls to `isinstance` to
        # avoid running into problems with inheritance
        if ty in (int, float, types.NoneType, bool):
            return constant(v)
        if ty in (bytes, str):
            if len(v) >= string_hashcon_length:
                addr = -abs(hash(v))
                memoized = memo.get(addr, MISSING)
                memo[addr] = current
                if isinstance(memoized, int):
                    return reference(current - memoized)
            return constant(v)
        addr = id(v)
        memoized = memo.get(addr, MISSING)
        if isinstance(memoized, int):
            memo[addr] = current
            return reference(current - memoized)
        if memoized is None:
            raise ValueError("Recursive value found")
        memo[addr] = None
        res: T
        if ty is list:
            res = sequence(reduce(x) for x in v)
        elif ty is dict:
            res = mapping(_gen_kv(v.items()))
        else:
            deconstructor = _get_custom(ty)
            fn, arg = deconstructor(v)
            res = custom(utils.get_locate_name(fn), _custom_arg(arg))
        memo[addr] = current
        return res

    _gen_kv = _mk_kv_reducer(reduce)

    return prod.root(reduce(obj))


class Exploder(Accumulator[Value]):
    def constant(
        self, constant: int | float | None | str | bytes | bool
    ) -> Value:
        return constant

    def mapping(self, items: Iterator[tuple[Value, Value]]) -> Value:
        acc: dict[Any, Any] = {}
        for k, v in items:
            # Do we need to bail out and return a Mapping because the key is
            # un-hashable?
            if not isinstance(k, int | float | str | bytes | bool | None):
                return Mapping([*acc.items(), (k, v), *items])
            acc[k] = v
        return acc

    def sequence(self, items: Iterator[Value]) -> Value:
        return list(items)

    def reference(self, offset: int) -> Value:
        return Reference(offset)

    def custom(self, constructor: str, value: Iterator[Value]) -> Value:
        (arg,) = value
        return Custom(constructor, arg)


class Stringifier(Accumulator[ast.expr]):
    "Convert a value into an ast fragment"

    def constant(
        self, constant: int | float | None | str | bytes | bool
    ) -> ast.expr:
        if not isinstance(constant, float) or math.isfinite(constant):
            return ast_utils.constant(constant)
        if math.isnan(constant):
            return ast_utils.name("nan")
        if constant == math.inf:
            return ast_utils.name("inf")
        assert constant == -math.inf
        return ast_utils.neg(ast_utils.name("inf"))

    def mapping(self, items: Iterator[tuple[ast.expr, ast.expr]]) -> ast.expr:
        keys = []
        values = []
        for k, v in items:
            keys.append(k)
            values.append(v)
        return ast.Dict(
            keys=keys,
            values=values,
            lineno=0,
            col_offset=0,
            end_lineno=0,
            end_col_offset=0,
        )

    def sequence(self, items: Iterator[ast.expr]) -> ast.expr:
        return ast.List(
            elts=list(items),
            ctx=ast.Load(),
            lineno=0,
            col_offset=0,
            end_lineno=0,
            end_col_offset=0,
        )

    def reference(self, offset: int) -> ast.expr:
        return ast_utils.call(ast_utils.name("ref"), ast_utils.constant(offset))

    def custom(self, constructor: str, value: Iterator[ast.expr]) -> ast.expr:
        (arg,) = value
        return ast_utils.call(ast_utils.dotted_path(constructor), arg)


COL_SEP = pretty.text(",") + pretty.BREAK
NULL_BREAK = pretty.break_with("")


class Prettyfier(Accumulator[pretty.Doc]):
    "Convert a value into an ast fragment"

    NAN: ClassVar[pretty.Doc] = pretty.text("nan")
    INFINITY: ClassVar[pretty.Doc] = pretty.text("inf")

    def __init__(self, indent: int) -> None:
        self.indent = indent

    def format_list(
        self,
        docs: Iterable[pretty.Doc],
        *,
        sep: pretty.Doc = COL_SEP,
        opar: str,
        cpar: str,
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
        return pretty.agrp(pretty.text(opar) + body + pretty.text(cpar))

    def constant(
        self, constant: int | float | None | str | bytes | bool
    ) -> pretty.Doc:
        if isinstance(constant, int):
            return pretty.text(f"{constant:_d}")
        if (
            isinstance(constant, str)
            and len(constant) > 40
            and "\n" in constant
        ):
            return self.format_list(
                (
                    pretty.text(repr(line))
                    for line in constant.splitlines(keepends=True)
                ),
                sep=pretty.BREAK,
                opar="(",
                cpar=")",
            )
        if isinstance(constant, float) and not math.isfinite(constant):
            if math.isnan(constant):
                return self.NAN
            if constant == math.inf:
                return self.INFINITY
            assert constant == -math.inf
            return pretty.text("-") + self.INFINITY
        assert isinstance(constant, float | bytes | str | None | bool)
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


# Helping out mypy a bit
_mk_list: Callable[[Iterable[Any]], Any] = list
_mk_dict: Callable[[Iterable[tuple[Any, Any]]], Any] = dict


class RuntimeValueBuilder(Accumulator[Any]):
    memo: list[Any]

    def __init__(self) -> None:
        self.memo = []

    def constant(
        self, constant: int | float | None | str | bytes | bool
    ) -> Any:
        self.memo.append(constant)
        return constant

    def reference(self, offset: int) -> Any:
        res = self.memo[len(self.memo) - offset]
        self.memo.append(res)
        return res

    def _custom(self, constructor: str, value: Iterator[Any]) -> Any:
        (arg,) = value
        return _get_named_imploder(constructor)(arg)

    def _reduce(
        self, fn: Callable[P, Any], *args: P.args, **kwargs: P.kwargs
    ) -> Any:
        current = len(self.memo)
        self.memo.append(None)
        res = fn(*args, **kwargs)
        self.memo[current] = res
        return res

    def custom(self, constructor: str, value: Iterator[Any]) -> Any:
        return self._reduce(self._custom, constructor, value)

    def sequence(self, items: Iterable[T]) -> Any:
        return self._reduce(_mk_list, items)

    def mapping(self, items: Iterable[tuple[Any, Any]]) -> Any:
        return self._reduce(_mk_dict, items)


def reduce_runtime_value(value: Value, prod: Accumulator[T]) -> T:
    constant = prod.constant
    reference = prod.reference
    sequence = prod.sequence
    mapping = prod.mapping
    custom = prod.custom

    def reduce(exp: Value) -> T:
        if isinstance(exp, (int, float, str, bytes, bool, type(None))):
            return constant(exp)
        elif isinstance(exp, list):
            return sequence((reduce(elt) for elt in exp))
        elif isinstance(exp, dict | Mapping):
            return mapping(_gen_kv(exp.items()))
        elif isinstance(exp, Reference):
            return reference(exp.offset)
        elif isinstance(exp, Custom):
            return custom(exp.constructor, _custom_arg(exp.value))
        else:
            raise TypeError(
                f"Object of type {type(exp).__name__} cannot be de-serialised "
                "by OTF"
            )

    _gen_kv = _mk_kv_reducer(reduce)

    def _custom_arg(arg: Any) -> Iterator[T]:
        yield reduce(arg)

    return prod.root(reduce(value))


def reduce_ast_expr(expr: ast.expr, orig: str, prod: Accumulator[T]) -> T:
    constant = prod.constant
    reference = prod.reference
    sequence = prod.sequence
    mapping = prod.mapping
    custom = prod.custom

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

    _gen_kv = _mk_kv_reducer(reduce)

    def _custom_arg(arg: Any) -> Iterator[T]:
        yield reduce(arg)

    def error(node: ast.expr) -> typing.NoReturn:
        ast_utils.raise_at(
            ValueError("Malformed node or string"),
            node=node,
            content=orig,
        )

    return prod.root(reduce(expr))


def explode(v: Any) -> Value:
    """Convert a python value to a simple, easy to serialise value.

    Args:
      v:
    """
    return reduce_runtime(v, Exploder())


def implode(v: Value) -> Any:
    """Invert operation from :func:`implode`.

    Args:
      v:
    """
    return reduce_runtime_value(v, RuntimeValueBuilder())


def dumps(obj: Any, indent: int | None = None, width: int = 60) -> str:
    """Serialise *obj*

    If *indent* is not ``None`` the output will be pretty-printed

    Args:
      obj: The value to serialise
      indent(int | None): indentation to use for the pretty printing
      width(int): Maximum line length (when *indent* is specified)
    """
    if indent is None:
        return ast.unparse(reduce_runtime(obj, Stringifier()))
    else:
        assert indent >= 0
        assert width > indent
        doc = reduce_runtime(obj, Prettyfier(indent=indent))
        return doc.to_string(width)


def edit(obj: Any, add_imports: bool = True) -> str:
    doc = reduce_runtime(obj, ModulePrinter(indent=4, add_imports=add_imports))
    return doc.to_string(80)


def loads(s: str) -> Any:
    """
    Args:
      s (str):
    """
    e = ast.parse(s, mode="eval")
    assert isinstance(e, ast.Expression), type(e)
    return reduce_ast_expr(e.body, orig=s, prod=RuntimeValueBuilder())


def copy(v: T) -> T:
    """Copy a value using its representation.

    ``copy(v)`` is equivalent to ``loads(dumps(v))``.

    Args:
      v:
    """
    # For performance reasons we might want to add the ability to not copy
    # immutable values.
    res: T = reduce_runtime(v, RuntimeValueBuilder())
    return res
