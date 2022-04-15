"""
``otf.pack``: Serialisation library
===================================

``Pack`` is a layer that is intended to be used on top of existing serialisation
libraries. It converts arbitrary python values to a simple intermediate
representation (:any:`Value`).

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
import abc
import ast
import copyreg
import dataclasses
import functools
import inspect
import math
import types
import typing
import weakref
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    Protocol,
    Type,
    TypeVar,
    Union,
)

from . import ast_utils, utils

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

T = TypeVar("T")
Contra = TypeVar("Contra", covariant=True)
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
        ``reduce`` to recreate the original value

    """

    constructor: str
    value: "Value"


@dataclasses.dataclass(slots=True, frozen=True)
class Mapping:
    data: list[tuple["Value", "Value"]]

    def items(self) -> Iterator[tuple["Value", "Value"]]:
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


@typing.runtime_checkable
class Implodable(Protocol[Contra]):
    @classmethod
    def _otf_reconstruct(
        cls: Type[Contra], _: Any
    ) -> Contra:  # pragma: no cover
        ...


CustomImploder = Union[CustomImplodeFn[T], Type[Implodable[T]]]


def _get_imploder(x: CustomImploder[T]) -> CustomImplodeFn[T]:
    """Hack to hide reconstruction functions

    This is a hidden feature that allows us to get cleaner ``Custom`` blocks.
    """
    if isinstance(x, Implodable):
        return x._otf_reconstruct
    return typing.cast(CustomImplodeFn[T], x)


@functools.lru_cache()
def _get_named_imploder(name: str) -> CustomImplodeFn[Any]:
    obj = typing.cast(CustomImploder[Any], utils.locate(name))
    return _get_imploder(obj)


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

    ``function`` is expected to take objects of type ``T`` and to return a tuple
    describing how to recreate the object: a function and serialisable
    value.

    If ``type`` is not specified, :func:`register` uses the type annotation on
    the first argument to deduce which type register ``function`` for.

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

      pickle: If set to ``True``, ``function`` is registered via
       :func:`copyreg.pickle` to be used in :mod:`pickle`.

    """

    def wrapper(function: Reducer[T]) -> Reducer[T]:
        cls = _infer_reducer_type(function) if type is None else type
        if pickle:

            @functools.wraps(function)
            def reduce(v: T) -> tuple[Callable[[V], T], tuple[V]]:
                x, y = function(v)
                return _get_imploder(x), (y,)

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
    """Get the custom serialiser for a given type."""
    serialiser = DISPATCH_TABLE.get(ty)
    if serialiser is None:
        raise TypeError(
            f"Object of type {ty.__name__} cannot be serialised by OTF"
        )
    return serialiser


def cexplode(v: Implodable[T]) -> Any:
    """Private function used to pull apart a custom type"""
    serialiser = _get_custom(type(v))
    _fn, res = serialiser(v)
    return res


def cimplode(type: Type[T], v: Any) -> T:
    """Private function used to rebuild a custom type"""
    return _get_imploder(type)(v)


class Serialiser(Generic[T], abc.ABC):

    cnt: int
    # id -> position
    memo: dict[int, int | None]
    string_hashcon_length: int

    def __init__(self, string_hashcon_length: int = 32) -> None:
        self.cnt = 0
        self.memo = {}
        self.string_hashcon_length = string_hashcon_length

    @abc.abstractmethod
    def constant(
        self, constant: int | float | None | str | bytes | bool
    ) -> T:  # pragma: no cover
        ...

    @abc.abstractmethod
    def mapping(self, items: list[tuple[T, T]]) -> T:  # pragma: no cover
        ...

    @abc.abstractmethod
    def sequence(self, items: list[T]) -> T:  # pragma: no cover
        ...

    @abc.abstractmethod
    def reference(self, offset: int) -> T:  # pragma: no cover
        ...

    @abc.abstractmethod
    def custom(self, constructor: str, value: T) -> T:  # pragma: no cover
        ...

    def serialise(self, v: Any) -> T:
        current = self.cnt
        self.cnt += 1
        ty = type(v)
        # We do exact type comparisons instead of calls to `isinstance` to
        # avoid running into problems with inheritance
        if ty in (int, float, types.NoneType, bool):
            return self.constant(v)
        if ty in (bytes, str):
            if len(v) >= self.string_hashcon_length:
                addr = -abs(hash(v))
                memoized = self.memo.get(addr, MISSING)
                self.memo[addr] = current
                if isinstance(memoized, int):
                    return self.reference(current - memoized)
            return self.constant(v)
        addr = id(v)
        memoized = self.memo.get(addr, MISSING)
        if isinstance(memoized, int):
            self.memo[addr] = current
            return self.reference(current - memoized)
        if memoized is None:
            raise ValueError("Recursive value found")
        self.memo[addr] = None
        res: T
        if ty is list:
            res = self.sequence([self.serialise(x) for x in v])
        elif ty is dict:
            data = []
            for key, value in v.items():
                # Note that the order is important here for references...
                ek = self.serialise(key)
                ev = self.serialise(value)
                data.append((ek, ev))
            res = self.mapping(data)
        else:
            serialiser = _get_custom(ty)
            fn, arg = serialiser(v)
            res = self.custom(utils.get_locate_name(fn), self.serialise(arg))
        self.memo[addr] = current
        return res


class Exploder(Serialiser[Value]):
    def constant(
        self, constant: int | float | None | str | bytes | bool
    ) -> Value:
        return constant

    def mapping(self, items: list[tuple[Value, Value]]) -> Value:
        constructor: Callable[[list[tuple[Value, Value]]], Value] = dict
        for k, _ in items:
            if not isinstance(k, int | float | str | bytes | bool | None):
                constructor = Mapping
                break
        return constructor(items)

    def sequence(self, items: list[Value]) -> Value:
        return items

    def reference(self, offset: int) -> Value:
        return Reference(offset)

    def custom(self, constructor: str, value: Value) -> Value:
        return Custom(constructor, value)


class Stringifier(Serialiser[ast.expr]):
    "Convert a value into an ast fragment"

    @staticmethod
    def constant(constant: int | float | None | str | bytes | bool) -> ast.expr:
        if not isinstance(constant, float) or math.isfinite(constant):
            return ast_utils.constant(constant)
        if math.isnan(constant):
            return ast_utils.name("nan")
        if constant == math.inf:
            return ast_utils.name("inf")
        assert constant == -math.inf
        return ast_utils.neg(ast_utils.name("inf"))

    @staticmethod
    def mapping(items: list[tuple[ast.expr, ast.expr]]) -> ast.expr:
        return ast.Dict(
            keys=[x for x, _ in items],
            values=[x for _, x in items],
            lineno=0,
            col_offset=0,
            end_lineno=0,
            end_col_offset=0,
        )

    @staticmethod
    def sequence(items: list[ast.expr]) -> ast.expr:
        return ast.List(
            elts=items,
            ctx=ast.Load(),
            lineno=0,
            col_offset=0,
            end_lineno=0,
            end_col_offset=0,
        )

    @staticmethod
    def reference(offset: int) -> ast.expr:
        return ast_utils.call(ast_utils.name("ref"), ast_utils.constant(offset))

    @staticmethod
    def custom(constructor: str, value: ast.expr) -> ast.expr:
        return ast_utils.call(ast_utils.dotted_path(constructor), value)


class Deserialiser(Generic[T]):
    memo: list[Any]

    def __init__(self) -> None:
        self.memo = []

    def custom(self, constructor: str, value: T) -> Any:
        return _get_named_imploder(constructor)(self.deserialise(value))

    def reference(self, offset: int) -> Any:
        return self.memo[len(self.memo) - 1 - offset]

    def sequence(self, items: list[T]) -> Any:
        return [self.deserialise(elt) for elt in items]

    def mapping(self, items: list[tuple[T, T]]) -> Any:
        res = {}
        for key, value in items:
            # Order of evaluation matters so we don't use a dictionary
            # comprehension
            ekey = self.deserialise(key)
            res[ekey] = self.deserialise(value)
        return res

    @abc.abstractmethod
    def visit(self, exp: T) -> Any:  # pragma: no cover
        ...

    def deserialise(self, exp: T) -> Any:
        current = len(self.memo)
        self.memo.append(None)
        res = self.visit(exp)
        self.memo[current] = res
        return res


class Imploder(Deserialiser[Value]):
    def visit(self, exp: Value) -> Any:
        if isinstance(exp, int | float | str | bytes | bool | None):
            return exp
        elif isinstance(exp, list):
            return self.sequence(exp)
        elif isinstance(exp, dict | Mapping):
            return self.mapping(list(exp.items()))
        elif isinstance(exp, Reference):
            return self.reference(exp.offset)
        elif isinstance(exp, Custom):
            return self.custom(exp.constructor, exp.value)
        else:
            raise TypeError(
                f"Object of type {type(exp).__name__} cannot be de-serialised "
                "by OTF"
            )


class UnStringifier(Deserialiser[ast.expr]):
    content: str

    def __init__(self, content: str) -> None:
        self.content = content
        super().__init__()

    def visit(self, expr: ast.expr) -> Any:
        # We need a smattering of type: ignore statements because version 0.942
        # of mypy.
        #
        # Resolve in: https://github.com/python/mypy/pull/12321
        match expr:
            # Primitives
            case ast.Constant(value) | ast.UnaryOp(  # type: ignore[misc]
                ast.UAdd(), ast.Constant(float(value) | int(value))
            ):
                return value
            case ast.UnaryOp(  # type: ignore[misc]
                ast.USub(),
                ast.Constant(float(num) | int(num)),  # type: ignore[misc]
            ):
                return -num
            case ast.Name("nan"):  # type: ignore[misc]
                return math.nan
            case (
                ast.Name("inf")  # type: ignore[misc]
                | ast.UnaryOp(ast.UAdd(), ast.Name("inf"))  # type: ignore[misc]
            ):
                return math.inf
            case ast.UnaryOp(ast.USub(), ast.Name("inf")):  # type: ignore[misc]
                return -math.inf
            # /Primitives
            case ast.List(l):  # type: ignore[misc]
                return self.sequence(l)
            case ast.Dict(k, v):  # type: ignore[misc]
                return self.mapping(list(zip(k, v)))
            case ast.Call(  # type: ignore[misc]
                ast.Name("ref"), [ast.Constant(int(offset))]
            ):
                return self.reference(offset)
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
                            self.error(constructor)
                return self.custom(".".join(reversed(path)), arg)
        self.error(expr)

    def error(self, node: ast.expr) -> typing.NoReturn:
        ast_utils.raise_at(
            ValueError("Malformed node or string"),
            node=node,
            content=self.content,
        )


def explode(v: Any) -> Value:
    """Convert a python value to a simple, easy to serialise value.

    Args:
      v:
    """
    return Exploder().serialise(v)


def implode(v: Value) -> Any:
    """Invert operation from :func:`implode`.

    Args:
      v:
    """
    return Imploder().deserialise(v)


def dumps(obj: Any) -> str:
    "Serialise *obj*."
    return ast.unparse(Stringifier().serialise(obj))


def loads(s: str) -> Any:
    """
    Args:
      s (str):
    """
    e = ast.parse(s, mode="eval")
    assert isinstance(e, ast.Expression), type(e)
    return UnStringifier(s).deserialise(e.body)


def copy(v: T) -> T:
    """Copy a value using its representation.

    ``copy(v)`` is equivalent to ``implode(explode(v))``.

    Args:
      v:
    """
    # For performance reasons we might want to make a specialised class down the
    # line.
    return implode(explode(v))  # type: ignore[no-any-return]
