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

+ ``str, int, float, bool, None, bytes``: Basic python primitives
+ ``list``: where all the elements are serialisable
+ ``dict``: where all the keys and values are serialisable
+ ``tuple``: where all the elements are serialisable
+ ``shared references``: but not recursive values

API:
----

"""
import copyreg
import dataclasses
import functools
import inspect
import typing
import weakref
from typing import Any, Callable, Optional, Type, TypeVar

__all__ = (
    "explode",
    "implode",
    "copy",
    "register",
    "Value",
    "Custom",
    "Reference",
)

T = TypeVar("T")
V = TypeVar("V")

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
    ty: Optional[Type[T]] = arg.annotation
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
    *, type: Optional[Type[T]] = None, pickle: bool = False
) -> Callable[[Reducer[T]], Reducer[T]]:  # pragma: no cover
    ...


def register(
    function: Optional[Reducer[T]] = None,
    /,
    *,
    type: Optional[Type[T]] = None,
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
                return x, (y,)

            copyreg.pickle(cls, reduce)  # type: ignore[arg-type]
        DISPATCH_TABLE[cls] = function
        return function

    if function is None:
        return wrapper
    return wrapper(function)


@register
def _explode_tuple(t: tuple[T, ...]) -> Reduced[tuple[T, ...]]:
    return tuple, list(t)


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
      reduce(function):
      value: A serialisable value that can be passed as an argument to
        ``reduce`` to recreate the original value
    """

    constructor: Callable[[Any], Any]
    value: "Value"


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
    | dict[Any, Any]
    | list[Any]
)

MISSING = object()


class Exploder:

    cnt: int
    # id -> position
    memo: dict[int, Optional[int]]
    string_hashcon_length: int

    def __init__(self, string_hashcon_length: int = 32) -> None:
        self.cnt = 0
        self.memo = {}
        self.string_hashcon_length = string_hashcon_length

    def _explode_dict(self, d: dict[Any, Any]) -> dict[Value, Value]:
        rd = {}
        for key, value in d.items():
            # Order of evaluation matters so we don't use a dictionary
            # comprehension
            ekey = self.explode(key)
            evalue = self.explode(value)
            rd[ekey] = evalue
        return rd

    def explode(self, v: Any) -> Value:
        current = self.cnt
        self.cnt += 1
        ty = type(v)
        # We do exact type comparisons instead of calls to `isinstance` to
        # avoid running into problems with inheritance
        if ty in (int, float, type(None), bool):
            return v  # type: ignore[no-any-return]
        if ty in (bytes, str):
            if len(v) >= self.string_hashcon_length:
                addr = -abs(hash(v))
                memoized = self.memo.get(addr, MISSING)
                self.memo[addr] = current
                if isinstance(memoized, int):
                    return Reference(current - memoized)
            return v  # type: ignore[no-any-return]
        addr = id(v)
        memoized = self.memo.get(addr, MISSING)
        if isinstance(memoized, int):
            self.memo[addr] = current
            return Reference(current - memoized)
        if memoized is None:
            raise ValueError("Recursive value found")
        self.memo[addr] = None
        res: Value
        if ty is list:
            res = [self.explode(x) for x in v]
        elif ty is dict:
            res = self._explode_dict(v)
        else:
            serialiser = DISPATCH_TABLE.get(ty)
            if serialiser is None:
                raise TypeError(
                    f"Object of type {ty.__name__} cannot be serialised by OTF"
                )
            fn, arg = serialiser(v)
            res = Custom(fn, self.explode(arg))
        self.memo[addr] = current
        return res


class Imploder:
    cnt: int
    memo: dict[int, Any]

    def __init__(self) -> None:
        self.cnt = 0
        self.memo = {}

    def implode(self, exp: Value) -> Any:
        current = self.cnt
        self.cnt += 1
        res: Any
        if isinstance(exp, (int, float, type(None), str, bytes, bool)):
            return exp
        elif isinstance(exp, list):
            res = [self.implode(elt) for elt in exp]
        elif isinstance(exp, dict):
            res = {}
            for key, value in exp.items():
                # Order of evaluation matters so we don't use a dictionary
                # comprehension
                ekey = self.implode(key)
                res[ekey] = self.implode(value)
        elif isinstance(exp, Reference):
            res = self.memo[current - exp.offset]
        elif isinstance(exp, Custom):
            res = exp.constructor(exp.value)
        else:
            raise TypeError(
                f"Object of type {type(exp).__name__} cannot be de-serialised "
                "by OTF"
            )
        self.memo[current] = res
        return res


def explode(v: Any) -> Value:
    """Convert a python value to a simple, easy to serialise value.

    Args:
      v:
    """
    return Exploder().explode(v)


def implode(v: Value) -> Any:
    """Invert operation from :func:`implode`.

    Args:
      v:
    """
    return Imploder().implode(v)


def copy(v: T) -> T:
    """Copy a value using its representation.

    ``copy(v)`` is equivalent to ``implode(explode(v))``.

    Args:
      v:
    """
    # For performance reasons we might want to make a specialised class down the
    # line.
    return implode(explode(v))  # type: ignore[no-any-return]
