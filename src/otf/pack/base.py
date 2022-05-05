from __future__ import annotations

import abc
import copyreg
import functools
import inspect
import types
import typing
import weakref
from typing import (
    Any,
    Callable,
    Collection,
    Generic,
    Iterable,
    Iterator,
    ParamSpec,
    Type,
    TypeAlias,
    TypeVar,
)

from otf import utils

P = ParamSpec("P")
T = TypeVar("T")
T_Co = TypeVar("T_Co", covariant=True)
V = TypeVar("V")

CustomImplodeFn: TypeAlias = Callable[..., T]

Reduced: TypeAlias = tuple[Callable[..., T], tuple[Any, ...], dict[str, Any]]

Reducer: TypeAlias = Callable[[T], Reduced[T]]


@functools.lru_cache()
def _get_named_imploder(name: str) -> CustomImplodeFn[Any]:
    return typing.cast(CustomImplodeFn[Any], utils.locate(name))


DISPATCH_TABLE = weakref.WeakKeyDictionary[Type[Any], Reducer[Any]]()


def _infer_reducer_type(f: Reducer[T]) -> Type[T]:
    values = list(inspect.signature(f, eval_str=True).parameters.values())
    if len(values) != 1:
        raise ValueError(
            "The registered function should take only one argument"
        )
    [arg] = values
    ty: Type[T] | None = arg.annotation
    origin = typing.get_origin(ty)
    if origin is not None:
        ty = origin
    assert ty is not None
    return ty


def pickle_call(
    fn: Callable[..., T], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> T:
    "Function that serves as a trampoline to pass kwargs"
    # Pickle's reduce only allows to use *args on the callback.
    return fn(*args, **kwargs)


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

        >>> @register
        ... def _reduce_complex(c: complex):
        ...   return complex, (c.real, c.imag)

        >>> @register()
        ... def _reduce_complex(c: complex):
        ...   return complex, (c.real, c.imag)

        >>> @register(type=complex)
        ... def _reduce_complex(c: complex):
        ...   return complex, (c.real, c.imag)

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
            def reduce(v: T) -> tuple[Callable[..., T], tuple[Any, ...]]:
                imploder, args, kwargs = function(v)
                if kwargs == {}:
                    return imploder, args
                return pickle_call, (imploder, args, kwargs)

            copyreg.pickle(cls, reduce)
        DISPATCH_TABLE[cls] = function
        return function

    if function is None:
        return wrapper
    return wrapper(function)


@register
def _explode_tuple(t: tuple[T, ...]) -> Reduced[tuple[T, ...]]:
    return tuple, (list(t),), {}


@register
def _explode_set(s: set[T]) -> Reduced[set[T]]:
    return set, (list(s),), {}


MISSING = object()


def _get_custom(ty: Type[T]) -> Reducer[T]:
    """Get the custom reducer for a given type."""
    reducer = DISPATCH_TABLE.get(ty)
    if reducer is None:
        raise TypeError(
            f"Object of type {ty.__name__} cannot be reduced by `OTF.pack`"
        )
    return reducer


def shallow_reduce(v: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Pull apart a custom type"""
    reducer = _get_custom(type(v))
    _fn, args, kwargs = reducer(v)
    return args, kwargs


def _iter_labels(size: int, kwnames: Collection[str]) -> Iterator[str | None]:
    for i in range(size - len(kwnames)):
        yield None
    yield from kwnames


def label_args(
    size: int, kwnames: Collection[str], args: Iterable[T]
) -> Iterator[tuple[str | None, T]]:
    """
    >>> list(label_args(7, ('a', 'b'), range(7)))
    [(None, 0), (None, 1), (None, 2), (None, 3), (None, 4), ('a', 5), ('b', 6)]
    """
    yield from zip(_iter_labels(size, kwnames), args, strict=True)


def group_args(
    size: int, kwnames: Collection[str], args: Iterable[T]
) -> tuple[list[T], dict[str, T]]:
    """
    >>> group_args(7, ('a', 'b'), range(7))
    ([0, 1, 2, 3, 4], {'a': 5, 'b': 6})
    """
    nargs = size - len(kwnames)
    it = iter(args)
    args = [next(it) for _ in range(nargs)]
    kwargs = {k: v for k, v in zip(kwnames, it, strict=True)}
    return args, kwargs


class Accumulator(Generic[T, V], abc.ABC):
    @abc.abstractmethod
    def constant(
        self, constant: int | float | None | str | bytes | bool
    ) -> T:  # pragma: no cover
        ...

    # We want to make sure we pass in iterators because that gives the
    # `sequence`, `mapping` and `custom` constructors a chance to do something
    # both before and after the sub-nodes are visited.
    @abc.abstractmethod
    def mapping(
        self, size: int, items: Iterator[tuple[T, T]]
    ) -> T:  # pragma: no cover
        ...

    @abc.abstractmethod
    def sequence(self, size: int, items: Iterator[T]) -> T:  # pragma: no cover
        ...

    @abc.abstractmethod
    def reference(self, offset: int) -> T:  # pragma: no cover
        ...

    # Takes an iterator that yields only one value (the argument to pass to the
    # constructor)
    @abc.abstractmethod
    def custom(
        self,
        constructor: str,
        size: int,
        kwnames: Collection[str],
        values: Iterator[T],
    ) -> T:  # pragma: no cover
        ...

    @abc.abstractmethod
    def root(self, value: T) -> V:  # pragma: no cover
        ...


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


def reduce_runtime_value(
    obj: Any, acc: Accumulator[T, V], string_hashcon_length: int = 32
) -> V:
    cnt: int = -1
    # id -> position
    memo: dict[int, int | None] = {}
    # Since we rely on `id` to detect duplicates we have to hold on to all the
    # intermediate values to make sure addresses do not get reused
    transient: list[Any] = []

    constant = acc.constant
    reference = acc.reference
    sequence = acc.sequence
    mapping = acc.mapping
    custom = acc.custom

    def _reduce_arg_values(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Iterator[T]:
        # Avoid the `id` being reused by other transient values
        for it in (args, kwargs.values()):
            for arg in it:
                if not isinstance(arg, int | float | None | bool | bytes | str):
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
            res = sequence(len(v), (reduce(x) for x in v))
        elif ty is dict:
            res = mapping(len(v), _gen_kv(v.items()))
        else:
            deconstructor = _get_custom(ty)
            fn, args, kwargs = deconstructor(v)
            # TODO: fix me
            res = custom(
                utils.get_locate_name(fn),
                size=len(args) + len(kwargs),
                kwnames=kwargs,
                values=_reduce_arg_values(args, kwargs),
            )
        memo[addr] = current
        return res

    _gen_kv = _mk_kv_reducer(reduce)

    return acc.root(reduce(obj))


# Helping out mypy a bit
_mk_list: Callable[[Iterable[Any]], Any] = list
_mk_dict: Callable[[Iterable[tuple[Any, Any]]], Any] = dict


class RuntimeValueBuilder(Accumulator[Any, Any]):
    """An accumulator that build runtime values."""

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

    def _custom(
        self,
        constructor: str,
        size: int,
        kwnames: Collection[str],
        values: Iterator[Any],
    ) -> Any:
        args, kwargs = group_args(size, kwnames, values)
        imploder = _get_named_imploder(constructor)
        if imploder is None:
            raise LookupError(f"Constructor not found: {constructor!r}")
        return imploder(*args, **kwargs)

    def _reduce(
        self, fn: Callable[P, Any], *args: P.args, **kwargs: P.kwargs
    ) -> Any:
        current = len(self.memo)
        self.memo.append(None)
        res = fn(*args, **kwargs)
        self.memo[current] = res
        return res

    def custom(
        self,
        constructor: str,
        size: int,
        kwnames: Collection[str],
        values: Iterator[Any],
    ) -> Any:
        return self._reduce(self._custom, constructor, size, kwnames, values)

    def sequence(self, size: int, items: Iterable[T]) -> Any:
        return self._reduce(_mk_list, items)

    def mapping(self, size: int, items: Iterable[tuple[Any, Any]]) -> Any:
        return self._reduce(_mk_dict, items)

    def root(self, obj: Any) -> Any:
        return obj


def copy(v: T) -> T:
    """Copy a value using its representation.

    ``copy(v)`` is equivalent to ``load_text(dump_text(v))``.

    Args:
      v:
    """
    # For performance reasons we might want to add the ability to not copy
    # immutable values.
    res: T = reduce_runtime_value(v, RuntimeValueBuilder())
    return res
