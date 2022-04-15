"""
``otf.runtime``: Support functions for the runtime
==================================================
"""

import dataclasses
import typing
from typing import Any, Callable, Generic, ParamSpec, TypeVar

from . import compiler, pack, utils

P = ParamSpec("P")
T = TypeVar("T")

__all__ = (
    "NamedReference",
    "Task",
    "task",
)

# For the docstring
dumps = pack.dumps


class NamedReference(Generic[utils.Addressable]):
    """Serialisable wrapper around a module, class or function.

    :class:`NamedReference` are serialised by copying the name to the wrapped
    object::

       >>> import math
       >>> fl = NamedReference(math.floor)
       >>> dumps(fl)
       "otf.NamedReference('math.floor')"

    If the wrapped object is callable then the wrapped will pass calls
    transparently to the object::

       >>> fl(12.6)
       12

    Attribute accesses are also transparently passed through to the wrapped
    value::

       >>> m = NamedReference(math)
       >>> m.ceil(5.6)
       6

    This means that, in most cases, the :class:`NamedReference` wrapper is
    transparent. Sometimes you actually need to access the wrapped value::

       >>> WrappedComplex = NamedReference(complex)
       >>> k = WrappedComplex(5)
       >>> isinstance(k, WrappedComplex)
       Traceback (most recent call last):
         ...
       TypeError: isinstance() arg 2 must be a type, a tuple of types, or a \
union
       >>>

    You can use the ``~`` operator to access the wrapped value::

       >>> ~WrappedComplex
       <class 'complex'>
       >>> isinstance(k, ~WrappedComplex)
       True

    Args:
       v (class|module|function): The object that will be wrapped.

    Raises:

       ValueError: if *obj* cannot be reloaded via its name (e.g.: if *obj* is
         defined inside a function).

       TypeError: if *obj* is not a module, class or function or if *obj* is a
         lambda.
    """

    __slots__ = ("_value", "_name")

    _name: str
    _value: utils.Addressable

    def __init__(self, obj: utils.Addressable) -> None:
        object.__setattr__(self, "_value", obj)
        object.__setattr__(self, "_name", utils.get_locate_name(obj))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return typing.cast(Callable[..., Any], self._value)(*args, **kwargs)

    def __repr__(self) -> str:
        return f"NamedReference({self._value!r})"

    def __str__(self) -> str:
        return f"NamedReference({self._value})"

    def __eq__(self, other: Any) -> Any:
        if type(self) != type(other):
            return NotImplemented
        return self._value == other._value

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            return object.__getattribute__(self, name)
        return getattr(self._value, name)

    def __setattr__(self, name: str, value: Any) -> Any:
        raise AttributeError("Cannot assign to fields of a wrapped value")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Cannot delete fields of a wrapped value")

    def __invert__(self) -> utils.Addressable:
        """Get the wrapped value."""
        return self._value

    @staticmethod
    def _otf_reconstruct(name: str) -> "NamedReference[Any]":
        res = NamedReference.__new__(NamedReference)
        object.__setattr__(res, "_name", name)
        object.__setattr__(res, "_value", utils.locate(name))
        return res


@pack.register(pickle=True)
def _explode_namedref(namedref: NamedReference[Any]) -> tuple[Any, str]:
    return NamedReference, namedref._name


@dataclasses.dataclass
class Task(Generic[T]):
    """A deferred function application"""

    closure: compiler.Closure[Any, T]
    args: list[Any]
    kwargs: dict[str, Any]

    def run(self) -> T:
        return self.closure(*self.args, **self.kwargs)


def task(
    fn: compiler.Closure[P, T] | compiler.Function[P, T],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> Task[T]:
    """Create a :class:`Task`.

    This is useful when implementing schedulers. We capture the function, the
    arguments and the environment it runs in.

    """
    if isinstance(fn, compiler.Function):
        closure = compiler.Closure[P, T](
            environment=compiler._OtfEnv.get(), target=fn
        )
    else:
        closure = fn
    return Task(closure=closure, args=[*args], kwargs=dict(**kwargs))
