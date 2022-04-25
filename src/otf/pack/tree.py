"""``otf.pack.tree``: In memory trees
==================================

Convert to and from :class:`Node`.

:mod:`otf.pack` convert directly between different types and doesn't have any
intermediate representation of values. Having access to a simple representation
makes it easier to write tests and debug application.


This is the easiest way to programatically check what a python object looks like
when it's reduced::

  >>> v = (1, 2)
  >>> explode([(1, 2)])
  [Custom(constructor='tuple', value=[1, 2])]

You can also leverage this module to check other reducers::

  >>> import otf.pack.text
  >>> otf.pack.text.reduce("{4: I.do_not.exist(1)}", NodeBuilder())
  {4: Custom(constructor='I.do_not.exist', value=1)}


API:
----
"""

from __future__ import annotations

import dataclasses
from typing import Any, Iterator, TypeVar

from otf.pack import base

T = TypeVar("T")

__all__ = (
    "Node",
    "Mapping",
    "Reference",
    "Custom",
    "NodeBuilder",
    "reduce",
    "explode",
    "implode",
)


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

      value(Node): A serialisable value that can be passed as an argument to
        *constructor* to recreate the original value

    """

    constructor: str
    value: Node


@dataclasses.dataclass(slots=True, frozen=True)
class Mapping:
    """A key value container

    Usually mapping are represented as dictionaries:

        >>> explode({None: "empty"})  # All the exploded keys are hashable
        {None: 'empty'}

    When the reduced value cannot be safely represented as a dictionary (because
    some keys are not hashable) we use a :class:`Mapping` :

        >>> explode({(): "empty"})
        Mapping(data=[(Custom(constructor='tuple', value=[]), 'empty')])

    Note that the mapping cannot implement the :class:`collections.Mapping`
    interface because it might contain duplicate keys:

        >>> v1 = (1,)
        >>> v2 = (2,)
        >>> e = explode({1: v1, v1: v2, v2: 2})
        >>> from pprint import pp
        >>> pp(e, width=60)
        Mapping(data=[(1, Custom(constructor='tuple', value=[1])),
                      (Reference(offset=3),
                       Custom(constructor='tuple', value=[2])),
                      (Reference(offset=3), 2)])

    Parameters:
        data(list[tuple[Node, Node]]):
    """

    data: list[tuple[Node, Node]]

    def items(self) -> Iterator[tuple[Node, Node]]:
        yield from self.data


# Mypy doesn't support recursive types
# (https://github.com/python/mypy/issues/731)

# We could use Protocol like in:
# https://github.com/python/typing/issues/182#issuecomment-893657366 but, in
# practice, this turned out to be a bigger headache than the problem it was
# trying to solve.

#:
Node = (
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


class NodeBuilder(base.Accumulator[Node]):
    """A :class:`base.Accumulator` used to build :class:`Node`"""

    def constant(
        self, constant: int | float | None | str | bytes | bool
    ) -> Node:
        return constant

    def mapping(self, items: Iterator[tuple[Node, Node]]) -> Node:
        acc: dict[Any, Any] = {}
        for k, v in items:
            # Do we need to bail out and return a Mapping because the key is
            # un-hashable?
            if not isinstance(k, int | float | str | bytes | bool | None):
                return Mapping([*acc.items(), (k, v), *items])
            acc[k] = v
        return acc

    def sequence(self, items: Iterator[Node]) -> Node:
        return list(items)

    def reference(self, offset: int) -> Node:
        return Reference(offset)

    def custom(self, constructor: str, value: Iterator[Node]) -> Node:
        (arg,) = value
        return Custom(constructor, arg)


def reduce(value: Node, acc: base.Accumulator[T]) -> T:
    constant = acc.constant
    reference = acc.reference
    sequence = acc.sequence
    mapping = acc.mapping
    custom = acc.custom

    def reduce(exp: Node) -> T:
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

    _gen_kv = base._mk_kv_reducer(reduce)

    def _custom_arg(arg: Any) -> Iterator[T]:
        yield reduce(arg)

    return acc.root(reduce(value))


def explode(v: Any) -> Node:
    """Convert an object to a :class:`Node`

    Args:
      v:
    """
    return base.reduce(v, NodeBuilder())


def implode(v: Node) -> Any:
    """Reconstruct the object represented by a :class:`Node`

    Args:
      v:
    """
    return reduce(v, base.RuntimeValueBuilder())
