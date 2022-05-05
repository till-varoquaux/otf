"""``otf.pack.tree``: In memory trees
==================================

Convert to and from :class:`Node`.

:mod:`otf.pack` convert directly between different types and doesn't have any
intermediate representation of values. Having access to a simple representation
makes it easier to write tests and debug application.


This is the easiest way to programmatically check what a python object looks
like when it's reduced::

  >>> v = (1, 2)
  >>> explode([(1, 2)])
  [Custom(constructor='tuple', args=([1, 2],), kwargs={})]

You can also leverage this module to check other reducers::

  >>> import otf.pack.text
  >>> otf.pack.reduce_text("{4: I.do_not.exist(1)}", NodeBuilder())
  {4: Custom(constructor='I.do_not.exist', args=(1,), kwargs={})}


API:
----

"""

from __future__ import annotations

import dataclasses
from typing import Any, Collection, Iterator, TypeVar

from otf.pack import base

T = TypeVar("T")
V = TypeVar("V")

__all__ = (
    "Node",
    "Mapping",
    "Reference",
    "Custom",
    "NodeBuilder",
    "reduce_tree",
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
    args: tuple[Node, ...]
    kwargs: dict[str, Node]

    def __init__(self, constructor: str, *args: Node, **kwargs: Node) -> None:
        object.__setattr__(self, "constructor", constructor)
        object.__setattr__(self, "args", args)
        object.__setattr__(self, "kwargs", kwargs)

    def nodes(self) -> Iterator[Node]:
        yield from self.args
        yield from self.kwargs.values()


@dataclasses.dataclass(slots=True, frozen=True)
class Mapping:
    """A key value container

    Usually mapping are represented as dictionaries:

        >>> explode({None: "empty"})  # All the exploded keys are hashable
        {None: 'empty'}

    When the reduced value cannot be safely represented as a dictionary (because
    some keys are not hashable) we use a :class:`Mapping` :

        >>> from pprint import pp
        >>> pp(explode({(): "empty"}), width=60)
        Mapping(data=[(Custom(constructor='tuple',
                              args=([],),
                              kwargs={}),
                       'empty')])

    Note that the mapping cannot implement the :class:`collections.Mapping`
    interface because it might contain duplicate keys:

        >>> v1 = (1,)
        >>> v2 = (2,)
        >>> e = explode({1: v1, v1: v2, v2: 2})
        >>> pp(e, width=60)
        Mapping(data=[(1,
                       Custom(constructor='tuple',
                              args=([1],),
                              kwargs={})),
                      (Reference(offset=3),
                       Custom(constructor='tuple',
                              args=([2],),
                              kwargs={})),
                      (Reference(offset=3), 2)])

    Parameters:
        data(list[tuple[Node, Node]]):
    """

    data: list[tuple[Node, Node]]

    def items(self) -> Iterator[tuple[Node, Node]]:
        yield from self.data

    def __len__(self) -> int:
        return len(self.data)


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


class NodeBuilder(base.Accumulator[Node, Node]):
    """A :class:`base.Accumulator` used to build :class:`Node`"""

    def constant(
        self, constant: int | float | None | str | bytes | bool
    ) -> Node:
        return constant

    def mapping(self, size: int, items: Iterator[tuple[Node, Node]]) -> Node:
        acc: dict[Any, Any] = {}
        for k, v in items:
            # Do we need to bail out and return a Mapping because the key is
            # un-hashable?
            if not isinstance(k, int | float | str | bytes | bool | None):
                return Mapping([*acc.items(), (k, v), *items])
            acc[k] = v
        return acc

    def sequence(self, size: int, items: Iterator[Node]) -> Node:
        return list(items)

    def reference(self, offset: int) -> Node:
        return Reference(offset)

    def custom(
        self,
        constructor: str,
        size: int,
        kwnames: Collection[str],
        values: Iterator[Node],
    ) -> Node:
        args, kwargs = base.group_args(size, kwnames, values)
        return Custom(constructor, *args, **kwargs)

    def root(self, node: Node) -> Node:
        return node


def reduce_tree(value: Node, acc: base.Accumulator[T, V]) -> V:
    constant = acc.constant
    reference = acc.reference
    sequence = acc.sequence
    mapping = acc.mapping
    custom = acc.custom

    def reduce(exp: Node) -> T:
        if isinstance(exp, (int, float, str, bytes, bool, type(None))):
            return constant(exp)
        elif isinstance(exp, list):
            return sequence(len(exp), (reduce(elt) for elt in exp))
        elif isinstance(exp, dict | Mapping):
            return mapping(len(exp), _gen_kv(exp.items()))
        elif isinstance(exp, Reference):
            return reference(exp.offset)
        elif isinstance(exp, Custom):
            return custom(
                exp.constructor,
                size=len(exp.args) + len(exp.kwargs),
                kwnames=tuple(exp.kwargs),
                values=(reduce(arg) for arg in exp.nodes()),
            )
        else:
            raise TypeError(
                f"Object of type {type(exp).__name__} cannot be de-serialised "
                "by OTF"
            )

    _gen_kv = base._mk_kv_reducer(reduce)

    return acc.root(reduce(value))


def explode(v: Any) -> Node:
    """Convert an object to a :class:`Node`

    Args:
      v:
    """
    return base.reduce_runtime_value(v, NodeBuilder())


def implode(v: Node) -> Any:
    """Reconstruct the object represented by a :class:`Node`

    Args:
      v:
    """
    return reduce_tree(v, base.RuntimeValueBuilder())
