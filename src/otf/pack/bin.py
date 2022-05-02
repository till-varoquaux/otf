"""
``otf.pack.bin``: msgpack based binary format
=============================================

Encode values in a binary format.

The binary format used is an extension of `MessagePack <https://msgpack.org/>`_
"""

from __future__ import annotations

import pickle
from typing import Any, Collection, Final, Iterator, TypeVar

import msgpack

from otf.pack import base, tree

__all__ = (
    "dumpb",
    "loadb",
    "Ext",
    "BinPacker",
    "reduce",
)

T = TypeVar("T")
V = TypeVar("V")

MAX_RAW_INT: Final = 2**64 - 1
MIN_RAW_INT: Final = -(2**63)

encode_long: Final = pickle.encode_long
decode_long: Final = pickle.decode_long


class Ext:
    """The `messagepack extensions`_ used by *OTF*

    Attributes:

      LONG(): An :class:`int` that is too big (resp too small) to be encoded
        directly in msgpack. The payload is encoded as two's complement
        little-endian int.
      REF(): A shared references.
      CUSTOM(): Call to a custom constructor.
      INTERNED_CUSTOM(): Reuse the definition of a previous :attr:`CUSTOM`
    """

    LONG: Final = 0
    REF: Final = 1
    CUSTOM: Final = 2
    INTERNED_CUSTOM: Final = 3


class BinPacker(base.Accumulator[None, bytes]):
    """Writer for binary encoded values."""

    packer: msgpack.Packer
    size: int
    shapes: dict[tuple[str, ...], int]

    def __init__(self) -> None:
        self.packer = msgpack.Packer(autoreset=False)
        self.size = -1
        self.shapes = {}

    def constant(
        self, constant: int | float | None | str | bytes | bool
    ) -> None:
        self.size += 1
        if (
            isinstance(constant, int)
            and not MIN_RAW_INT <= constant <= MAX_RAW_INT
        ):
            self.packer.pack_ext_type(Ext.LONG, encode_long(constant))
        else:
            self.packer.pack(constant)

    def mapping(self, size: int, items: Iterator[tuple[None, None]]) -> None:
        self.size += 1
        self.packer.pack_map_header(size)
        for _ in items:
            pass

    def sequence(self, size: int, items: Iterator[None]) -> None:
        self.size += 1
        self.packer.pack_array_header(size)
        for _ in items:
            pass

    def reference(self, offset: int) -> None:
        self.size += 1
        self.packer.pack_ext_type(Ext.REF, encode_long(offset))

    def custom(
        self,
        constructor: str,
        size: int,
        kwnames: Collection[str],
        values: Iterator[None],
    ) -> None:
        self.size += 1
        self.packer.pack_array_header(size + 1)

        shape = (constructor, *kwnames)
        prev = self.shapes.get(shape, None)
        if prev is None:
            self.packer.pack_ext_type(
                Ext.CUSTOM, ":".join(shape).encode("utf-8")
            )
        else:
            self.packer.pack_ext_type(
                Ext.INTERNED_CUSTOM, encode_long(self.size - prev)
            )
        self.shapes[shape] = self.size
        for _ in values:
            pass

    def root(self, node: None) -> bytes:
        return self.packer.bytes()


def reduce(packed: bytes, acc: base.Accumulator[T, V]) -> V:
    """Read a binary encoded value."""
    constant = acc.constant
    reference = acc.reference
    sequence = acc.sequence
    mapping = acc.mapping
    custom = acc.custom
    cnt = 0
    shapes: dict[int, list[str]] = {}

    def reduce(exp: Any) -> T:
        nonlocal cnt
        cnt += 1
        match exp:
            case int() | float() | str() | bytes() | bool() | None:
                return constant(exp)
            case msgpack.ExtType(code=Ext.LONG, data=data):
                return constant(decode_long(data))
            case msgpack.ExtType(code=Ext.REF, data=data):
                return reference(decode_long(data))
            case tree.Mapping():
                return mapping(len(exp), _gen_kv(exp.items()))
            case (
                msgpack.ExtType(
                    code=Ext.CUSTOM | Ext.INTERNED_CUSTOM as ext, data=data
                ),
                *_,
            ):
                if ext == Ext.CUSTOM:
                    shape = data.decode("utf-8").split(":")
                else:
                    assert ext == Ext.INTERNED_CUSTOM
                    shape = shapes[cnt - decode_long(data)]
                shapes[cnt] = shape
                constructor, *kwnames = shape
                size = len(exp) - 1
                args = iter(exp)
                next(args)  # Skip the header
                return custom(
                    constructor, size, kwnames, (reduce(arg) for arg in args)
                )
            case tuple():
                return sequence(len(exp), (reduce(elt) for elt in exp))
        # Unreachable
        assert False, exp  # pragma: no cover

    _gen_kv = base._mk_kv_reducer(reduce)

    value = msgpack.unpackb(
        packed,
        use_list=False,
        object_pairs_hook=tree.Mapping,
        strict_map_key=False,
    )
    return acc.root(reduce(value))


def dumpb(obj: Any) -> bytes:
    """Serialise *obj* to the binary format

    Note:

      This feature is only available if otf was installed with ``msgpack``
      (e.g.: via ``pip install otf[msgpack]``).
    """
    return base.reduce(obj, BinPacker())


def loadb(packed: bytes) -> Any:
    """Read an object written in binary format

    Note:

      This feature is only available if otf was installed with ``msgpack``
      (e.g.: via ``pip install otf[msgpack]``).

    """
    return reduce(packed, base.RuntimeValueBuilder())
