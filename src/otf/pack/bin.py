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
    """

    LONG: Final = 0
    REF: Final = 1
    CUSTOM: Final = 2


class BinPacker(base.Accumulator[None, bytes]):
    """Writer for binary encoded values."""

    packer: msgpack.Packer

    def __init__(self) -> None:
        self.packer = msgpack.Packer(autoreset=False)

    def constant(
        self, constant: int | float | None | str | bytes | bool
    ) -> None:
        # use pickle.encode_long/pickle_decode_long
        if (
            isinstance(constant, int)
            and not MIN_RAW_INT <= constant <= MAX_RAW_INT
        ):
            self.packer.pack_ext_type(Ext.LONG, encode_long(constant))
        else:
            self.packer.pack(constant)

    def mapping(self, size: int, items: Iterator[tuple[None, None]]) -> None:
        self.packer.pack_map_header(size)
        for _ in items:
            pass

    def sequence(self, size: int, items: Iterator[None]) -> None:
        self.packer.pack_array_header(size)
        for _ in items:
            pass

    def reference(self, offset: int) -> None:
        raw = encode_long(offset)
        self.packer.pack_ext_type(Ext.REF, raw)

    def custom(
        self,
        constructor: str,
        size: int,
        kwnames: Collection[str],
        values: Iterator[None],
    ) -> None:
        self.packer.pack_array_header(size + 1)
        self.packer.pack_ext_type(
            Ext.CUSTOM, ":".join((constructor, *kwnames)).encode("utf-8")
        )
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

    def reduce(exp: Any) -> T:
        match exp:
            case int() | float() | str() | bytes() | bool() | None:
                return constant(exp)
            case msgpack.ExtType(code=Ext.LONG, data=data):
                return constant(decode_long(data))
            case msgpack.ExtType(code=Ext.REF, data=data):
                return reference(decode_long(data))
            case tree.Mapping():
                return mapping(len(exp), _gen_kv(exp.items()))
            case (msgpack.ExtType(code=Ext.CUSTOM, data=data), *_):
                constructor, *kwnames = data.decode("utf-8").split(":")
                size = len(exp) - 1
                args = iter(exp)
                next(args)
                return custom(
                    constructor, size, kwnames, (reduce(arg) for arg in args)
                )
            case tuple():
                return sequence(len(exp), (reduce(elt) for elt in exp))
        # Unreachable
        assert False  # pragma: no cover

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
