"""
``otf.pack.bin``: msgpack based binary format
=============================================

Encode values in a binary format.

The binary format used is an extension of `MessagePack <https://msgpack.org/>`_
"""

from __future__ import annotations

import pickle
import typing

if typing.TYPE_CHECKING:  # pragma: no cover
    import types
    from typing import Any, Collection, Final, Iterator, TextIO, Type, TypeVar

    import msgpack

    T = TypeVar("T")
    V = TypeVar("V")


from otf.pack import base, tree

__all__ = (
    "dump_bin",
    "load_bin",
    "Ext",
    "BinPacker",
    "reduce_bin",
    "dis",
)

MAX_RAW_INT: Final = 2**64 - 1
MIN_RAW_INT: Final = -(2**63)

encode_long: Final = pickle.encode_long
decode_long: Final = pickle.decode_long


class ImportGuard:
    def __enter__(self) -> None:
        pass

    def __exit__(
        self,
        exctype: Type[BaseException] | None,
        excinst: BaseException | None,
        exctb: types.TracebackType | None,
    ) -> None:
        if exctype is not None and issubclass(exctype, ModuleNotFoundError):
            import warnings

            warnings.warn(
                "Support for binary serialisation is not available because of "
                "missing dependencies. You can fix this by running ``pip "
                "install otf[msgpack]``"
            )


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
        with ImportGuard():
            import msgpack

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


def reduce_bin(packed: bytes, acc: base.Accumulator[T, V]) -> V:
    """Read a binary encoded value."""
    constant = acc.constant
    reference = acc.reference
    sequence = acc.sequence
    mapping = acc.mapping
    custom = acc.custom
    cnt = 0
    shapes: dict[int, list[str]] = {}

    with ImportGuard():
        import msgpack

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


def dump_bin(obj: Any) -> bytes:
    """Serialise *obj* to the binary format

    Note:

      This feature is only available if otf was installed with ``msgpack``
      (e.g.: via ``pip install otf[msgpack]``).
    """
    return base.reduce_runtime_value(obj, BinPacker())


def load_bin(packed: bytes) -> Any:
    """Read an object written in binary format

    Note:

      This feature is only available if otf was installed with ``msgpack``
      (e.g.: via ``pip install otf[msgpack]``).

    """
    return reduce_bin(packed, base.RuntimeValueBuilder())


def dis(raw: bytes, out: TextIO | None = None, level: int = 1) -> None:
    """Output a disassembly of **raw**.

    warning:

        This function is for diagnose purposes only. We make no guarantees that
        the output format will not change.


    The level argument controls how much details get printed. At ``level<=1``
    only the Otf instructions are printed:

        >>> dis(dump_bin((1, 2, 3)))
        0001: CUSTOM('tuple'):
        0002:   LIST:
        0003:     1
        0004:     2
        0005:     3

    At ``level=2`` the msgpack instruction are also printed:

        >>> dis(dump_bin((1, 2, 3)), level=2)
        msgpack: Array(len=2)
        <BLANKLINE>
        msgpack: ExtType(2, b'tuple')
        otf:     0001: CUSTOM('tuple'):
        <BLANKLINE>
        msgpack: Array(len=3)
        otf:     0002:   LIST:
        <BLANKLINE>
        msgpack: int(1)
        otf:     0003:     1
        <BLANKLINE>
        msgpack: int(2)
        otf:     0004:     2
        <BLANKLINE>
        msgpack: int(3)
        otf:     0005:     3

    At ``level>=3`` a hexdump of the source is also included:

        >>> dis(dump_bin((1, 2, 3)), level=3)
        raw:     92
        msgpack: Array(len=2)
        <BLANKLINE>
        raw:     C7 05 02 74 75 70 6C 65
        msgpack: ExtType(2, b'tuple')
        otf:     0001: CUSTOM('tuple'):
        <BLANKLINE>
        raw:     93
        msgpack: Array(len=3)
        otf:     0002:   LIST:
        <BLANKLINE>
        raw:     01
        msgpack: int(1)
        otf:     0003:     1
        <BLANKLINE>
        raw:     02
        msgpack: int(2)
        otf:     0004:     2
        <BLANKLINE>
        raw:     03
        msgpack: int(3)
        otf:     0005:     3

    Arguments:

      raw(bytes): value to disassemble
      out(file-like):
        `text file <https://docs.python.org/3/glossary.html#term-text-file>`_
        where the output will be written (defaults to :data:`sys.stdout`).
      level(int): between ``1`` and ``3``.
    """
    with ImportGuard():
        import msgpack  # noqa: F401

    from . import _bin_tools

    if out is None:
        import sys

        out = sys.stdout
    return _bin_tools.dis(raw, out, level)
