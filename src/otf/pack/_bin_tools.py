"""
Implementation of ``pack.otf.bin.dis``
======================================

This module contains the implementation of ``pack.otf.dis``. We rely heavily on
``msgpack-python``'s internal here so this should not be considered stable
enough to be used in production code.

"""
from __future__ import annotations

import contextlib
import dataclasses
import enum
import pickle
import sys
from typing import Any, Generic, Iterable, Iterator, TextIO, TypeVar

import msgpack
from msgpack import fallback

from otf import utils

T = TypeVar("T")

__all__ = ("dis",)


class ParseError(ValueError):
    pass


decode_long = pickle.decode_long


@enum.unique
class PackType(enum.IntEnum):
    IMMEDIATE = fallback.TYPE_IMMEDIATE  # type: ignore[attr-defined]
    ARRAY = fallback.TYPE_ARRAY  # type: ignore[attr-defined]
    MAP = fallback.TYPE_MAP  # type: ignore[attr-defined]
    RAW = fallback.TYPE_RAW  # type: ignore[attr-defined]
    BIN = fallback.TYPE_BIN  # type: ignore[attr-defined]
    EXT = fallback.TYPE_EXT  # type: ignore[attr-defined]


def srepr(obj: Any) -> str:
    return utils.cram(repr(obj), 60)


@dataclasses.dataclass
class MsgpackFrame:
    typ: PackType
    count: int | None
    obj: Any
    raw: bytearray

    def __str__(self) -> str:
        match self.typ:
            case PackType.IMMEDIATE:
                if type(self.obj) in (bool, type(None)):
                    return repr(self.obj)
                return f"{type(self.obj).__name__}({self.obj})"
            case PackType.RAW:
                return f"Raw({srepr(bytes(self.obj))})"
            case PackType.BIN:
                return f"Bin({srepr(bytes(self.obj))})"
            case PackType.ARRAY:
                return f"Array(len={self.count})"
            case PackType.MAP:
                return f"Map(len={self.count})"
            case PackType.EXT:
                return f"ExtType({self.count}, {srepr(bytes(self.obj))})"
        assert False, self.typ  # pragma: no cover


@dataclasses.dataclass
class OtfFrame:
    description: str
    indent: int
    origin: tuple[MsgpackFrame, ...]


def _gen_msgpack_frames(unpacker: fallback.Unpacker) -> Iterator[MsgpackFrame]:
    start = unpacker._buff_i  # type: ignore[attr-defined]
    _typ, n, obj = unpacker._read_header()  # type: ignore[attr-defined]
    end = unpacker._buff_i  # type: ignore[attr-defined]
    raw = unpacker._buffer[start:end]  # type: ignore[attr-defined]
    typ = PackType(_typ)
    yield MsgpackFrame(typ=typ, count=n, obj=obj, raw=raw)
    if typ == PackType.ARRAY:
        for _ in range(n):
            yield from _gen_msgpack_frames(unpacker)
    if typ == PackType.MAP:
        for _ in range(2 * n):
            yield from _gen_msgpack_frames(unpacker)


class PeakableIterator(Generic[T]):
    """Like a normal iterator but allows you to 'peek' at the next value

    This is useful for writing parsers with look-ahead.
    """

    future: list[T]
    underlying: Iterator[T] | None

    def __init__(self, underlying: Iterable[T]):
        self.future = []
        self.underlying = iter(underlying)

    def __iter__(self) -> Iterator[T]:
        return self

    def _get(self) -> T:
        if self.underlying is None:
            raise StopIteration
        try:
            return next(self.underlying)
        except StopIteration:
            self.underlying = None
            raise

    def __next__(self) -> T:
        if self.future:
            return self.future.pop()
        return self._get()

    def peek(self) -> T | None:
        if self.future:
            return self.future[0]
        try:
            v = self._get()
            self.future.append(v)
            return v
        except StopIteration:
            return None


def gen_otf_frames(raw: bytes) -> Iterator[OtfFrame]:
    unpacker = fallback.Unpacker(max_buffer_size=len(raw))
    unpacker.feed(raw)
    msgp_frames = PeakableIterator(_gen_msgpack_frames(unpacker))
    from .bin import Ext

    def gen(indent: int) -> Iterator[OtfFrame]:
        match next(msgp_frames):
            case MsgpackFrame(PackType.ARRAY, count=count) as f:
                assert count is not None
                if count > 0:
                    match msgp_frames.peek():
                        case MsgpackFrame(
                            PackType.EXT,
                            count=(Ext.CUSTOM | Ext.INTERNED_CUSTOM) as typ,
                            obj=data,
                        ) as f2:
                            next(msgp_frames)
                            if typ == Ext.CUSTOM:
                                description = (
                                    f"CUSTOM({data.decode('utf-8')!r}):"
                                )
                            else:
                                assert typ == Ext.INTERNED_CUSTOM
                                description = (
                                    f"INTERNED_CUSTOM({decode_long(data)}):"
                                )
                            yield OtfFrame(
                                indent=indent,
                                description=description,
                                origin=(f, f2),
                            )
                            for _ in range(count - 1):
                                yield from gen(indent + 1)
                            return
                yield OtfFrame(
                    indent=indent,
                    description="LIST:",
                    origin=(f,),
                )
                for _ in range(count):
                    yield from gen(indent + 1)
            case MsgpackFrame(PackType.MAP, count=count) as f:
                yield OtfFrame(
                    indent=indent,
                    description="MAP:",
                    origin=(f,),
                )
                assert count is not None
                for _ in range(count * 2):
                    yield from gen(indent + 1)
            case MsgpackFrame(PackType.RAW, obj=obj) as f:
                yield OtfFrame(
                    indent=indent,
                    description=repr(obj.decode("utf-8")),
                    origin=(f,),
                )
            case MsgpackFrame(PackType.BIN, obj=obj) as f:
                yield OtfFrame(
                    indent=indent,
                    description=repr(bytes(obj)),
                    origin=(f,),
                )
            case MsgpackFrame(PackType.IMMEDIATE, obj=obj) as f:
                yield OtfFrame(
                    indent=indent,
                    description=repr(obj),
                    origin=(f,),
                )
            case MsgpackFrame(PackType.EXT, count=Ext.LONG, obj=data) as f:
                yield OtfFrame(
                    indent=indent,
                    description=repr(decode_long(data)),
                    origin=(f,),
                )
            case MsgpackFrame(PackType.EXT, count=Ext.REF, obj=data) as f:
                yield OtfFrame(
                    indent=indent,
                    description=f"REF({decode_long(data)})",
                    origin=(f,),
                )
            case MsgpackFrame(PackType.EXT, count=ext_num) as f:
                raise ParseError(f"Unknown msgpack extension: {ext_num}")
            case f:  # pragma: no cover
                # Unreachable
                assert False, f

    try:
        yield from gen(0)
    except msgpack.exceptions.OutOfData:
        raise ParseError("Input truncated")
    if unpacker._got_extradata():  # type: ignore[attr-defined]
        raise ParseError("Garbage at the end of the file")
    with contextlib.suppress(StopIteration):
        next(msgp_frames)
        assert False, "Internal error"  # pragma: no cover


def hex_str(raw: bytes) -> str:
    """
    >>> hex_str(b"hello world!")
    '68 65 6C 6C 6F 20 77 6F 72 6C 64 21'
    """
    encoded = raw.hex().upper()
    return " ".join(encoded[i : i + 2] for i in range(0, len(encoded), 2))


def dis(raw: bytes, out: TextIO = sys.stdout, level: int = 1) -> None:
    digits = 4
    try:
        for idx, of in enumerate(gen_otf_frames(raw), 1):
            if level >= 2:
                if idx > 1:
                    out.write("\n")
                for j, mf in enumerate(of.origin):
                    if j > 0:
                        out.write("\n")
                    if level >= 3:
                        out.write("raw:     ")
                        out.write(hex_str(mf.raw))
                        out.write("\n")
                    out.write("msgpack: ")
                    out.write(str(mf))
                    out.write("\n")
                out.write("otf:     ")
            out.write(str(idx).zfill(digits))
            out.write(": ")
            out.write("  " * of.indent)
            out.write(of.description)
            out.write("\n")
    except ParseError as v:
        out.write("ERROR: ")
        out.write(str(v))
        out.write("\n")
    out.flush()
