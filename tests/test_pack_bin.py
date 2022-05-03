from __future__ import annotations

import io
from typing import Any, NamedTuple

import msgpack
import msgpack.fallback
import pytest

from otf.pack import _bin_tools, bin, tree

from . import pack_utils


def test_msgpack_bounds():
    msgpack.packb(bin.MAX_RAW_INT)
    with pytest.raises(OverflowError):
        msgpack.packb(bin.MAX_RAW_INT + 1)

    msgpack.packb(bin.MIN_RAW_INT)
    with pytest.raises(OverflowError):
        msgpack.packb(bin.MIN_RAW_INT - 1)


@pytest.mark.parametrize(
    "value",
    (
        [1, 2, 3],
        {1: 1, 2: 2},
        2**80,
        float("inf"),
        pack_utils.Sig(1, 2, 3, x="x", y="y"),
    ),
)
def test_roundtrip(value):
    packed = bin.dumpb(value)
    assert bin.loadb(packed) == value


def test_ref():
    e = []
    v = [e, e, []]
    packed = bin.dumpb(v)
    assert bin.reduce(packed, tree.NodeBuilder()) == [[], tree.Reference(1), []]


def test_interned_shape():
    v = [pack_utils.Sig(x, double=2 * x, square=x * x) for x in range(1, 10)]
    packed = bin.dumpb(v)
    # Shape interning keeps us nice and trim
    assert len(packed) < 10 * (len("double") + len("square"))
    assert bin.loadb(packed) == v


def test_module_not_found():

    with pytest.warns(UserWarning), pytest.raises(
        ModuleNotFoundError
    ), bin.ImportGuard():
        import adfasdfasdf  # noqa: F401


def test_msgpack_frames():
    def desc(value):
        packed = bin.dumpb(value)
        unpacker = msgpack.fallback.Unpacker()
        unpacker.feed(packed)
        return [str(f) for f in _bin_tools._gen_msgpack_frames(unpacker)]

    assert desc([False, None, 2, "string", b"bytes"]) == [
        "Array(len=5)",
        "False",
        "None",
        "int(2)",
        "Raw(b'string')",
        "Bin(b'bytes')",
    ]
    assert desc({1: 1}) == ["Map(len=1)", "int(1)", "int(1)"]

    assert desc(2**65) == [
        r"ExtType(0, b'\x00\x00\x00\x00\x00\x00\x00\x00\x02')"
    ]


def test_peakable_iterator():
    def peakable_range(v):
        return _bin_tools.PeakableIterator(range(v))

    assert list(peakable_range(5)) == [0, 1, 2, 3, 4]

    pr = peakable_range(3)
    assert pr.peek() == 0
    assert pr.peek() == 0

    assert next(pr) == 0
    assert next(pr) == 1
    assert next(pr) == 2

    assert pr.peek() is None

    with pytest.raises(StopIteration):
        next(pr)


@pytest.mark.parametrize(
    "value",
    (
        [1, 2, 3],
        {1: 1, 2: 2},
        2**80,
        float("inf"),
        pack_utils.Sig(1, 2, 3, x="x", y="y"),
    ),
)
def test_frames_round_trip(value):
    packed = bin.dumpb(value)
    reconstituted = b"".join(
        msg_pack_frame.raw
        for otf_frame in _bin_tools.gen_otf_frames(packed)
        for msg_pack_frame in otf_frame.origin
    )
    assert reconstituted == packed


class Raw(NamedTuple):
    value: bytes


def dis(value: Any, level=1) -> str:
    if isinstance(value, Raw):
        packed = value.value
    else:
        packed = bin.dumpb(value)
    buf = io.StringIO()
    bin.dis(packed, buf, level=level)
    return buf.getvalue()


V_DIS3 = r"""
raw:     92
msgpack: Array(len=2)
otf:     0001: LIST:

raw:     92
msgpack: Array(len=2)

raw:     C7 05 02 74 75 70 6C 65
msgpack: ExtType(2, b'tuple')
otf:     0002:   CUSTOM('tuple'):

raw:     90
msgpack: Array(len=0)
otf:     0003:     LIST:

raw:     D4 01 02
msgpack: ExtType(1, b'\x02')
otf:     0004:   REF(2)
"""

V_DIS2 = r"""
msgpack: Array(len=2)
otf:     0001: LIST:

msgpack: Array(len=2)

msgpack: ExtType(2, b'tuple')
otf:     0002:   CUSTOM('tuple'):

msgpack: Array(len=0)
otf:     0003:     LIST:

msgpack: ExtType(1, b'\x02')
otf:     0004:   REF(2)
"""

V_DIS1 = r"""
0001: LIST:
0002:   CUSTOM('tuple'):
0003:     LIST:
0004:   REF(2)
"""


def test_dis_levels():
    v = ()
    assert dis([v, v], 3) == V_DIS3[1:]
    assert dis([v, v], 2) == V_DIS2[1:]
    assert dis([v, v], 1) == V_DIS1[1:]


def stripped_dis(value):
    res = []
    for line in dis(value).split("\n")[:-1]:
        if line[0].isdigit():
            res.append(line[5:].strip())
        else:
            res.append(line)
    return res


def test_dis():
    assert stripped_dis(5) == ["5"]
    assert stripped_dis(
        [pack_utils.Sig(x="x", y="y"), pack_utils.Sig(x="X", y="Y")]
    ) == [
        "LIST:",
        "CUSTOM('tests.pack_utils.Sig:x:y'):",
        "'x'",
        "'y'",
        "INTERNED_CUSTOM(3):",
        "'X'",
        "'Y'",
    ]
    assert stripped_dis(b"123") == ["b'123'"]

    assert stripped_dis(Raw(bin.dumpb([1, 2, 3]) + b"garbage")) == [
        "LIST:",
        "1",
        "2",
        "3",
        "ERROR: Garbage at the end of the file",
    ]

    assert stripped_dis(Raw(bin.dumpb([1, 2, 3])[:-1])) == [
        "LIST:",
        "1",
        "2",
        "ERROR: Input truncated",
    ]

    assert stripped_dis(
        Raw(msgpack.packb([1, 2, 3, 4, msgpack.ExtType(42, b""), 6, 7]))
    ) == ["LIST:", "1", "2", "3", "4", "ERROR: Unknown msgpack extension: 42"]
