from __future__ import annotations

import msgpack
import pytest

from otf.pack import bin, tree

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
