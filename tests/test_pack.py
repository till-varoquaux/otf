import copyreg
import io
import math
import pickle
import pickletools
import sys

import pytest

from otf import pack

LOREM_IPSUM = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo
consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum
dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident,
sunt in culpa qui officia deserunt mollit anim id est laborum.
"""

QUICK_BROWN_FOX = "The quick brown fox jumps over the lazy dog"


@pytest.fixture(autouse=True)
def _auto_clean(monkeypatch):
    old_dispatch_table = copyreg.dispatch_table.copy()
    monkeypatch.setattr(pack, "DISPATCH_TABLE", pack.DISPATCH_TABLE.copy())
    yield
    copyreg.dispatch_table.clear()
    copyreg.dispatch_table.update(old_dispatch_table)


def dis(x):
    "Helper function that dumps out how pickle would serialize a given value."
    s = pickletools.optimize(pickle.dumps(x))
    buf = io.StringIO()
    pickletools.dis(s, buf)
    raise AssertionError(buf.getvalue())


def roundtrip(v):
    v2 = pack.copy(v)
    if v != v:
        assert math.isnan(v)
        assert math.isnan(v2)
    else:
        assert v == v2
    assert pickle.dumps(v) == pickle.dumps(v2)
    return v2


class A:
    pass


class MyInt(int):
    pass


def test_simple():
    with pytest.raises(TypeError):
        pack.explode(A())
    with pytest.raises(TypeError):
        pack.implode(A())
    assert pack.explode(5) == 5
    assert (
        pack.explode([5, None, "a"])
        == [5, None, "a"]
        == pack.implode([5, None, "a"])
        == pack.copy([5, None, "a"])
    )
    assert pack.explode({5: None}) == {5: None} == pack.implode({5: None})
    with pytest.raises(TypeError):
        pack.explode(MyInt(5))


def test_shared():
    v = [1, 2]
    exploded = [[1, 2], pack.Reference(3), pack.Reference(1)]
    assert pack.explode([v, v, v]) == exploded
    v2 = roundtrip([v, v, v])
    assert v2[0] is v2[1] is v2[2]


def test_mapping():
    k1 = (1, 2)
    k2 = (2, 3)
    v = {k1: k2, k2: None}
    roundtrip(v)


@pytest.mark.parametrize(
    "x",
    (
        math.nan,
        math.inf,
        -math.inf,
        -0.0,
        0.0,
        1 / 3,
        sys.float_info.min,
        sys.float_info.max,
    ),
)
def test_weird_floats(x):
    roundtrip(x)
    roundtrip({x: x})


def test_hash_cons_str():
    # copy.copy will actually do nothing. We want to get a new copy of the same
    # string.
    lorem_copy = "\n".join(LOREM_IPSUM.split("\n"))
    assert lorem_copy == LOREM_IPSUM
    assert id(lorem_copy) != id(LOREM_IPSUM)
    assert pack.explode(
        [lorem_copy, LOREM_IPSUM, lorem_copy, QUICK_BROWN_FOX]
    ) == [LOREM_IPSUM, pack.Reference(1), pack.Reference(1), QUICK_BROWN_FOX]


def test_reccursive():
    v = []
    v.append(v)
    with pytest.raises(ValueError):
        pack.explode(v)


def test_tuple():
    v = 1, 2
    assert pack.explode(v) == pack.Custom("tuple", [1, 2])
    assert pack.cexplode(v) == [1, 2]
    assert pack.cimplode(tuple, [1, 2]) == (1, 2)
    assert pack.implode(pack.Custom("tuple", [1, 2])) == (1, 2)


def test_infer_reducer_type():
    def f(x: list[int]):
        pass

    assert pack._infer_reducer_type(f) is list

    def f(x, y):
        pass

    with pytest.raises(ValueError):
        assert pack._infer_reducer_type(f)


class ItemGetter:
    # A simple python class that can't be pickled by default
    def __init__(self, item):
        self._item = item

        def func(obj):
            return obj[item]

        self._call = func

    def __call__(self, obj):
        return self._call(obj)


class ItemGetter2(ItemGetter):
    pass


def test_register_pickle():
    v = ItemGetter(2)
    with pytest.raises(Exception, match="Can't pickle"):
        pickle.dumps(v)

    @pack.register(pickle=True)
    def reduce_item_getter(i: ItemGetter):
        return ItemGetter, i._item

    v2 = pickle.loads(pickle.dumps(v))

    assert v2([0, 1, 2, 3, 4]) == 2

    # Inheritance doesn't work
    with pytest.raises(Exception, match="Can't pickle"):
        pickle.dumps(ItemGetter2(2))


class IdGetter:
    def __init__(self):
        self.value = id(self)

    @classmethod
    def _otf_reconstruct(cls, value: int):
        v = cls.__new__(cls)
        v.value = value
        return v


def test_register_random_holder():

    rg = IdGetter()

    @pack.register(pickle=True)
    def reduce_random_getter(i: IdGetter):
        return IdGetter, i.value

    reconstructed = pickle.loads(pickle.dumps(rg))
    reconstructed2 = pack.copy(rg)
    assert reconstructed.value == reconstructed2.value == rg.value
