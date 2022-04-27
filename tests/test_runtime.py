from __future__ import annotations

import math
import pickle

import pytest

import otf
from otf import pack, runtime


class K:

    _protected = 2
    __dunder__ = 5

    def __init__(self, v):
        self.v = v


def test_namedref():
    assert K.__dunder__ == 5
    Kr = runtime.NamedReference(K)
    with pytest.raises(AttributeError):
        Kr.__dunder__

    assert Kr._protected == 2
    assert ~Kr == K

    assert str(Kr) == "NamedReference(<class 'tests.test_runtime.K'>)"
    assert repr(Kr) == "NamedReference(<class 'tests.test_runtime.K'>)"

    assert Kr == runtime.NamedReference(K)
    assert Kr != 5

    k = Kr(5)
    assert k.v == 5

    with pytest.raises(AttributeError):
        Kr.v = 6

    with pytest.raises(AttributeError):
        del Kr.v


def test_serialize_namedref():
    mth = runtime.NamedReference(math)
    assert pack.tree.explode(mth) == pack.tree.Custom(
        "otf.NamedReference", "math"
    )
    assert mth.floor(mth.e) == 2
    copied = pickle.loads(pickle.dumps(mth))
    assert copied.floor(copied.e) == 2


def test_task():
    @otf.function
    def add(x=0, y=0):
        return x + y

    assert pack.base.shallow_reduce(runtime.Task.make(add)) == (
        (),
        {"function": add},
    )
    assert pack.base.shallow_reduce(runtime.Task.make(add, 5, y=6)) == (
        (),
        {
            "function": add,
            "args": [5],
            "kwargs": {"y": 6},
        },
    )

    t2 = pack.copy(runtime.Task.make(add, 5, y=6))

    assert t2.run() == 11
