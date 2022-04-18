from __future__ import annotations

import math
import pickle

import pytest

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


def test_serialize():
    mth = runtime.NamedReference(math)
    assert pack.explode(mth) == pack.Custom("otf.NamedReference", "math")
    assert mth.floor(mth.e) == 2
    copied = pickle.loads(pickle.dumps(mth))
    assert copied.floor(copied.e) == 2