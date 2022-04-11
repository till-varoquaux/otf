import inspect
import math

import pytest

from otf import utils


class C:
    @staticmethod
    def fn():
        pass


def _rebound_fn():
    pass


REBOUND_FN = _rebound_fn


def _rebound_fn():
    pass


def unbound():
    pass


UNBOUND_FN = unbound

del unbound


def test_get_name():
    def f():
        pass

    with pytest.raises(ValueError, match="defined inside of functions"):
        utils.get_locate_name(f)

    with pytest.raises(TypeError, match="lambdas"):
        utils.get_locate_name(lambda: 5)

    with pytest.raises(TypeError, match="Type .* not supported"):
        utils.get_locate_name(None)

    with pytest.raises(ValueError, match="it's overridden"):
        utils.get_locate_name(REBOUND_FN)

    with pytest.raises(ValueError, match="cannot be reloaded"):
        utils.get_locate_name(UNBOUND_FN)

    assert utils.get_locate_name(utils) == "otf.utils"

    # We work fine with primitives that aren't defined as functions in the
    # python type system.
    assert inspect.isbuiltin(math.floor)
    assert not inspect.isfunction(math.floor)
    assert utils.get_locate_name(math.floor) == "math.floor"

    # builtin names get compressed
    assert utils.get_locate_name(id) == "id"
    assert utils.locate("id") == id
