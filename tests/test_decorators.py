from __future__ import annotations

import pytest

from otf import decorators


def test_dec():
    @decorators.function
    def f(x, y):
        return x + y

    assert f(5, 6) == 11


def test_dec2():
    @decorators.function
    @decorators.environment(i=0)
    def incr():
        global i
        i += 1
        return i

    assert incr() == 1
    assert incr() == 2


def test_env_error():
    with pytest.raises(TypeError):

        @decorators.environment(i=0)
        class k:
            pass


def test_strict():

    with pytest.raises(SyntaxError, match="'yield' not supported"):

        @decorators.function
        def f():
            yield 5

    @decorators.function(strict=False)
    def f2():
        yield 5


def test_unbound():

    with pytest.raises(
        SyntaxError, match="variable 'i' not found in the environment"
    ):

        @decorators.function
        def f():
            return i

    @decorators.function
    @decorators.environment(i=0)
    def f2():
        return i

    @decorators.function(strict=False)
    def f3():
        return i

    # It's not an error to use builtins
    @decorators.function
    def f4(args):
        return sorted(args)
