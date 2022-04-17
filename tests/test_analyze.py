from __future__ import annotations

import ast
import functools
import inspect

import pytest

from otf import analyze, parser

from . import utils


class Instance:
    def __init__(self, ty):
        self._ty = ty

    def __eq__(self, other):
        return isinstance(other, self._ty)


ANY_NAME = Instance(ast.Name)
ANY_PARAMETER = Instance(inspect.Parameter)


@functools.cache
def parse(f):
    return parser.Function.from_function(f)


def visit(fn):
    return analyze.visit_function(parse(fn))


def visit_node(fn, *path):
    ff = parse(fn)
    tgt = utils.drill(ff.statements, path)
    if isinstance(tgt, list):
        return analyze.visit_block(tgt, filename=ff.filename)
    return analyze.visit_node(tgt, filename=ff.filename)


def test_visit():
    async def f(**foo):
        global c
        foo = 5  # noqa: F841
        b.i = c  # noqa: F821
        var: int
        var2: int = 5 * x  # noqa
        await c
        c += 1
        print(foo)

    assert visit(f) == analyze.AstInfos(
        async_ctrl=Instance(ast.Await),
        bound_vars={
            "foo": ANY_PARAMETER,
            "var": ANY_NAME,
            "var2": ANY_NAME,
        },
        free_vars={
            "c": Instance(ast.Global),
            "b": ANY_NAME,
            "x": ANY_NAME,
        },
    )

    visit_node(f, 1) == analyze.AstInfos(bound_vars={"foo": ANY_NAME})


def test_reachable():
    def f(x):
        if x > 5:
            raise ValueError("Too big")
        else:
            return
        return 5

    with pytest.raises(SyntaxError, match="Unreachable code"):
        visit(f)

    def f(x):
        if x > 5:
            return True
        return False

    visit(f)

    def f(x):
        while True:
            x = 5
            if x:
                break
            else:
                continue

    assert visit(f).exits is False
    assert visit_node(f, 0).exits is False
    assert visit_node(f, 0, "body").exits is True


def test_internal_variable():
    def f():
        return _otf_var  # noqa: F821

    with pytest.raises(SyntaxError, match="reserved for the otf runtime"):
        visit(f)


def test_assign_builtins():
    def f():
        print = 1  # noqa: F841

    with pytest.raises(SyntaxError, match="Modifying builtins"):
        visit(f)


SYNTAX_ERROR = """\
def f():
    x = c
    global c
"""


def test_visit2():
    with pytest.raises(SyntaxError, match="used prior to global declaration"):
        exec(SYNTAX_ERROR)


def test_visit_error():
    def f():
        class K:
            pass

    with pytest.raises(SyntaxError, match="'class' not supported"):
        visit(f)
