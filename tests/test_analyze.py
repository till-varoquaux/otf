import ast
import inspect

import pytest

from otf import analyze, parser


class Instance:
    def __init__(self, ty):
        self._ty = ty

    def __eq__(self, other):
        return isinstance(other, self._ty)


ANY_NAME = Instance(ast.Name)
ANY_PARAMETER = Instance(inspect.Parameter)


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

    ff = parser.Function.from_function(f)
    assert analyze.visit_function(ff) == analyze.AstInfos(
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
            "print": ANY_NAME,
        },
    )

    assert analyze.visit_node(
        ff.statements[1], ff.filename
    ) == analyze.AstInfos(bound_vars={"foo": ANY_NAME})


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

    ff = parser.Function.from_function(f)
    with pytest.raises(SyntaxError):
        analyze.visit_function(ff)
