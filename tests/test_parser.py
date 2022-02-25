import ast
import inspect

import pytest

from otf import parser


def _to_stmts(x: str | ast.stmt) -> list[ast.stmt]:
    if isinstance(x, str):
        return ast.parse(x).body
    else:
        return [x]


def unparse(*elts: str | ast.stmt) -> str:
    return "\n".join(ast.unparse(stmt) for x in elts for stmt in _to_stmts(x))


def test_get_body_pb(monkeypatch):
    class Kls:
        pass

    with pytest.raises(TypeError, match="Argument is not a function"):
        parser.Function.from_function(Kls)

    # If, for some reason, we fail to parse, we get an intelligible error.
    with monkeypatch.context() as m, pytest.raises(
        ValueError, match="Could not find function definition for"
    ):
        m.setattr("inspect.isfunction", lambda _: True)
        parser.Function.from_function(Kls)

    with pytest.raises(TypeError, match="lambdas not supported"):
        parser.Function.from_function(lambda x: x * x)

    with pytest.raises(TypeError, match="Argument is not a function"):
        assert parser.Function.from_function(getattr)


def test_get_body():
    def id(f):
        return f

    # make sure we can handle decorators
    @id
    def f(a):
        return a

    ff = parser.Function.from_function(f)

    assert (
        unparse(*ff.statements)
        == unparse(*(line.lstrip() for line in ff.lines))
        == unparse(ast.Return(ast.Name("a")))
        == "return a"
    )


def test_get_body_async():
    async def f(g):
        a = await g()
        return await a

    ff = parser.Function.from_function(f)

    assert (
        unparse(*ff.statements)
        == unparse(*(line.lstrip() for line in ff.lines))
        == unparse("a = await g()", "return await a")
        == "a = await g()\nreturn await a"
    )


def test_keeps_inner_indent():
    def f():
        return """a
        b"""

    ff = parser.Function.from_function(f)

    # A naive parser would dedent the whole body of the function and mess up
    # the indent.
    assert unparse(*ff.statements) == r"return 'a\n        b'"


def test_get_lines_error1():
    g = {}
    exec("def f():\n   return 5", g)
    with pytest.raises(OSError, match="source code not available"):
        parser._get_lines(g["f"])


def test_get_line_error2(tmp_path):
    (tmp_path / "a").touch()
    g = {}
    exec(compile("def f():\n   return 5", str(tmp_path / "a"), "exec"), g)
    with pytest.raises(OSError, match="could not get source code"):
        parser._get_lines(g["f"])


def test_strip_annot():
    def f(x: int, /, y: float = 5.0, *, z=7, **rest) -> int:
        return 5

    def g(x, /, y=5.0, *, z=7, **rest):
        return 5

    assert parser._get_signature(f) == inspect.signature(g)


@pytest.mark.parametrize(
    ("asig", "exploded"),
    (
        ("a, b", ["a", "b"]),
        (
            "w, x = 4, /, y = 5., *, z=7, **rest",
            {
                "args": ["w", "x", "/", "y", "*", "z", "**rest"],
                "defaults": [4, 5.0],
                "kwdefaults": {"z": 7},
            },
        ),
        ("arg1, arg2, *args, opt_args", ["arg1", "arg2", "*args", "opt_args"]),
        ("arg1, arg2, /", ["arg1", "arg2", "/"]),
    ),
)
def test_explode_sig(asig, exploded):
    # We create a function that takes the arguments described in asig
    program = f"def fn({asig}): pass"
    g = {}
    exec(program, g)
    fn = g["fn"]
    sig = inspect.signature(fn)
    assert parser._explode_signature(sig) == exploded
    assert parser._implode_signature(exploded) == sig
