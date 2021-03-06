from __future__ import annotations

import ast
import inspect
import linecache
import pickle
import textwrap

import pytest

from otf import pack, parser

from . import utils


def test_get_body_pb(monkeypatch):
    class Kls:
        pass

    with pytest.raises(TypeError, match="Argument is not a function"):
        parser.Function.from_function(Kls)

    # If, for some reason, we fail to parse, we get an intelligible error.
    tgt = lambda _: 5  # noqa: E731
    tgt.__name__ = "fake_f"
    with monkeypatch.context() as m, pytest.raises(
        ValueError, match="Could not find function definition for"
    ):
        m.setattr("inspect.isfunction", lambda _: True)
        parser.Function.from_function(tgt)

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

    utils.assert_eq_ast(
        ff.statements,
        textwrap.dedent(ff.body),
        ast.fix_missing_locations(ast.Return(ast.Name("a", ctx=ast.Load()))),
        "return a",
    )


def test_get_body2():
    def empty():
        pass

    ff = parser.Function.from_function(empty)

    utils.assert_eq_ast(
        ff.statements,
        textwrap.dedent(ff.body),
        ast.fix_missing_locations(ast.Pass()),
        "pass",
    )


def test_get_body_async():
    async def f(g):
        a = await g()
        return await a

    ff = parser.Function.from_function(f)

    utils.assert_eq_ast(
        ff.statements, textwrap.dedent(ff.body), "a = await g()\nreturn await a"
    )


# https://github.com/till-varoquaux/otf/issues/2
def test_get_body_redef():
    def f():
        "original"

    orig = f

    def f():  # noqa: F811
        "redefined"

    ff = parser.Function.from_function(f)
    assert ast.literal_eval(ff.body.strip()) == "redefined"

    ff = parser.Function.from_function(orig)
    assert ast.literal_eval(ff.body.strip()) == "original"


def get_body(f):
    return parser.Function.from_function(f).body


def test_body_same_line():

    # fmt: off

    # Multi statements, one line
    def f1(): x = 5; return x  # noqa: E702

    # Weird break
    def f2(
    ): return 5

    def f3(
    ): x = (  # noqa: E702
    ); return x  # noqa: E702

    # fmt: on

    assert get_body(f1).strip() == "...: x = 5; return x"
    assert get_body(f2).strip() == "...: return 5"
    assert get_body(f3).strip() == "...: x = (  # noqa: E702\n    ); return x"


def test_body_comments():
    def f():
        # Before
        x = 5
        # In between
        return x  # After

    assert textwrap.dedent(get_body(f)).split("\n") == [
        "x = 5",
        "# In between",
        "return x",
    ]


def test_keeps_inner_indent():
    def f():
        return """a
        b"""

    ff = parser.Function.from_function(f)

    # A naive parser would dedent the whole body of the function and mess up
    # the indent.
    utils.assert_eq_ast(ff.statements, r"return 'a\n        b'")


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
        (
            "x, y = 4",
            {
                "args": ["x", "y"],
                "defaults": [4],
            },
        ),
        (
            "x, *, y = 4",
            {
                "args": ["x", "*", "y"],
                "kwdefaults": {"y": 4},
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


def explode(f):
    fn = parser.Function.from_function(f)
    return pack.tree.explode(fn)


def test_explode_function():
    # fmt: off

    def f(a, *, b=5):
        return a + b

    # fmt: on

    exploded = pack.tree.Custom(
        "otf.parser._implode_function",
        name="f",
        signature={"args": ["a", "*", "b"], "kwdefaults": {"b": 5}},
        body="        return a + b",
    )
    assert explode(f) == exploded
    reconstituted = pack.tree.implode(exploded)
    assert linecache.getlines(reconstituted.filename) == [
        "def f(a, *, b=...):\n",
        "        return a + b\n",
    ]


def test_explode_function2():
    # fmt: off

    def f(a, *, b=5): return a + b

    # fmt: on

    exploded = pack.tree.Custom(
        "otf.parser._implode_function",
        name="f",
        signature={"args": ["a", "*", "b"], "kwdefaults": {"b": 5}},
        body="                 ...: return a + b",
    )
    assert explode(f) == exploded
    reconstituted = pack.tree.implode(exploded)
    assert linecache.getlines(reconstituted.filename) == [
        "def f(a, *, b=...): return a + b\n"
    ]


def test_pickle():
    def f(a, *, b=5):
        return a + b

    ff = parser.Function.from_function(f)

    s = pickle.dumps(ff)

    reloaded = pickle.loads(s)

    # We check that we didn't use the python pickle mechanism
    assert reloaded.filename != ff.filename
    assert reloaded.body == ff.body
