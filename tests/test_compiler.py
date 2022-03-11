import ast
import inspect
import pickle
import textwrap

import pytest

from otf import compiler, parser

POS_FIELDS = ("lineno", "col_offset", "end_lineno", "end_col_offset")


def explode_ast(node, strip_pos: bool = False):
    """Turns ast nodes in a format that can easily be compared and introspected.

    This is useful because the mapping between python's syntax and python's AST
    is not always straightforward.

    """
    ignore_fields = POS_FIELDS if strip_pos else ()
    if isinstance(node, (str, int, type(None))):
        return node
    if isinstance(node, list):
        return [explode_ast(v, strip_pos) for v in node]
    assert isinstance(node, ast.AST), node
    return {
        k: explode_ast(v, strip_pos=strip_pos)
        for k, v in ast.iter_fields(node)
        if k not in ignore_fields
    } | {"__type__": type(node).__name__}


@pytest.mark.parametrize(
    ("sig_str",), (("x, * , y",), ("x, /, y",), ("x, y, *args, **kwargs",))
)
def test_mk_arguments(sig_str):
    program = f"def fn({sig_str}): pass"
    g = {}
    exec(program, g)
    fn = g["fn"]
    sig = inspect.signature(fn)

    got = compiler._mk_arguments(sig)
    expected = ast.parse(program).body[0].args

    assert ast.unparse(expected) == ast.unparse(got) == str(sig)[1:-1]
    assert explode_ast(expected) == explode_ast(got)


def test_mk_function_def():
    def f():
        return x + y  # noqa: F821

    fn = parser.Function.from_function(f)
    node = compiler._mk_function_def(fn)
    assert ast.unparse(node) == "def f():\n    return x + y"


def test_defaults():

    env = compiler.Environment(i=0)

    @env.function
    def f(x, y, z="z", *, kx="kx", ky):
        return (x, y, z, kx, ky)

    assert f("x", "y", ky="ky") == ("x", "y", "z", "kx", "ky")


def test_funcref_introspection():
    env = compiler.Environment(i=0)

    @env.function
    def mul(a, *, b=1):
        "multiply two numbers"
        return a * b

    deadbeef = 3735928559
    try:
        compiler.id = lambda _: deadbeef

        assert repr(mul) == "<OtfFunction::mul(a, *, b=1) at 0xdeadbeef>"
    finally:
        del compiler.id

    # Make sure update_wrapper did its job ok
    assert mul.__module__ == __name__
    assert str(inspect.signature(mul)) == "(a, *, b=1)"
    assert mul.__doc__ == "multiply two numbers"


def test_env_capture():

    env = compiler.Environment(i=0)

    @env.function
    def incr():
        global i
        i += 1
        return i

    assert incr() == 1
    assert incr() == 2

    assert env["i"] == 2

    @env.function
    def incr2():
        global i
        i += 2
        return i

    # Rebinding incr in the environment updates the pointer
    env["incr"] = env["incr2"]

    # incr still points at the old function
    assert incr() == 3
    assert textwrap.dedent(incr.origin.body) == (
        "global i\n" "i += 1\n" "return i"
    )

    # but the environment has the new definition
    assert textwrap.dedent(env["incr"]._origin.body) == (
        "global i\n" "i += 2\n" "return i"
    )

    # This also rebinds in the environment

    @env.function
    def incr():
        global i
        i += 1.0
        return i

    assert incr() == 4.0


def test_lazy_env():

    env = compiler.Environment(i=0)

    @env.function(lazy=True)
    def incr():
        global i
        i += 1
        return i

    assert env["incr"]._fun is None

    assert incr() == 1

    assert env["incr"]._fun is not None

    assert incr() == 2


def test_pickle():
    def f(x, y):
        return x + y

    with pytest.raises(AttributeError, match=r"Can't pickle local object"):
        pickle.dumps(f)

    env = compiler.Environment(i=0)

    @env.function
    def incr():
        global i
        i += 1
        return i

    copy_incr = pickle.loads(pickle.dumps(incr))
    assert incr() == 1

    # We now have a copy of the closure incr(). I.e.: it has its own environment
    copy_incr = pickle.loads(pickle.dumps(incr))
    assert incr() == 2
    assert copy_incr() == 2
