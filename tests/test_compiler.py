from __future__ import annotations

import ast
import inspect
import pickle
import textwrap
from unittest import mock

import pytest

from otf import compiler, pack, parser

from . import utils


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

    utils.assert_eq_ast(expected, got)
    assert ast.unparse(got) == str(sig)[1:-1]


def test_mk_function_def():
    def f():
        return x + y  # noqa: F821

    fn = parser.Function.from_function(f)
    node = compiler._mk_function_def(fn)
    utils.assert_eq_ast(node, "def f():\n    return x + y")


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


def test_pickle_and_copy():
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
    assert incr.environment["i"] == 1
    assert copy_incr.environment["i"] == 0

    assert copy_incr() == 1

    copy_incr2 = pack.copy(incr)
    assert incr() == 2
    assert copy_incr2() == 2


def test_explode_function():
    @compiler.Environment().function
    def add(x, y):
        return x + y

    assert pack.explode(add) == pack.Custom(
        "otf.Closure",
        {
            "environment": {
                "add": pack.Custom(
                    "otf.Function",
                    value={
                        "name": "add",
                        "signature": ["x", "y"],
                        "body": "        return x + y",
                    },
                )
            },
            "target": utils.InstanceOf(pack.Reference),
        },
    )


_NODE = ast.Pass(lineno=0, col_offset=1, end_lineno=0, end_col_offset=1)


def test_template():
    Tmpl = compiler.Template(
        "if __var_srep in _otf_variables: "
        " __var_dest = _otf_variables[__var_srep]"
    )

    Expected = "if 'a' in _otf_variables: a = _otf_variables['a']"

    srep = ast.Constant(kind=None, value="a")
    dest = ast.Name(id="a", ctx=ast.Store())
    utils.assert_eq_ast(Tmpl(_NODE, srep=srep, dest=dest), Expected)


def test_pos_fill():
    Tmpl = compiler.Template("return __var_x")
    [stmt] = Tmpl(_NODE, x=ast.Constant(value=5, lineno=5, col_offset=3))
    assert stmt.lineno == 0
    assert stmt.value.lineno == 5


LOOP_TEST_INPUT = r"""
if x:
  break
for y in x:
  if y:
    break
else:
  continue
"""

LOOP_TEST_OUTPUT = r"""
if x:
    _otf_pos = 1
    continue
for y in x:
    if y:
        break
else:
    _otf_pos = 2
    continue
"""


def test_loop_rewriter():
    stmts = ast.parse(LOOP_TEST_INPUT).body
    rewritter = compiler.LoopCtrlRewriter(
        "<input>", mock.Mock(idx=1), mock.Mock(idx=2)
    )
    rewritten = rewritter.visit_block(stmts)
    utils.assert_eq_ast(rewritten, LOOP_TEST_OUTPUT)


COMPILED = r"""
def wf(_otf_variables, _otf_pos, _otf_val):
    if 'a' in _otf_variables:
        a = _otf_variables['a']
    if 'a_' in _otf_variables:
        a_ = _otf_variables['a_']
    if 'b' in _otf_variables:
        b = _otf_variables['b']
    if 'b_' in _otf_variables:
        b_ = _otf_variables['b_']
    if 'x' in _otf_variables:
        x = _otf_variables['x']
    if 'y' in _otf_variables:
        y = _otf_variables['y']
    while True:
        match _otf_pos:
            case 0:
                (a_, b_) = (f1(x), f2(y))
                return _otf_suspend(position=1, variables=locals(), awaiting=b_)
            case 1:
                b = _otf_val
                return _otf_suspend(position=2, variables=locals(), awaiting=a_)
            case 2:
                a = _otf_val
                return _otf_suspend(
                  position=3,
                  variables=locals(),
                  awaiting=sleep(5)
                )
            case 3:
                return a + b
"""


def test_compile_workflow():
    async def wf(x, y):
        a_, b_ = f1(x), f2(y)  # noqa: F821
        b = await b_
        a: int = await a_
        await sleep(5)  # noqa: F821
        return a + b

    e = compiler.Environment(
        f1=lambda x: x,
        f2=lambda y: y,
    )

    comp = compiler.compile_worflow(
        parser.Function.from_function(wf), environment=e
    )
    utils.assert_eq_ast([compiler._mk_function_def(comp)], COMPILED)


COMPILED_IF = """
def wf(_otf_variables, _otf_pos, _otf_val):
    if 'x' in _otf_variables:
        x = _otf_variables['x']
    if 'y' in _otf_variables:
        y = _otf_variables['y']
    while True:
        match _otf_pos:
            case 0:
                if x is not None:
                    return _otf_suspend(
                        position=1,
                        variables=locals(),
                        awaiting=y(x),
                    )
                else:
                    _otf_pos = -1
                    continue
            case 1:
                x = _otf_val
                _otf_pos = -1
                continue
            case -1:
                return x
"""


def test_compile_workflow_if():
    async def wf(x, y):
        if x is not None:
            x = await y(x)
        return x

    e = compiler.Environment(
        f1=lambda x: x,
        f2=lambda y: y,
    )

    comp = compiler.compile_worflow(
        parser.Function.from_function(wf), environment=e
    )
    utils.assert_eq_ast([compiler._mk_function_def(comp)], COMPILED_IF)


EXPECTED_IMPLICIT_RETURN = """
def wf(_otf_variables, _otf_pos, _otf_val):
    if 'x' in _otf_variables:
        x = _otf_variables['x']
    if 'y' in _otf_variables:
        y = _otf_variables['y']
    while True:
        match _otf_pos:
            case 0:
                return _otf_suspend(
                    position=1,
                    variables=locals(),
                    awaiting=x(y)
                )
            case 1:
                return
"""


def test_compile_workflow_implicit_return():
    async def wf(x, y):
        await x(y)

    e = compiler.Environment(
        f1=lambda x: x,
        f2=lambda y: y,
    )

    comp = compiler.compile_worflow(
        parser.Function.from_function(wf), environment=e
    )
    utils.assert_eq_ast(
        [compiler._mk_function_def(comp)], EXPECTED_IMPLICIT_RETURN
    )


EXPECTED_WHILE = """
while True:
    match _otf_pos:
        case 0:
            futs = [fn(x) for x in lst]
            _otf_pos = -1
            continue
        case -1:
            if futs:
                return _otf_suspend(
                  position=1,
                  variables=locals(),
                  awaiting=futs.pop()
                )
            else:
                _otf_pos = -2
                continue
        case -2:
            return
        case 1:
            v = _otf_val
            if v:
                _otf_pos = -2
                continue
            _otf_pos = -1
            continue
"""


def test_compile_workflow_while():
    async def wf(fn, lst):
        futs = [fn(x) for x in lst]
        while futs:
            v = await futs.pop()
            if v:
                break
        return

    e = compiler.Environment()

    comp = compiler.compile_worflow(
        parser.Function.from_function(wf), environment=e
    )
    utils.assert_eq_ast(
        [compiler._mk_function_def(comp).body[-1]], EXPECTED_WHILE
    )


def test_compile_workflow_bad_while():
    async def wf(needle, haystack):
        index = 0

        while index < len(haystack):
            item = await future(haystack[index])
            if item == needle:
                return index
            index += 1
        else:
            return index - 1

    e = compiler.Environment()

    with pytest.raises(SyntaxError, match="While.* not supported"):
        compiler.compile_worflow(
            parser.Function.from_function(wf), environment=e
        )


def test_workflow_bad_await():
    async def wf(y):
        if await y:
            return 5

    e = compiler.Environment()

    with pytest.raises(SyntaxError):
        e.workflow(wf)

    async def wf(y):
        for i in y:
            await i

    with pytest.raises(SyntaxError):
        e.workflow(wf)


# Dummy scheduler to make testing easier
class future:
    def __init__(self, value):
        self.value = value


def run(wf, *args, **kwargs):
    if not isinstance(wf, compiler.Workflow):
        e = compiler.Environment(future=future)
        wf = e.workflow(wf)
    trace = [wf(*args, **kwargs)]
    while isinstance(trace[-1], compiler.Suspension):
        copied = pickle.loads(pickle.dumps(trace[-1]))
        trace.append(copied.resume(copied.awaiting.value))
    return trace


def test_simple_workflow():
    async def wf(x, y):
        a_, b_ = future(x * 2), future(y * 3)  # noqa: F821
        b = await b_
        a = await a_
        return a + b

    trace = run(wf, x=5, y=6)
    assert trace[-1] == 28


def test_remove_es():
    async def wf(inpt):
        [*chars] = inpt
        res = ""
        while chars:
            c = await future(chars.pop(0))
            if c == "e":
                continue
            res += c
        return res

    trace = run(wf, "entrepreneur")
    assert trace[-1] == "ntrprnur"


def test_find_squares():
    # A very stupid to find all the squares in [0...limit]

    # This test exercises loop control flow a fair amount
    # (nested loops, continue, break...)
    async def wf(limit):
        i = 0
        res = []
        while i < limit:
            j = 0
            while True:
                j_squared = await future(j * j)
                if j_squared > i:
                    break
                j += 1
                if j_squared < i:
                    continue
                res.append(i)
            i += 1
        return res

    trace = run(wf, 20)
    assert trace[-1] == [0, 1, 4, 9, 16]
