from __future__ import annotations

import ast
import copyreg
import io
import linecache
import math
import pickle
import pickletools
import sys

import pytest

from otf import ast_utils, decorators, pack, pretty

from . import utils

LOREM_IPSUM = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo
consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum
dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident,
sunt in culpa qui officia deserunt mollit anim id est laborum.
"""

QUICK_BROWN_FOX = "The quick brown fox jumps over the lazy dog"


@pytest.fixture(autouse=True)
def _auto_clean(monkeypatch):
    old_dispatch_table = copyreg.dispatch_table.copy()
    monkeypatch.setattr(
        pack.base, "DISPATCH_TABLE", pack.base.DISPATCH_TABLE.copy()
    )
    yield
    copyreg.dispatch_table.clear()
    copyreg.dispatch_table.update(old_dispatch_table)


def dis(x):
    "Helper function that dumps out how pickle would serialize a given value."
    s = pickletools.optimize(pickle.dumps(x))
    buf = io.StringIO()
    pickletools.dis(s, buf)
    raise AssertionError(buf.getvalue())


def unedit(s: str):
    *prelude, last = ast.parse(s).body
    assert isinstance(last, ast.Expr)
    env = {}
    filename = ast_utils.fill_linecache(s)
    try:
        before = compile(
            ast.Module(prelude, type_ignores=[]), filename=filename, mode="exec"
        )
        main = compile(
            ast.Expression(last.value), filename=filename, mode="eval"
        )
        exec(before, env)
        return eval(main, env)
    except BaseException:
        raise
    else:
        del linecache.cache[filename]


def explode_exec(text):
    return pack.text.reduce(text, pack.tree.NodeBuilder())


def roundtrip(v):
    exploded = pack.tree.explode(v)
    v2 = pack.tree.implode(exploded)
    indented = pack.dumps(v, indent=4)
    executable = pack.dumps(v, format=pack.EXECUTABLE)

    if exploded == exploded:
        assert pack.text.reduce(indented, pack.tree.NodeBuilder()) == exploded
        assert explode_exec(executable) == exploded

    flat = pack.dumps(v)
    flat2 = pack.tree.reduce(exploded, pack.text.Simple(4))

    assert flat == pretty.single_line(flat2)
    utils.assert_eq_ast(flat, indented)
    v3 = pack.loads(flat)
    v4 = unedit(executable)
    # nan can wreak havoc in comparisons so we compare pickles...
    assert (
        pickle.dumps(v)
        == pickle.dumps(v2)
        == pickle.dumps(v3)
        == pickle.dumps(v4)
    )
    return v2


class A:
    pass


class MyInt(int):
    pass


@decorators.function
def ackerman(m, n):
    if m == 0:
        return n + 1
    if n == 0:
        return ackerman(m - 1, 1)
    return ackerman(m - 1, ackerman(m, n - 1))


PRETTY_ACK = """\
otf.closure(
    {
        'environment': {
            'ackerman': otf.Function(
                {
                    'name': 'ackerman',
                    'signature': ['m', 'n'],
                    'body': (
                        '    if m == 0:\\n'
                        '        return n + 1\\n'
                        '    if n == 0:\\n'
                        '        return ackerman(m - 1, 1)\\n'
                        '    return ackerman(m - 1, ackerman(m, n - 1))'
                    )
                }
            )
        },
        'target': ref(11)
    }
)\
"""


def test_dumps():
    assert (
        pack.dumps((4, -5.0, float("nan"), float("inf"), -float("inf")))
        == "tuple([4, -5.0, nan, inf, -inf])"
    )
    tuples = [(x, x + 1) for x in range(5)]
    succ = {x: y for x, y in zip(tuples[:-1], tuples[1:])}
    # assert succ == {}
    assert pack.dumps(succ) == (
        "{tuple([0, 1]): tuple([1, 2]), ref(4): tuple([2, 3]), "
        "ref(4): tuple([3, 4]), ref(4): tuple([4, 5])}"
    )
    assert pack.dumps(ackerman, indent=4) == PRETTY_ACK


def test_get_import(monkeypatch):
    assert pack.text._get_import("otf.Function") == "otf"
    assert pack.text._get_import("otf.compiler.Function") == "otf.compiler"

    assert pack.text._get_import("math") == "math"
    assert pack.text._get_import("float") is None
    assert pack.text._get_import("otf.Function") == "otf"

    assert isinstance(pack.text._get_import("I.DoNot.Exist"), ImportError)


def bad_locate(_):
    raise RuntimeError("locate failed")


def test_get_import_pb(monkeypatch):

    monkeypatch.setattr("otf.utils.locate", bad_locate)

    assert isinstance(pack.text._get_import("what.ever"), RuntimeError)


# It's important that we reload those with the exact same exploded structure to
# make sure all the transformations between format do not change the size of a
# tree.
def test_load_exec_float():
    assert explode_exec("float(5)") == pack.tree.Custom(
        constructor="float", value=5
    )
    for pos_inf in ("float('inf')", "+float('inf')", "inf", "+inf"):
        assert explode_exec(pos_inf) == float("inf")
    for neg_inf in ("-float('inf')", "-float('inf')", "-inf"):
        assert explode_exec(neg_inf) == -float("inf")
    for nan in ("nan", "float('nan')"):
        assert math.isnan(explode_exec(nan))


def test_dump_exec():
    assert pack.dumps([], format=pack.EXECUTABLE) == "[]\n"
    assert pack.dumps(
        (math.nan, math.inf, -math.inf), format=pack.EXECUTABLE
    ) == ('tuple([float("nan"), float("inf"), -float("inf")])\n')
    x = {}
    v = (x, x)
    assert (
        pack.dumps([v, v, v, v], format=pack.EXECUTABLE)
        == "_0 = {}\n\n_1 = tuple([_0, _0])\n\n[_1, _1, _1, _1]\n"
    )


BAD_IMPORT_EXPECTED = """\
# There were errors trying to import the following constructors
#
# + 'tuple': RuntimeError locate failed

tuple([1, 2, 3])
"""


def _dump_exec_no_imports(obj):
    doc = pack.base.reduce(
        obj, pack.text.ModulePrinter(indent=4, add_imports=False)
    )
    return doc.to_string(80)


def test_dump_exec_bad_import(monkeypatch):
    def bad_get_import(s):
        return RuntimeError("locate failed")

    monkeypatch.setattr(pack.text, "_get_import", bad_get_import)
    assert pack.dumps((1, 2, 3), format=pack.EXECUTABLE) == BAD_IMPORT_EXPECTED

    assert _dump_exec_no_imports((1, 2, 3)) == "tuple([1, 2, 3])\n"


EDITABLE_ACKERMAN = r"""import otf

_0 = otf.Function(
    {
        'name': 'ackerman',
        'signature': ['m', 'n'],
        'body': (
            '    if m == 0:\n'
            '        return n + 1\n'
            '    if n == 0:\n'
            '        return ackerman(m - 1, 1)\n'
            '    return ackerman(m - 1, ackerman(m, n - 1))'
        )
    }
)

otf.closure({'environment': {'ackerman': _0}, 'target': _0})
"""


def test_dump_exec_acckerman():
    assert pack.dumps(ackerman, format=pack.EXECUTABLE) == EDITABLE_ACKERMAN


def test_loads_prelude():
    with pytest.raises(ValueError, match="Empty document"):
        pack.loads("# Comments don't show up in the ast\n\n")
    with pytest.raises(ValueError, match="The last node"):
        pack.loads("v=5")
    with pytest.raises(ValueError, match="import ... as"):
        pack.loads("import c, a as b; 5")
    with pytest.raises(ValueError, match="`from ... import`"):
        pack.loads("from a import b; 5")
    with pytest.raises(ValueError, match="Expression"):
        pack.loads("5; 5")
    with pytest.raises(ValueError, match="Assigning to multiple targets"):
        pack.loads("a = b = 5; 5")
    with pytest.raises(
        ValueError, match="Only assigning to top level variables is supported"
    ):
        pack.loads("a, b = 5; 5")
    with pytest.raises(ValueError, match="reserved keyword"):
        pack.loads("nan = 5; 5")

    with pytest.raises(ValueError, match="Cannot rebind a variable"):
        pack.loads("x = 6; x = 5; 5")

    with pytest.raises(
        ValueError, match="Cannot import after declaring variables"
    ):
        pack.loads("x = 6; import math; 5")

    with pytest.raises(
        ValueError, match="Cannot redefine a name already used by an import"
    ):
        pack.loads("import math; math = 5; 5")

    with pytest.raises(ValueError, match="Only bindings"):
        pack.loads("assert False; 5")

    assert pack.loads("import i.donot.exist; 5") == 5
    assert pack.loads("import i.donot.exist, a, a.b, c; 5") == 5
    assert pack.loads("import i.donot.exist; 5") == 5
    assert pack.loads("a: int = 5; 5") == 5


def test_loads_exec_refs():

    with pytest.raises(ValueError, match="Unbound variable"):
        pack.loads("[a, a]")
    with pytest.raises(ValueError, match="Unbound variable"):
        pack.loads("import a.b.c; [a, a]")
    with pytest.raises(ValueError, match="reference an imported module"):
        pack.loads("import a; [a, a]")

    assert pack.loads("a = 5; [a, a]") == [5, 5]
    assert explode_exec("a = [1, 2]; b = [a, a]; [b, a]") == [
        [[1, 2], pack.tree.Reference(offset=3)],
        pack.tree.Reference(offset=1),
    ]

    with pytest.raises(ValueError, match="circular reference"):
        pack.loads("a = [a, a]; a") == ""

    with pytest.raises(ValueError, match="isn't defined yet"):
        pack.loads("a = [b, b]; b = 5; a")

    assert pack.loads("b = 5; a = [b, b];  a") == [5, 5]


def double(v):
    return 2 * v


RANGE = """
# We support comments in serialized data
[
  1,
  2,
  # Pluses
  +3,
  4,
  # and Trailing commans
  5,
]
"""


def test_loads():
    assert pack.loads("5") == pack.loads("+5") == 5
    assert pack.loads("-5") == -5
    assert math.isnan(pack.loads("nan"))
    assert pack.loads("inf") == pack.loads("+inf") == math.inf
    assert pack.loads("-inf") == -math.inf
    assert pack.loads("[None, True]") == [None, True]
    assert pack.loads("{1: None, 2: True}") == {1: None, 2: True}
    assert pack.loads("{1: [], 2: ref(2)}") == {1: [], 2: []}
    assert pack.loads("{1: [], 2: ref(2)}") == {1: [], 2: []}

    assert pack.loads("tests.test_pack.double(5)") == 10

    assert pack.loads(RANGE) == [1, 2, 3, 4, 5]

    with pytest.raises(ValueError):
        pack.loads("[1, 2, 3, \n5+5\n]")

    with pytest.raises(ValueError):
        pack.loads("a['a'](5)")


def test_simple():
    with pytest.raises(TypeError):
        pack.tree.explode(A())
    with pytest.raises(TypeError):
        pack.tree.implode(A())
    assert pack.tree.explode(5) == 5
    assert (
        pack.tree.explode([5, None, "a"])
        == [5, None, "a"]
        == pack.tree.implode([5, None, "a"])
        == pack.copy([5, None, "a"])
    )
    assert (
        pack.tree.explode({5: None})
        == {5: None}
        == pack.tree.implode({5: None})
    )
    with pytest.raises(TypeError):
        pack.tree.explode(MyInt(5))


def test_shared():
    v = [1, 2]
    exploded = [[1, 2], pack.tree.Reference(3), pack.tree.Reference(1)]
    assert pack.tree.explode([v, v, v]) == exploded
    v2 = roundtrip([v, v, v])
    assert v2[0] is v2[1] is v2[2]


def test_mapping():
    k1 = (1, 2)
    k2 = (2, 3)
    v = {k1: k2, k2: None}
    roundtrip(v)


@pytest.mark.parametrize(
    "x",
    (
        math.nan,
        math.inf,
        -math.inf,
        -0.0,
        0.0,
        1 / 3,
        sys.float_info.min,
        sys.float_info.max,
    ),
)
def test_weird_floats(x):
    roundtrip(x)
    roundtrip({x: x})


def test_hash_cons_str():
    # copy.copy will actually do nothing. We want to get a new copy of the same
    # string.
    lorem_copy = "\n".join(LOREM_IPSUM.split("\n"))
    assert lorem_copy == LOREM_IPSUM
    assert id(lorem_copy) != id(LOREM_IPSUM)
    assert pack.tree.explode(
        [lorem_copy, LOREM_IPSUM, lorem_copy, QUICK_BROWN_FOX]
    ) == [
        LOREM_IPSUM,
        pack.tree.Reference(1),
        pack.tree.Reference(1),
        QUICK_BROWN_FOX,
    ]


def test_reccursive():
    v = []
    v.append(v)
    with pytest.raises(ValueError):
        pack.tree.explode(v)


def test_tuple():
    v = 1, 2
    assert pack.tree.explode(v) == pack.tree.Custom("tuple", [1, 2])
    assert pack.base.shallow_reduce(v) == [1, 2]
    assert pack.tree.implode(pack.tree.Custom("tuple", [1, 2])) == (1, 2)


def test_infer_reducer_type():
    def f(x: list[int]):
        pass

    assert pack.base._infer_reducer_type(f) is list

    def f(x, y):
        pass

    with pytest.raises(ValueError):
        assert pack.base._infer_reducer_type(f)


class ItemGetter:
    # A simple python class that can't be pickled by default
    def __init__(self, item):
        self._item = item

        def func(obj):
            return obj[item]

        self._call = func

    def __call__(self, obj):
        return self._call(obj)


class ItemGetter2(ItemGetter):
    pass


def test_register_pickle():
    v = ItemGetter(2)
    with pytest.raises(Exception, match="Can't pickle"):
        pickle.dumps(v)

    @pack.register(pickle=True)
    def reduce_item_getter(i: ItemGetter):
        return ItemGetter, i._item

    v2 = pickle.loads(pickle.dumps(v))

    assert v2([0, 1, 2, 3, 4]) == 2

    # Inheritance doesn't work
    with pytest.raises(Exception, match="Can't pickle"):
        pickle.dumps(ItemGetter2(2))


class IdGetter:
    def __init__(self):
        self.value = id(self)


def id_getter(value: int):
    v = IdGetter.__new__(IdGetter)
    v.value = value
    return v


def test_register_random_holder():

    rg = IdGetter()

    @pack.register(pickle=True)
    def reduce_random_getter(i: IdGetter):
        return id_getter, i.value

    reconstructed = pickle.loads(pickle.dumps(rg))
    reconstructed2 = pack.copy(rg)
    assert reconstructed.value == reconstructed2.value == rg.value
