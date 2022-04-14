import ast
import linecache

import pytest

from otf import ast_utils

from . import utils


@pytest.mark.parametrize(
    ("v"),
    (
        1,
        None,
        True,
        5.0,
        b"\000 yes",
        -5.0,
        -0.0,
    ),
)
def test_const(v):
    exp = ast.parse(repr(v), mode="eval").body
    utils.assert_eq_ast(ast_utils.constant(v), exp)


def test_dotted_path():
    exp = ast.parse("a.c", mode="eval").body
    utils.assert_eq_ast(ast_utils.dotted_path("a.c"), exp)

    exp = ast.parse("a", mode="eval").body
    utils.assert_eq_ast(ast_utils.dotted_path("a"), exp)

    [exp] = ast.parse("a.c=5").body[0].targets
    utils.assert_eq_ast(ast_utils.dotted_path("a.c", ctx=ast.Store()), exp)


def test_call():
    exp_args = ast.parse("5 + 5", mode="eval").body
    exp = ast.parse("a.c(5, 5 + 5)", mode="eval").body
    utils.assert_eq_ast(ast_utils.call("a.c", 5, exp_args), exp)


def test_fill_linecache():
    content = "1\n2\n3\n4\n"
    filename = ast_utils.fill_linecache(content)
    assert linecache.getline(filename, 1) == "1\n"
    assert linecache.getline(filename, 3) == "3\n"
    linecache.checkcache()  # Check that we aren't purged by checkcache
    assert linecache.getline(filename, 3) == "3\n"
    # content is hashconsed:
    assert ast_utils.fill_linecache(content) == filename
