from __future__ import annotations

import ast


def _to_stmts(x):
    if isinstance(x, str):
        return ast.parse(x).body
    elif isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]


def unparse(*elts):
    return "\n".join(ast.unparse(stmt) for x in elts for stmt in _to_stmts(x))


Primitives = bool | bytes | str | int | float | None


def explode_ast(node):
    """Turns ast nodes in a format that can easily be compared and introspected.

    This is useful because the mapping between python's syntax and python's AST
    is not always straightforward.

    """
    if isinstance(node, Primitives):
        return node
    if isinstance(node, list | tuple):
        return [explode_ast(v) for v in node]
    assert isinstance(node, ast.AST), node
    return {k: explode_ast(v) for k, v in ast.iter_fields(node)} | {
        "__type__": type(node).__name__
    }


def _check_ast_eq(left, right, path):
    assert type(left) == type(right), f"At {path}"
    if isinstance(left, Primitives):
        assert left == right, f"At {path}"
    elif isinstance(left, list):
        assert len(left) == len(right), f"At {path}"
        for idx, (le, re) in enumerate(zip(left, right)):
            _check_ast_eq(le, re, [*path, idx])
    elif isinstance(left, ast.AST):
        for fld in left._fields:
            _check_ast_eq(
                getattr(left, fld, "<MISSING>"),
                getattr(right, fld, "<MISSING>"),
                [*path, fld],
            )
    else:
        raise TypeError(f"At: {path}, {type(left)}")


MISSING = ()


def _check_fields_attributes(e, path):
    if isinstance(e, Primitives):
        return
    elif isinstance(e, list):
        for idx, x in enumerate(e):
            _check_fields_attributes(x, [*path, idx])
    elif isinstance(e, ast.AST):
        names = {*e._fields, *e._attributes}
        assert {x for x in e.__dict__} - names == set(), f"At {path}: {e}"
        for fld in names:
            fld_v = getattr(e, fld, MISSING)
            assert fld_v is not MISSING, f"At {path}: {e} missing {fld}"
            _check_fields_attributes(fld_v, [*path, fld])
    else:
        raise TypeError(f"At: {path}, {type(e)}: {e}")


def assert_eq_ast(fst, *rest):
    reference = _to_stmts(fst)
    _check_fields_attributes(reference, [])
    unparsed = unparse(reference)
    for v in rest:
        stmts = _to_stmts(v)
        _check_fields_attributes(stmts, [])
        assert unparsed == unparse(stmts)
        _check_ast_eq(reference, stmts, [])


def drill(nodes, path):
    for elt in path:
        if isinstance(elt, int):
            nodes = nodes[elt]
        else:
            nodes = getattr(nodes, elt)
    return nodes


def dump(x, *path):
    nodes = drill(_to_stmts(x), path)
    assert False, explode_ast(nodes)


class InstanceOf:
    """Utility class to check that a given value is an instance of a class."""

    def __init__(self, ty):
        self.ty = ty

    def __eq__(self, x):
        return isinstance(x, self.ty)
