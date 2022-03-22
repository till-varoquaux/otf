import ast
import json


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


def explode_ast(node):
    """Turns ast nodes in a format that can easily be compared and introspected.

    This is useful because the mapping between python's syntax and python's AST
    is not always straightforward.

    """
    if isinstance(node, (str, int, type(None))):
        return node
    if isinstance(node, (list, tuple)):
        return [explode_ast(v) for v in node]
    assert isinstance(node, ast.AST), node
    return {k: explode_ast(v) for k, v in ast.iter_fields(node)} | {
        "__type__": type(node).__name__
    }


def assert_eq_ast(fst, *rest):
    reference = _to_stmts(fst)
    unparsed = unparse(reference)
    jsoned = json.dumps(explode_ast(reference), indent=2)
    for v in rest:
        stmts = _to_stmts(v)
        assert unparsed == unparse(stmts)
        assert jsoned == json.dumps(explode_ast(stmts), indent=2)
