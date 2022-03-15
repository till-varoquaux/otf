import ast
import linecache
import typing


def syntax_error(msg: str, filename: str, node: ast.AST) -> typing.NoReturn:
    """Raise a syntax error on a given ast position"""
    # https://github.com/python/cpython/blob/3.10/Objects/exceptions.c#L1474
    raise SyntaxError(
        msg,
        (
            filename,
            node.lineno,
            node.col_offset + 1,
            linecache.getline(filename, node.lineno),
            node.end_lineno,
            node.end_col_offset,
        ),
    )
