"""Analyze python Asts
"""
import ast
import dataclasses
import functools
import inspect
import itertools
import linecache
import types
import typing

from otf import parser

__all__ = ("visit_node", "visit_function", "AstInfos")


@dataclasses.dataclass(frozen=True, slots=True)
class AstInfos:
    """Informations collected from an AST"""

    is_async: bool = False
    bound_vars: typing.Mapping[
        str, ast.Name | inspect.Parameter
    ] = types.MappingProxyType({})
    free_vars: typing.Mapping[
        str, ast.Name | ast.Global
    ] = types.MappingProxyType({})

    def __iadd__(self, other: "AstInfos") -> "AstInfos":
        if self is _EMPTY_INFOS:
            return other
        if other is _EMPTY_INFOS:
            return self
        bound_vars = {}
        free_vars = {}
        for k, bv in itertools.chain(
            self.bound_vars.items(), other.bound_vars.items()
        ):
            if self._is_global(k) or other._is_global(k):
                continue
            if k not in bound_vars:
                bound_vars[k] = bv

        for k, fv in itertools.chain(
            self.free_vars.items(), other.free_vars.items()
        ):
            if k in bound_vars:
                continue
            # it's a syntax error to declare a variable global after using
            # it. By taking the first declaration we ensure we'll always get the
            # global.
            if k in free_vars:
                continue
            free_vars[k] = fv

        return AstInfos(
            is_async=self.is_async or other.is_async,
            bound_vars=types.MappingProxyType(bound_vars),
            free_vars=types.MappingProxyType(free_vars),
        )

    def _is_global(self, k: str) -> bool:
        v = self.free_vars.get(k, None)
        return isinstance(v, ast.Global)


_EMPTY_INFOS = AstInfos()


class AstInfosCollector(ast.NodeVisitor):
    """Visitor that extracts information for nodes in an AST

    AstInfoCollector memoizes all the nodes it visits. It's tied to a given
    source file.

    """

    _filename: str
    _cache: dict[ast.AST, AstInfos]

    def __init__(self, filename: str) -> None:
        self._filename = filename
        self._cache = {}

    def visit(self, node: ast.AST) -> AstInfos:
        res = self._cache.get(node, None)
        if res is None:
            res = self._cache[node] = super().visit(node)
        return res

    def generic_visit(self, node: ast.AST) -> AstInfos:
        acc = _EMPTY_INFOS
        for node in ast.iter_child_nodes(node):
            acc += self.visit(node)
        return acc

    def visit_Name(self, node: ast.Name) -> AstInfos:
        ctx_ty = type(node.ctx)
        if ctx_ty == ast.Store:
            return AstInfos(bound_vars=types.MappingProxyType({node.id: node}))
        assert ctx_ty in (ast.Load, ast.Del), node
        return AstInfos(free_vars=types.MappingProxyType({node.id: node}))

    def visit_Global(self, node: ast.Global) -> AstInfos:
        # TODO: this needs to always override bound...
        return AstInfos(
            free_vars=types.MappingProxyType({x: node for x in node.names}),
        )

    def visit_AnnAssign(self, node: ast.AnnAssign) -> AstInfos:
        acc = self.visit(node.target)
        if node.value is not None:
            acc += self.visit(node.value)
        return acc

    def _invalid_node(self, name: str, node: ast.AST) -> typing.NoReturn:
        # https://github.com/python/cpython/blob/3.10/Objects/exceptions.c#L1474
        line = linecache.getline(self._filename, node.lineno)
        raise SyntaxError(
            f"{name!r} not supported in otf functions.",
            (
                self._filename,
                node.lineno,
                node.col_offset,
                line,
                node.end_lineno,
                node.end_col_offset,
            ),
        )

    visit_FunctionDef = (  # type: ignore[assignment]
        visit_AsyncFunctionDef
    ) = functools.partialmethod(
        _invalid_node, "def"
    )  # type: ignore[assignment]
    visit_ClassDef = functools.partialmethod(  # type: ignore[assignment]
        _invalid_node, "class"
    )
    visit_AsyncFor = functools.partialmethod(  # type: ignore[assignment]
        _invalid_node, "async for"
    )
    visit_AsyncWith = functools.partialmethod(  # type: ignore[assignment]
        _invalid_node, "async with"
    )
    visit_Import = functools.partialmethod(  # type: ignore[assignment]
        _invalid_node, "import"
    )
    visit_Lambda = functools.partialmethod(  # type: ignore[assignment]
        _invalid_node, "lambda"
    )
    visit_Match = functools.partialmethod(  # type: ignore[assignment]
        _invalid_node, "match"
    )
    visit_Nonlocal = functools.partialmethod(  # type: ignore[assignment]
        _invalid_node, "nonlocal"
    )
    visit_Yield = (  # type: ignore[assignment]
        visit_YieldFrom
    ) = functools.partialmethod(
        _invalid_node, "yield"
    )  # type: ignore[assignment]


@functools.lru_cache()
def _get_visitor(filename: str) -> AstInfosCollector:
    return AstInfosCollector(filename)


def visit_node(node: ast.AST, filename: str) -> AstInfos:
    return _get_visitor(filename).visit(node)


def visit_function(fn: parser.Function) -> AstInfos:
    visitor = _get_visitor(fn.filename)
    acc = AstInfos(bound_vars=types.MappingProxyType(fn.signature.parameters))
    for stmt in fn.statements:
        acc += visitor.visit(stmt)
    return acc
