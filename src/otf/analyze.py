"""Analyze python Asts
"""
import ast
import builtins
import dataclasses
import functools
import inspect
import itertools
import types
import typing
from typing import Any, Iterable, Mapping, Optional

from otf import parser, utils

__all__ = ("visit_node", "visit_function", "AstInfos")


@dataclasses.dataclass(frozen=True, slots=True)
class AstInfos:
    """Informations collected from an AST"""

    async_ctrl: Optional[ast.Await] = None
    bound_vars: Mapping[
        str, ast.Name | inspect.Parameter
    ] = types.MappingProxyType({})
    free_vars: Mapping[str, ast.Name | ast.Global] = types.MappingProxyType({})
    # If True the code after this statement is unreachable
    exits: bool = False
    has_break: bool = False
    has_continue: bool = False

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
            # it. By taking the first declaration we ensure we'll always
            # prioritize the global.
            if k in free_vars:
                continue
            free_vars[k] = fv

        return AstInfos(
            async_ctrl=self.async_ctrl or other.async_ctrl,
            bound_vars=types.MappingProxyType(bound_vars),
            free_vars=types.MappingProxyType(free_vars),
            exits=False,
            has_continue=self.has_continue or other.has_continue,
            has_break=self.has_break or other.has_break,
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
    _builtins: frozenset[str]

    def __init__(self, filename: str) -> None:
        self._filename = filename
        self._cache = {}
        self._builtins = frozenset(builtins.__dict__)

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

    def visit_Return(self, node: ast.Return) -> AstInfos:
        if node.value is None:
            return AstInfos(exits=True)
        return dataclasses.replace(self.visit(node.value), exits=True)

    def visit_Raise(self, node: ast.Raise) -> AstInfos:
        return dataclasses.replace(self.generic_visit(node), exits=True)

    def visit_Break(self, node: ast.Break) -> AstInfos:
        return AstInfos(exits=True, has_break=True)

    def visit_Continue(self, node: ast.Continue) -> AstInfos:
        return AstInfos(exits=True, has_continue=True)

    def visit_For(self, node: ast.For) -> AstInfos:
        acc = self.visit(node.iter)
        acc += self.visit(node.target)
        acc += dataclasses.replace(
            self.visit_block(node.body), has_continue=False, has_break=False
        )
        acc += self.visit_block(node.orelse)
        return acc

    def visit_While(self, node: ast.While) -> AstInfos:
        acc = self.visit(node.test)
        acc += dataclasses.replace(
            self.visit_block(node.body), has_continue=False, has_break=False
        )
        acc += self.visit_block(node.orelse)
        return acc

    def visit_block(self, stmts: Iterable[ast.stmt]) -> AstInfos:
        acc = _EMPTY_INFOS
        exits = False
        for node in stmts:
            if exits:
                utils.syntax_error(
                    "Unreachable code",
                    filename=self._filename,
                    node=node,
                )
            infos = self.visit(node)
            acc += infos
            exits = infos.exits
        return dataclasses.replace(acc, exits=exits)

    def visit_If(self, node: ast.If) -> AstInfos:
        acc = self.visit(node.test)
        body = self.visit_block(node.body)
        orelse = self.visit_block(node.orelse)
        acc += body
        acc += orelse
        return dataclasses.replace(acc, exits=body.exits and orelse.exits)

    def visit_Name(self, node: ast.Name) -> AstInfos:
        ctx_ty = type(node.ctx)
        if node.id in self._builtins:
            if ctx_ty == ast.Load:
                return AstInfos()
            utils.syntax_error(
                "Modifying builtins is not supported in otf functions.",
                filename=self._filename,
                node=node,
            )
        if node.id.startswith("_otf_"):
            utils.syntax_error(
                'variables with names starting with "_otf_" are reserved for '
                "the otf runtime.",
                filename=self._filename,
                node=node,
            )
        if ctx_ty == ast.Store:
            return AstInfos(bound_vars=types.MappingProxyType({node.id: node}))
        assert ctx_ty in (ast.Load, ast.Del), node
        return AstInfos(free_vars=types.MappingProxyType({node.id: node}))

    def visit_Await(self, node: ast.Await) -> AstInfos:
        return AstInfos(async_ctrl=node)

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
        utils.syntax_error(
            f"{name!r} not supported in otf functions.",
            filename=self._filename,
            node=node,
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


def visit_block(block: Iterable[ast.stmt], filename: str) -> AstInfos:
    return _get_visitor(filename).visit_block(block)


def visit_function(fn: parser.Function[Any, Any]) -> AstInfos:
    visitor = _get_visitor(fn.filename)
    acc = AstInfos(bound_vars=types.MappingProxyType(fn.signature.parameters))
    acc += visitor.visit_block(fn.statements)
    return dataclasses.replace(acc, exits=False)
