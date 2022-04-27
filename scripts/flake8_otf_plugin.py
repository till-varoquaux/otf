"""A flake8 plugin enforce the style rules of the project
"""

from __future__ import annotations

import ast
import importlib.metadata
from typing import Any, Iterator, Type


class StringAnnotFinder(ast.NodeVisitor):
    errors: list[tuple[int, int, str]]

    def __init__(self) -> None:
        self.errors = []
        self.has_from_future = False

    def _check_annot(self, node: ast.expr | None) -> None:
        if node is None:
            return
        for sub_node in ast.walk(node):
            match sub_node:
                case ast.Constant(str(_)):
                    self.errors.append(
                        (
                            sub_node.lineno,
                            sub_node.col_offset,
                            "OTF2: string type annotation",
                        )
                    )

    def visit_FunctionDef(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        self._check_annot(node.returns)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        self._check_annot(node.annotation)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_annot(node.annotation)
        self.visit(node.target)
        if node.value is not None:
            self.visit(node.value)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module != "__future__":
            return
        for alias in node.names:
            if alias.name == "annotations":
                self.has_from_future = True
                return

    def visit_Module(self, node: ast.Module) -> None:
        self.generic_visit(node)
        if not self.has_from_future:
            self.errors.append(
                (1, 0, 'OTF1: missing "from __future__ import annotations"')
            )

    visit_AsyncFunctionDef = visit_FunctionDef


class CheckLazyAnnot:
    options = None
    name = __name__
    version = importlib.metadata.version("otf")

    def __init__(self, tree: ast.Module, filename: str) -> None:
        self._tree = tree
        self._filename = filename

    def run(self) -> Iterator[tuple[int, int, str, Type[Any]]]:
        v = StringAnnotFinder()
        v.visit(self._tree)
        for line, col, msg in v.errors:
            yield (line, col, msg, type(self))
