"""A sphinx plugin that contains custom directives used in otf's documentation.

"""
from __future__ import annotations

import abc
import ast
import dataclasses
import inspect
import io
from typing import Any, Iterator

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.directives import code
from sphinx.domains import Domain
from sphinx.ext import graphviz
from sphinx.util.docutils import SphinxDirective

from otf import compiler, parser


@dataclasses.dataclass(frozen=True, slots=True)
class FakeTranslator:
    body: list[nodes.Node] = dataclasses.field(default_factory=list)


def text_to_fsm(text: str) -> compiler.WorkflowCompiler:
    fn = ast.parse(text).body[0]
    assert isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef)
    g: dict[str, Any] = {}
    exec(f"def fn({ast.unparse(fn.args)}): pass", g)
    sig = inspect.signature(g["fn"])
    parsed = parser.Function[Any, Any](
        name=fn.name,
        filename="<bogus>",
        statements=tuple(fn.body),
        signature=sig,
    )

    return compiler.WorkflowCompiler.mk(
        parsed, environment=compiler.Environment()
    )


C = """
digraph foo {
   "bar" -> "baz";
}
"""


def state_name(state: compiler.CodeBlock) -> str:
    if isinstance(state, compiler.PublicState):
        return f"public_{state.idx}"
    if isinstance(state, compiler.PrivateState):
        return f"private_{-state.idx}"
    return f"inline_{id(state)}"


# def get_all_states():
GR_TRANS = str.maketrans(
    {
        "\n": r"\l",
        '"': r"\"",
        ">": r"\>",
        "<": r"\<",
        "[": r"\[",
        "]": r"\]",
        " ": r"\ ",
    }
)


def compile_fsm_state(state: compiler.CodeBlock) -> str:
    label_fields = {}
    assign = getattr(state, "assign", None)
    if assign is not None:
        label_fields["assign"] = f"{ast.unparse(assign)} = ..."
    if state.statements:
        label_fields["body"] = "\n".join(
            ast.unparse(stmt) for stmt in state.statements
        )
    if isinstance(state.out_transition, compiler.Conditional):
        label_fields[
            "condition"
        ] = f"if {ast.unparse(state.out_transition.test)}:\n  ..."
    if isinstance(state.out_transition, compiler.Suspend):
        label_fields[
            "awaiting"
        ] = f"await {ast.unparse(state.out_transition.awaiting)}"

    label = "|".join(
        f"<{field_name}> {field_content.translate(GR_TRANS)}\\l"
        for field_name, field_content in label_fields.items()
    )
    return f'{state_name(state)} [nojustify=true label="{{{label}}}"];'


# Get rid of the empty node for an inline jump (often the `if false ` branch of
# a while loop)
def compress(state: compiler.CodeBlock) -> compiler.CodeBlock:
    match state:
        case compiler.InlineCodeBlock(
            statements=[], out_transition=compiler.Jump(dest=dest)
        ):
            return dest
        case _:
            return state


def compile_edges(src: compiler.CodeBlock) -> Iterator[str]:
    if src.continue_state is not None:
        yield (
            f"{state_name(src)}:body:w -> {state_name(src.continue_state)}:w "
            "[label=continue]"
        )
    if src.break_state is not None:
        yield (
            f"{state_name(src)}:body:e -> {state_name(src.break_state)}:e "
            "[label=break]"
        )
    match src.out_transition:
        case compiler.Jump(dest=dst):
            yield f"{state_name(src)}:s -> {state_name(dst)}"
        case compiler.Suspend(dest=dst):
            yield f"{state_name(src)}:s -> {state_name(dst)} [style=bold]"
        case compiler.Conditional(body=body, orelse=orelse):
            yield (
                f"{state_name(src)}:s -> {state_name(compress(body))}:n "
                "[label=True]"
            )
            yield (
                f"{state_name(src)}:s -> {state_name(compress(orelse))}:n "
                "[label=False]"
            )


def get_all_nodes(fsm: compiler.WorkflowCompiler) -> list[compiler.CodeBlock]:
    pending = list[compiler.CodeBlock](reversed(fsm.states))
    seen = set[int]()
    res: list[compiler.CodeBlock] = []
    while pending:
        s = pending.pop()
        if id(s) in seen:
            continue
        res.append(s)
        seen.add(id(s))
        if isinstance(s.out_transition, compiler.Conditional):
            pending.append(compress(s.out_transition.body))
            pending.append(compress(s.out_transition.orelse))
    return res


def fsm_to_digraph(fsm: compiler.WorkflowCompiler) -> str:
    buf = io.StringIO()
    buf.write(
        f"digraph {fsm.src.name} {{\n"
        "\n"
        '  node [shape=record fontname="Sans serif" fontsize="12"];\n'
        "  edge [style=dashed];\n\n"
        "  Start [style=invis]\n;"
    )
    states = get_all_nodes(fsm)
    for x in states:
        buf.write(compile_fsm_state(x))
    buf.write("\n")
    buf.write(f"  Start -> {state_name(states[0])} [style=bold];")
    for x in states:
        for edge in compile_edges(x):
            buf.write(f"  {edge};\n")
    buf.write("}")
    return buf.getvalue()


class CompileDirective(SphinxDirective, abc.ABC):

    has_content = False
    required_arguments = 1
    optional_arguments = 0

    def run(self) -> list[nodes.Node]:
        rel_filename, filename = self.env.relfn2path(self.arguments[0])
        self.env.note_dependency(rel_filename)
        # Makes working on this extension easier
        self.env.note_dependency(__file__)

        location = self.get_source_info()
        reader = code.LiteralIncludeReader(
            filename, options={"language": "python"}, config=self.config
        )

        text, _lines = reader.read(location=location)

        return [self.render(text_to_fsm(text), filename=filename)]

    @abc.abstractmethod
    def render(
        self, fsm: compiler.WorkflowCompiler, filename: str
    ) -> nodes.Node:
        ...


class GraphCompileDirective(CompileDirective):
    def render(
        self, fsm: compiler.WorkflowCompiler, filename: str
    ) -> nodes.Node:
        node = graphviz.graphviz()
        node["code"] = fsm_to_digraph(fsm)
        node["align"] = "center"
        node["options"] = {"docname": self.env.docname}
        self.add_name(node)
        return node


class MatchCaseCompileDirective(CompileDirective):
    def render(
        self, fsm: compiler.WorkflowCompiler, filename: str
    ) -> nodes.Node:

        processed = (
            fsm.link()
            .statements[-1]
            .body[0]  # type: ignore[attr-defined]
            .cases
        )

        text = ast.unparse(processed)

        retnode: nodes.Node = nodes.literal_block(text, text, source=filename)
        self.set_source_info(retnode)

        return retnode


class OtfDomain(Domain):
    name = "otf"

    directives = {
        "graph": GraphCompileDirective,
        "match_case": MatchCaseCompileDirective,
    }


def setup(app: Sphinx) -> dict[str, Any]:
    app.add_domain(OtfDomain)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
