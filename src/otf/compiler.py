from __future__ import annotations

import abc
import ast
import collections
import contextvars
import copy
import dataclasses
import functools
import inspect
import types
import typing
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterable,
    Mapping,
    ParamSpec,
    TypedDict,
    TypeVar,
)

from . import analyze, ast_utils, pack, parser, utils

T = TypeVar("T")
P = ParamSpec("P")

FunctionType = TypeVar("FunctionType", bound=Callable[..., Any])

_OtfEnv: contextvars.ContextVar[Environment] = contextvars.ContextVar(
    "OtfEnvironment"
)

_OtfWorkflow: contextvars.ContextVar[
    Workflow[Any, Any]
] = contextvars.ContextVar("OtfWorkflow")


def _mk_arguments(sig: inspect.Signature) -> ast.arguments:
    """Turn a signature into ast.arguments"""
    # We ignore the default values, they should be punched in via __defaults__
    # and __kwdefaults__ on the function
    posonlyargs = []
    args = []
    vararg = None
    kwonlyargs = []
    kwarg = None

    for param in sig.parameters.values():
        arg = ast.arg(
            arg=param.name,
            lineno=0,
            col_offset=1,
            end_lineno=0,
            end_col_offset=1,
        )
        kind = param.kind
        if kind == inspect.Parameter.POSITIONAL_ONLY:
            posonlyargs.append(arg)
        elif kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            args.append(arg)
        elif kind == inspect.Parameter.VAR_POSITIONAL:
            vararg = arg
        elif kind == inspect.Parameter.KEYWORD_ONLY:
            kwonlyargs.append(arg)
        elif inspect.Parameter.VAR_KEYWORD:
            kwarg = arg
        else:  # pragma: no cover
            # unreachable
            assert False, kind
    return ast.arguments(
        posonlyargs=posonlyargs,
        args=args,
        vararg=vararg,
        kwonlyargs=kwonlyargs,
        kw_defaults=[None] * len(kwonlyargs),
        kwarg=kwarg,
        defaults=[],
    )


def _mk_function_def(fn: parser.Function[Any, Any]) -> ast.FunctionDef:
    return ast.FunctionDef(
        fn.name,
        args=_mk_arguments(fn.signature),
        body=list(fn.statements),
        decorator_list=[],
        returns=None,
        type_comment=None,
        lineno=fn.lineno,
        col_offset=fn.col_offset,
        end_lineno=fn.end_lineno,
        end_col_offset=fn.end_col_offset,
    )


_value_missing = object()


def _mk_runnable(
    fn: parser.Function[P, T], env: dict[str, Any]
) -> Callable[P, T]:
    fun_def = _mk_function_def(fn)
    code = compile(
        ast.Module(body=[fun_def], type_ignores=[]),
        filename=fn.filename,
        mode="exec",
    )
    prev = env.get(fn.name, _value_missing)
    exec(code, env)
    raw = env[fn.name]

    kwdefaults = {}
    defaults = []

    for param in fn.signature.parameters.values():
        if param.default is param.empty:
            continue
        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            kwdefaults[param.name] = param.default
        else:
            defaults.append(param.default)

    raw.__defaults__ = tuple(defaults)
    if kwdefaults:
        raw.__kwdefaults__ = kwdefaults
    if prev is _value_missing:
        del env[fn.name]
    else:
        env[fn.name] = prev
    return raw  # type: ignore[no-any-return]


class Function(Generic[P, T]):
    """A wrapper around a compiled function"""

    # This allows us to do things like having a defined "__reduce__" function.
    _fun: Callable[P, T] | None
    _origin: parser.Function[P, T]

    def __init__(
        self,
        name: str,
        signature: parser.ExplodedSignature,
        body: str,
    ) -> None:
        self._fun = None
        self._origin = parser._implode_function(
            name=name, body=body, signature=signature
        )

    @classmethod
    def from_parser_function(
        cls, origin: parser.Function[P, T]
    ) -> Function[P, T]:
        res = cls.__new__(cls)
        res._fun = None
        res._origin = origin
        return res

    def _compile(self, env: Environment) -> None:
        self._fun = _mk_runnable(self._origin, env.data)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        if self._fun is None:
            self._compile(_OtfEnv.get())
            assert self._fun is not None
        return self._fun(*args, **kwargs)


@pack.register(pickle=True)
def _explode_function(value: Function[P, T]) -> pack.base.Reduced[Any]:
    args, kwargs = pack.base.shallow_reduce(value._origin)
    return Function, args, kwargs


class ExplodedClosure(TypedDict, total=True):
    """A serializable representation of a Closure"""

    environment: Environment
    # TODO: Change to a generic TypeDict when we move to python 3.11
    # https://bugs.python.org/issue44863
    target: Function[Any, Any]


class Closure(Generic[P, T]):
    """A callable otf function"""

    environment: Environment
    target: Function[P, T]

    def __init__(
        self, environment: Environment, target: Function[P, T]
    ) -> None:
        self.environment = environment
        self.target = target

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        token = _OtfEnv.set(self.environment)
        try:
            return self.target(*args, **kwargs)
        finally:
            _OtfEnv.reset(token)

    @property
    def origin(self) -> parser.Function[P, T]:
        return self.target._origin

    def __str__(self) -> str:
        origin = self.target._origin
        return f"OtfFunction::{origin.name}{origin.signature!s}"

    def __repr__(self) -> str:
        return f"<{self!s} at {hex(id(self))}>"


@pack.register(pickle=True)
def _explode_closure(c: Closure[P, T]) -> pack.base.Reduced[Any]:
    exploded: ExplodedClosure = {
        "environment": c.environment,
        "target": c.target,
    }
    return Closure, (), typing.cast(dict[str, Any], exploded)


class TemplateHole(ast.AST):
    """Used in the templating mechanism to indicate a node that needs to be
    replaced."""

    _fields = ("name",)
    name: str


@dataclasses.dataclass(frozen=True, slots=True)
class _TemplateExpander:
    """Class used to perform the actual template expansion"""

    lineno: int
    col_offset: int
    end_lineno: int | None
    end_col_offset: int | None
    subst: dict[str, ast.AST]

    @typing.overload
    def expand(
        self, node: int | None | str
    ) -> int | None | str:  # pragma: no cover
        ...

    @typing.overload
    def expand(self, node: ast.expr) -> ast.expr:  # pragma: no cover
        ...

    @typing.overload
    def expand(self, node: ast.stmt) -> ast.stmt:  # pragma: no cover
        ...

    @typing.overload
    def expand(self, node: list[ast.AST]) -> list[ast.AST]:  # pragma: no cover
        ...

    def expand(self, node: Any) -> Any:
        if isinstance(node, int | None | str):
            return node
        if isinstance(node, list):
            return [self.expand(x) for x in node]
        if isinstance(node, TemplateHole):
            return self.subst[node.name]
        assert isinstance(node, ast.AST)
        res = type(node)()
        for field_name, field_value in ast.iter_fields(node):
            setattr(res, field_name, self.expand(field_value))
        if "lineno" in res._attributes:
            res.lineno = self.lineno
        if "end_lineno" in res._attributes:
            res.end_lineno = self.end_lineno
        if "col_offset" in res._attributes:
            res.col_offset = self.col_offset
        if "end_col_offset" in res._attributes:
            res.end_col_offset = self.end_col_offset
        return res


class TemplateNamedHolePuncher(ast.NodeTransformer):
    "replaces all the variable named __var_ with template holes."

    PREFIX: ClassVar[str] = "__var_"

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id.startswith(self.PREFIX):
            return TemplateHole(node.id[len(self.PREFIX) :])
        return node


@dataclasses.dataclass()
class Template:
    """AST templates.

    The template are parsed lazily and then processed via ``hole_puncher``.
    """

    txt: str
    hole_puncher: ast.NodeTransformer = TemplateNamedHolePuncher()

    @functools.cached_property
    def statements(self) -> list[ast.stmt]:
        return [self.hole_puncher.visit(x) for x in ast.parse(self.txt).body]

    def __call__(self, loc: ast.AST, **kwds: ast.expr) -> list[ast.stmt]:
        """ "Replaces all the TemplateHole with values from ``kwds``."""
        exp = _TemplateExpander(
            lineno=loc.lineno,
            col_offset=loc.col_offset,
            end_lineno=loc.end_lineno,
            end_col_offset=loc.end_col_offset,
            subst={},
        )
        for k, v in kwds.items():
            if not hasattr(v, "lineno"):
                # Add the missing lineno
                exp.subst[k] = exp.expand(v)
            else:
                exp.subst[k] = v
        return [exp.expand(s) for s in self.statements]


# Workflow generation: Async workflows are compiled to state machines where the
# continuation of every await is in its own state. Those state machines are then
# translate to a ``match `` statements that dispatches to the right state.


class Transition(abc.ABC):
    """Async state machine transition"""

    @abc.abstractmethod
    def emit(self) -> list[ast.stmt]:  # pragma: no cover
        ...


class Exit(Transition):
    """Marks a node as final.

    The state machine will halt when hitting an Exit transition.
    """

    def emit(self) -> list[ast.stmt]:
        return [
            ast.Return(
                value=None,
                lineno=0,
                col_offset=0,
                end_lineno=0,
                end_col_offset=0,
            )
        ]


Loop = TypeVar("Loop", bound=ast.While | ast.For)


@dataclasses.dataclass
class LoopCtrlRewriter(ast.NodeTransformer):
    "Rewrite all the top level ``continue`` and ``break`` statements to jumps."

    filename: str
    break_state: WorkflowState | None
    continue_state: WorkflowState | None

    def visit_block(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        res = []
        for stmt in stmts:
            expanded = self.visit(stmt)
            if isinstance(expanded, list):
                res.extend(expanded)
            else:
                res.append(expanded)
        return res

    def generic_visit(self, node: ast.AST) -> ast.AST:
        infos = analyze.visit_node(node, filename=self.filename)
        if not infos.has_break and not infos.has_continue:
            return node
        return super().generic_visit(copy.copy(node))

    def visit_loop(self, node: Loop) -> Loop:
        node = copy.copy(node)
        node.orelse = self.visit_block(node.orelse)
        return node

    visit_For = visit_While = visit_loop

    def visit_Break(self, node: ast.Break) -> list[ast.stmt]:
        assert self.break_state is not None
        assert self.break_state.idx is not None
        return GotoTemplate(
            node,
            dest=ast_utils.constant(anchor=node, value=self.break_state.idx),
        )

    def visit_Continue(self, node: ast.Continue) -> list[ast.stmt]:
        assert self.continue_state is not None
        assert self.continue_state.idx is not None
        return GotoTemplate(
            node,
            dest=ast_utils.constant(anchor=node, value=self.continue_state.idx),
        )


TCodeBlock = TypeVar("TCodeBlock", bound="CodeBlock")


class CodeBlock:
    """A block of code that ends in a state machine transition."""

    filename: str
    statements: list[ast.stmt]
    out_transition: Transition | None
    break_state: WorkflowState | None
    continue_state: WorkflowState | None
    _sealed: bool

    def __init__(
        self,
        filename: str,
        break_state: WorkflowState | None,
        continue_state: WorkflowState | None,
    ) -> None:
        self.statements = []
        self.filename = filename
        self.out_transition = None
        self.break_state = break_state
        self.continue_state = continue_state
        self._sealed = False

    def append(self, e: ast.stmt) -> None:
        assert not self._sealed
        self.statements.append(e)

    def get_body(self) -> list[ast.stmt]:
        assert self._sealed
        res = self.statements
        if self.break_state or self.continue_state:
            loop_rewritter = LoopCtrlRewriter(
                self.filename,
                break_state=self.break_state,
                continue_state=self.continue_state,
            )
            res = loop_rewritter.visit_block(self.statements)
        else:
            res = self.statements[:]
        if self.out_transition is not None:
            res.extend(self.out_transition.emit())
        return res

    def seal(self: TCodeBlock, out_transition: Transition) -> TCodeBlock:
        self.out_transition = out_transition
        self._sealed = True
        infos = analyze.visit_block(self.statements, filename=self.filename)
        if infos.exits:
            self.out_transition = None
        if not infos.has_continue:
            self.continue_state = None
        if not infos.has_break:
            self.break_state = None
        return self


class InlineCodeBlock(CodeBlock):
    pass


class WorkflowState(CodeBlock):
    """A State in the state machine

    Unlike a normal :class:`CodeBlock` a State can be jumped to and will be
    compiled as a ``case ..``
    """

    idx: int

    def __init__(
        self,
        filename: str,
        idx: int,
        break_state: WorkflowState | None,
        continue_state: WorkflowState | None,
    ) -> None:
        super().__init__(filename, break_state, continue_state)
        self.idx = idx

    def as_case(self) -> ast.match_case:
        return ast.match_case(
            guard=None,
            pattern=(
                ast.MatchValue(
                    lineno=0,
                    col_offset=1,
                    end_lineno=0,
                    end_col_offset=1,
                    value=ast_utils.constant(self.idx),
                )
            ),
            body=self.get_body(),
        )


class PrivateState(WorkflowState):
    pass


AssignTemplate = Template("__var_dest = _otf_val")


class PublicState(WorkflowState):
    """State that can be jumped to from outside the state machines

    These states are either the node or places where we resume from an ``await``
    statement.

    """

    assign: ast.expr | None

    def __init__(
        self,
        filename: str,
        idx: int,
        break_state: WorkflowState | None,
        continue_state: WorkflowState | None,
        assign: ast.expr | None,
    ) -> None:
        super().__init__(
            filename,
            idx,
            break_state=break_state,
            continue_state=continue_state,
        )
        self.assign = assign

    def get_body(self) -> list[ast.stmt]:
        body = super().get_body()
        if self.assign is not None:
            return [*AssignTemplate(self.assign, dest=self.assign), *body]
        return body


@dataclasses.dataclass
class Conditional(Transition):
    anchor: ast.stmt
    test: ast.expr
    body: CodeBlock
    orelse: CodeBlock

    def emit(self) -> list[ast.stmt]:
        return [
            ast.If(
                test=self.test,
                body=self.body.get_body(),
                orelse=self.orelse.get_body(),
                lineno=self.anchor.lineno,
                col_offset=self.anchor.col_offset,
                end_lineno=self.anchor.end_lineno,
                end_col_offset=self.anchor.end_col_offset,
            ),
        ]


GotoTemplate = Template("_otf_pos = __var_dest\n" "continue")


@dataclasses.dataclass
class Jump(Transition):
    anchor: ast.stmt
    dest: WorkflowState

    def emit(self) -> list[ast.stmt]:
        return GotoTemplate(
            self.anchor,
            dest=ast_utils.constant(anchor=self.anchor, value=self.dest.idx),
        )


SuspendTemplate = Template(
    # fmt: off
    "return _otf_suspend("
    "  position=__var_dest, variables=locals(), awaiting=__var_value"
    ")"
    # fmt: on
)


@dataclasses.dataclass
class Suspend(Transition):
    awaiting: ast.expr
    dest: PublicState

    def emit(self) -> list[ast.stmt]:
        return SuspendTemplate(
            self.awaiting,
            dest=ast_utils.constant(anchor=self.awaiting, value=self.dest.idx),
            value=self.awaiting,
        )


InitTemplate = Template(
    "if __var_srep in _otf_variables: "
    " __var_dest = _otf_variables[__var_srep]"
)


@dataclasses.dataclass
class _FsmState:
    src: parser.Function[Any, Any]
    environment: Mapping[str, Any]
    states: list[WorkflowState]
    public_states: int
    private_states: int

    @property
    def filename(self) -> str:
        return self.src.filename

    def mk_public_state(
        self,
        assign: ast.expr | None,
        break_state: WorkflowState | None,
        continue_state: WorkflowState | None,
    ) -> PublicState:
        idx = self.public_states
        self.public_states += 1
        state = PublicState(
            filename=self.filename,
            idx=idx,
            assign=assign,
            break_state=break_state,
            continue_state=continue_state,
        )
        self.states.append(state)
        return state

    def mk_private_state(
        self,
        break_state: WorkflowState | None,
        continue_state: WorkflowState | None,
    ) -> PrivateState:
        self.private_states += 1
        idx = -self.private_states
        state = PrivateState(
            filename=self.filename,
            idx=idx,
            break_state=break_state,
            continue_state=continue_state,
        )
        self.states.append(state)
        return state

    def mk_codeblock(
        self,
        break_state: WorkflowState | None,
        continue_state: WorkflowState | None,
    ) -> InlineCodeBlock:
        return InlineCodeBlock(
            self.filename,
            break_state=break_state,
            continue_state=continue_state,
        )


class FsmCompiler(abc.ABC):
    @property
    @abc.abstractmethod
    def filename(self) -> str:  # pragma: no cover
        ...

    @abc.abstractmethod
    def public_state(
        self, assign: ast.expr | None = None
    ) -> PublicState:  # pragma: no cover
        ...

    @abc.abstractmethod
    def private_state(self) -> PrivateState:  # pragma: no cover
        ...

    @abc.abstractmethod
    def codeblock(self) -> CodeBlock:  # pragma: no cover
        ...

    @abc.abstractmethod
    def loop_compiler(
        self, start: WorkflowState, stop: WorkflowState
    ) -> FsmCompiler:  # pragma: no cover
        ...

    def handle(self, node: ast.stmt, current: CodeBlock) -> CodeBlock:
        infos = analyze.visit_node(node, filename=self.filename)
        if infos.awaits == ():
            current.append(node)
            return current
        match node:
            case ast.Assign(targets=targets, value=ast.Await(value=value)):
                assert len(targets) == 1, ast.unparse(node)
                return self.emit_suspend(
                    target=targets[0], value=value, origin=current
                )
            case ast.AnnAssign(target=target, value=ast.Await(value=value)):
                return self.emit_suspend(
                    target=target, value=value, origin=current
                )
            case ast.Expr(ast.Await(value=value)):
                return self.emit_suspend(
                    target=None, value=value, origin=current
                )
            case ast.If(test=test, body=body, orelse=orelse):
                self.assert_sync(test)
                body_start = self.codeblock()
                else_start = self.codeblock()
                current.seal(
                    Conditional(
                        anchor=node,
                        test=test,
                        body=body_start,
                        orelse=else_start,
                    )
                )
                body_end = self.compile_list(body, current=body_start)
                else_end = self.compile_list(orelse, current=else_start)
                join = self.private_state()
                goto = Jump(anchor=node, dest=join)
                body_end.seal(goto)
                else_end.seal(goto)
                return join
            case ast.While(test=test, body=body, orelse=orelse):
                self.assert_sync(test)
                if orelse:
                    # Even guido thinks this feature was not worth having:
                    # https://mail.python.org/pipermail/python-ideas/2009-October/006157.html
                    utils.syntax_error(
                        msg="While ... Else .. not supported",
                        filename=self.filename,
                        node=orelse[0],
                    )
                start = self.private_state()
                stop = self.private_state()
                goto_stop = Jump(anchor=node, dest=stop)
                goto_start = Jump(anchor=node, dest=start)
                current.seal(goto_start)
                lc = self.loop_compiler(start=start, stop=stop)
                match test:
                    # Peephole optimisation for `while True loops`, This is
                    # mostly useful to keep the example graphs in the
                    # documentation easy to read.
                    case ast.Constant(value=True):
                        start.break_state = start
                        start.continue_state = stop
                        lc.compile_list(body, current=start).seal(goto_start)
                    case _:
                        body_start = lc.codeblock()
                        lc.compile_list(body, current=body_start).seal(
                            goto_start
                        )
                        start.seal(
                            Conditional(
                                anchor=node,
                                test=test,
                                body=body_start,
                                orelse=self.codeblock().seal(goto_stop),
                            )
                        )
                return stop
            case _:
                utils.syntax_error(
                    msg="Await not supported here",
                    filename=self.filename,
                    node=infos.awaits[0],
                )

    def compile_list(
        self, statements: Iterable[ast.stmt], current: CodeBlock
    ) -> CodeBlock:
        for stmt in statements:
            current = self.handle(stmt, current)
        return current

    def assert_sync(self, node: ast.AST) -> None:
        infos = analyze.visit_node(node, filename=self.filename)
        if infos.awaits != ():
            utils.syntax_error(
                msg="Await not supported here",
                filename=self.filename,
                node=infos.awaits[0],
            )

    def emit_suspend(
        self, target: ast.expr | None, value: ast.expr, origin: CodeBlock
    ) -> PublicState:
        dest = self.public_state(assign=target)
        origin.seal(Suspend(awaiting=value, dest=dest))
        return dest


@dataclasses.dataclass
class LoopCompiler(FsmCompiler):
    fsm: _FsmState
    start: WorkflowState
    stop: WorkflowState

    @property
    def filename(self) -> str:
        return self.fsm.src.filename

    def public_state(self, assign: ast.expr | None = None) -> PublicState:
        return self.fsm.mk_public_state(
            assign=assign, break_state=self.stop, continue_state=self.start
        )

    def private_state(self) -> PrivateState:
        return self.fsm.mk_private_state(
            break_state=self.stop, continue_state=self.start
        )

    def codeblock(self) -> InlineCodeBlock:
        return self.fsm.mk_codeblock(
            break_state=self.stop, continue_state=self.start
        )

    def loop_compiler(
        self, start: WorkflowState, stop: WorkflowState
    ) -> FsmCompiler:
        return LoopCompiler(fsm=self.fsm, start=start, stop=stop)


class WorkflowCompiler(_FsmState, FsmCompiler):
    def __init__(
        self, src: parser.Function[Any, Any], environment: Mapping[str, Any]
    ) -> None:
        super().__init__(
            src=src,
            environment=environment,
            states=[],
            public_states=0,
            private_states=0,
        )
        self.public_state()

    def public_state(self, assign: ast.expr | None = None) -> PublicState:
        return self.mk_public_state(
            assign=assign, break_state=None, continue_state=None
        )

    def private_state(self) -> PrivateState:
        return self.mk_private_state(break_state=None, continue_state=None)

    def codeblock(self) -> CodeBlock:
        return self.mk_codeblock(break_state=None, continue_state=None)

    def loop_compiler(
        self, start: WorkflowState, stop: WorkflowState
    ) -> FsmCompiler:
        return LoopCompiler(fsm=self, start=start, stop=stop)

    def link(self) -> parser.Function[Any, Any]:
        infos = analyze.visit_function(self.src)
        vars = sorted(
            (
                *infos.bound_vars,
                *(x for x in infos.free_vars if x not in self.environment),
            )
        )
        body = []
        node = self.src.statements[0]
        for x in vars:
            body.extend(
                InitTemplate(
                    node,
                    srep=ast.Constant(x, kind=None),
                    dest=ast.Name(x, ctx=ast.Store()),
                )
            )
        match_stmt = ast.Match(
            lineno=self.src.lineno,
            col_offset=self.src.col_offset,
            end_lineno=self.src.end_lineno,
            end_col_offset=self.src.end_col_offset,
            subject=ast_utils.name(id="_otf_pos", anchor=self.src),
            cases=[x.as_case() for x in self.states],
        )
        true = ast_utils.constant(value=True, anchor=self.src)

        body.append(
            ast.While(
                test=true,
                body=[match_stmt],
                orelse=[],
                lineno=self.src.lineno,
                col_offset=self.src.col_offset,
                end_lineno=self.src.end_lineno,
                end_col_offset=self.src.end_col_offset,
            )
        )

        return parser.Function(
            name=self.src.name,
            filename=self.src.filename,
            signature=inspect.Signature(
                [
                    inspect.Parameter(
                        name=name,
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                    for name in ("_otf_variables", "_otf_pos", "_otf_val")
                ]
            ),
            statements=tuple(body),
        )

    @classmethod
    def mk(
        cls, fn: parser.Function[Any, T], environment: Mapping[str, Any]
    ) -> WorkflowCompiler:
        wfc = cls(fn, environment=environment)
        last = wfc.compile_list(fn.statements, current=wfc.states[0])
        last.seal(Exit())
        return wfc


def compile_worflow(
    fn: parser.Function[Any, T], environment: Mapping[str, Any]
) -> parser.Function[Any, T | Suspension]:
    wfc = WorkflowCompiler.mk(fn, environment)
    return wfc.link()


class Workflow(Generic[P, T]):
    environment: Environment
    _compiled: Callable[[dict[str, Any], int, Any], T | Suspension]
    origin: parser.Function[P, T]
    awaits: tuple[ast.Await, ...]

    def __init__(
        self, environment: Environment, origin: parser.Function[P, T]
    ) -> None:
        self.environment = environment
        self.origin = origin
        compiled_fn = compile_worflow(origin, environment=environment)
        infos = analyze.visit_function(origin)
        self.awaits = infos.awaits
        self._compiled = _mk_runnable(  # type: ignore
            compiled_fn, env=environment.data
        )

    def _resume(
        self, variables: dict[str, Any], position: int, value: Any
    ) -> T | Suspension:
        env_token = _OtfEnv.set(self.environment)
        workflow_token = _OtfWorkflow.set(self)
        try:
            return self._compiled(  # type: ignore
                _otf_variables=variables, _otf_pos=position, _otf_val=value
            )
        finally:
            _OtfEnv.reset(env_token)
            _OtfWorkflow.reset(workflow_token)

    def freeze(self, *args: P.args, **kwargs: P.kwargs) -> Suspension:
        "Create a suspension of this workflow on the first line"
        bound = self.origin.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return Suspension._make(
            position=0,
            variables=bound.arguments,
            awaiting=None,
            workflow=self,
        )


class ExplodedSuspension(TypedDict, total=True):
    """A serializable representation of a Suspension"""

    code: str
    position: int
    variables: dict[str, Any]
    awaiting: Any
    environment: Environment


@dataclasses.dataclass
class Suspension:
    """Represents a checkpoint in the execution of a workflow

    A suspension captures the code of a workflow, a position in that code and
    all the local variables. It's a `continuation
    <https://en.wikipedia.org/wiki/Continuation>`_ that can be reified.
    """

    position: int
    variables: dict[str, Any]
    awaiting: Any
    workflow: Workflow[Any, Any]

    def __init__(
        self,
        code: str,
        *,
        position: int,
        environment: Environment,
        variables: dict[str, Any],
        awaiting: Any,
    ) -> None:
        self.position = position
        self.variables = variables
        self.awaiting = awaiting
        function = parser._gen_function(
            name="workflow",
            body=code,
            signature=inspect.Signature(),
        )
        self.workflow = Workflow[Any, Any](
            environment=environment, origin=function
        )

    @staticmethod
    def _make(
        position: int,
        variables: dict[str, Any],
        awaiting: Any,
        workflow: Workflow[Any, Any],
    ) -> Suspension:
        # We want the default constructor to be the one used by the
        # serialisation code (to make dump_text pretty).
        # _make is a constructor that should only be used internally.
        res = Suspension.__new__(Suspension)
        res.position = position
        res.variables = variables
        res.awaiting = awaiting
        res.workflow = workflow
        return res

    def resume(self, value: Any) -> Any:
        return self.workflow._resume(
            position=self.position, variables=self.variables, value=value
        )

    @property
    def code(self) -> str:
        return self.workflow.origin.body

    @property
    def environment(self) -> Environment:
        return self.workflow.environment

    @property
    def lineno(self) -> int | None:
        """The line we are currently awaiting on (set to None if we haven't entered the
        body of the function yet).

        """
        if self.position == 0:
            return None
        point = self.workflow.awaits[self.position - 1]
        return point.lineno - self.workflow.origin.lineno + 1


@pack.register(pickle=True)
def _explode_suspension(
    suspension: Suspension,
) -> pack.base.Reduced[Any]:
    exploded: ExplodedSuspension = {
        "code": suspension.code,
        "position": suspension.position,
        "awaiting": suspension.awaiting,
        "variables": suspension.variables,
        "environment": suspension.environment,
    }
    return Suspension, (), typing.cast(dict[str, Any], exploded)


def _otf_suspend(
    position: int, variables: dict[str, Any], awaiting: Any
) -> Suspension:
    return Suspension._make(
        position=position,
        variables={
            k: v for k, v in variables.items() if not k.startswith("_otf_")
        },
        awaiting=awaiting,
        workflow=_OtfWorkflow.get(),
    )


@functools.cache
def _get_builtins() -> Mapping[str, Any]:
    env = {"_otf_suspend": _otf_suspend}
    code = compile(
        ast.Module(body=[], type_ignores=[]),
        filename="<_otf_builtin>",
        mode="exec",
    )
    exec(code, env)
    return types.MappingProxyType(env)


class Environment(collections.UserDict[str, Any]):
    """An otf environment contains functions and values."""

    def __init__(self, /, **kwargs: Any) -> None:
        super().__init__(dict(_get_builtins()), **kwargs)

    @typing.overload
    def function(
        self, /, *, lazy: bool = False
    ) -> Callable[[Callable[P, T]], Closure[P, T]]:  # pragma: no cover
        ...

    @typing.overload
    def function(
        self, fn: Callable[P, T], /, *, lazy: bool = False
    ) -> Closure[Any, T]:  # pragma: no cover
        ...

    def function(
        self, fn: Callable[P, T] | None = None, /, *, lazy: bool = False
    ) -> Closure[P, T] | Callable[[Callable[P, T]], Closure[P, T]]:
        """A decorator to add a function to this environment.

        The decorator can be called directly on a function::

            @env.function
            def f(x: int) -> int:
              ...

        Or with arguments::

            @env.function(lazy=True)
            def f(x: int) -> int:
              ...

        """

        def bind(fn: Callable[P, T]) -> Closure[P, T]:
            parsed = parser.Function[P, T].from_function(fn)
            wrapped = Function[P, T].from_parser_function(origin=parsed)
            if not lazy:
                wrapped._compile(self)
            self.data[parsed.name] = wrapped
            res = Closure[P, T](target=wrapped, environment=self)
            functools.update_wrapper(res, fn)
            return res

        if fn is None:
            return bind
        return bind(fn)

    def workflow(self, fn: Callable[P, T]) -> Workflow[P, T]:
        """Workflow decorator

        Turn a function written with ``async``/``await`` into a workflow.
        """

        # TODO: talk about :mod:`asyncio`
        parsed = parser.Function[P, T].from_function(fn)
        return Workflow[P, T](environment=self, origin=parsed)

    def _as_dict(self) -> dict[str, Any]:
        builtins = _get_builtins()
        return {k: v for k, v in self.data.items() if k not in builtins}


@pack.register(pickle=True)
def _explode_environment(
    environment: Environment,
) -> tuple[Any, tuple[()], dict[str, Any]]:
    return Environment, (), environment._as_dict()
