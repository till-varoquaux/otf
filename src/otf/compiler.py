import ast
import collections
import contextvars
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
    Mapping,
    Optional,
    ParamSpec,
    TypedDict,
    TypeVar,
)

from otf import analyze, parser, utils

T = TypeVar("T")
P = ParamSpec("P")

FunctionType = TypeVar("FunctionType", bound=Callable[..., Any])

_OtfEnv: contextvars.ContextVar["Environment"] = contextvars.ContextVar(
    "OtfEnvironment"
)

_OtfWorkflow: contextvars.ContextVar[
    "Workflow[Any, Any]"
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


class _OtfFunWrapper(Generic[P, T]):
    """A wrapper around a compiled function"""

    # This allows us to do things like having a defined "__reduce__" function.
    _fun: Optional[Callable[P, T]]
    _origin: parser.Function[P, T]

    def __init__(self, origin: parser.Function[P, T]) -> None:
        self._fun = None
        self._origin = origin

    def _compile(self, env: "Environment") -> None:
        # TODO: Figure out what makes mypy unhappy here
        self._fun = _mk_runnable(
            self._origin, env.data  # type: ignore[arg-type]
        )

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        if self._fun is None:
            self._compile(_OtfEnv.get())
            assert self._fun is not None
        return self._fun(*args, **kwargs)

    def __getstate__(self) -> parser.ExplodedFunction:
        return parser._explode_function(self._origin)

    def __setstate__(self, origin: parser.ExplodedFunction) -> None:
        self._origin = parser._implode_function(origin)
        self._fun = None


class ExplodedClosure(TypedDict, total=True):
    """A serializable representation of a Closure"""

    environment: "Environment"
    # TODO: Change to a generic TypeDict when we move to python 3.11
    # https://bugs.python.org/issue44863
    target: _OtfFunWrapper[Any, Any]


class Closure(Generic[P, T]):
    """A callable otf function"""

    environment: "Environment"
    target: _OtfFunWrapper[P, T]

    def __init__(
        self, environment: "Environment", target: _OtfFunWrapper[P, T]
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

    def __getstate__(self) -> ExplodedClosure:
        # We want to get rid of any __wrapped__ etc that might have been added
        # by functools.update_wrapper
        return {"environment": self.environment, "target": self.target}

    def __str__(self) -> str:
        origin = self.target._origin
        return f"OtfFunction::{origin.name}{origin.signature!s}"

    def __repr__(self) -> str:
        return f"<{self!s} at {hex(id(self))}>"


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
    end_lineno: Optional[int]
    end_col_offset: Optional[int]
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
        if isinstance(node, (int, type(None), str)):
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


@dataclasses.dataclass()
class WorkflowStep:

    idx: int
    statements: list[ast.stmt] = dataclasses.field(default_factory=list)

    def append(self, e: ast.stmt | list[ast.stmt]) -> None:
        if isinstance(e, list):
            self.statements.extend(e)
        else:
            self.statements.append(e)

    def as_case(self) -> ast.match_case:
        return ast.match_case(
            guard=None,
            pattern=(
                ast.MatchValue(
                    lineno=0,
                    col_offset=1,
                    end_lineno=0,
                    end_col_offset=1,
                    value=ast.Constant(
                        kind=None,
                        value=self.idx,
                        lineno=0,
                        col_offset=1,
                        end_lineno=0,
                        end_col_offset=1,
                    ),
                )
            ),
            body=self.statements,
        )


SuspendTemplate = Template(
    # fmt: off
    "return _otf_suspend("
    "position=__var_dest, variables=locals(), awaiting=__var_value"
    ")"
    # fmt: on
)


AssignTemplate = Template("__var_dest = _otf_val")

InitTemplate = Template(
    "if __var_srep in _otf_variables: "
    " __var_dest = _otf_variables[__var_srep]"
)


@dataclasses.dataclass()
class WorkflowCompiler:

    src: parser.Function[Any, Any]
    environment: Mapping[str, Any]
    steps: list[WorkflowStep] = dataclasses.field(
        default_factory=lambda: [WorkflowStep(0)]
    )

    @property
    def filename(self) -> str:
        return self.src.filename

    @property
    def current(self) -> WorkflowStep:
        return self.steps[-1]

    def add_step(self) -> WorkflowStep:
        step = WorkflowStep(len(self.steps))
        self.steps.append(step)
        return step

    def handle(self, node: ast.stmt) -> None:
        infos = analyze.visit_node(node, filename=self.filename)
        if infos.async_ctrl is None:
            self.current.append(node)
            return
        match node:
            case ast.Assign(targets=targets, value=ast.Await(value=value)):
                assert len(targets) == 1, ast.unparse(node)
                self.emit_suspend(target=targets[0], value=value)
            case ast.AnnAssign(target=target, value=ast.Await(value=value)):
                self.emit_suspend(target=target, value=value)
            case ast.Expr(ast.Await(value=value)):  # type: ignore[misc]
                self.emit_suspend(target=None, value=value)
            case _:
                utils.syntax_error(
                    msg="Await not supported here",
                    filename=self.filename,
                    node=infos.async_ctrl,
                )

    def emit_suspend(self, target: Optional[ast.expr], value: ast.expr) -> None:
        origin = self.current
        dest = self.add_step()
        origin.append(
            SuspendTemplate(
                value, dest=ast.Constant(dest.idx, kind=None), value=value
            )
        )
        if target is not None:
            dest.append(AssignTemplate(target, dest=target))

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
        body.append(
            ast.Match(
                lineno=self.src.lineno,
                col_offset=self.src.col_offset,
                end_lineno=self.src.end_lineno,
                end_col_offset=self.src.end_col_offset,
                subject=ast.Name(
                    id="_otf_pos",
                    ctx=ast.Load(),
                    lineno=self.src.lineno,
                    col_offset=self.src.col_offset,
                    end_lineno=self.src.end_lineno,
                    end_col_offset=self.src.end_col_offset,
                ),
                cases=[x.as_case() for x in self.steps],
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


def compile_worflow(
    fn: parser.Function[Any, T], environment: Mapping[str, Any]
) -> parser.Function[Any, T | "Suspension"]:
    wfc = WorkflowCompiler(fn, environment=environment)
    for stmt in fn.statements:
        wfc.handle(stmt)
    return wfc.link()


class Workflow(Generic[P, T]):
    environment: "Environment"
    _compiled: Callable[[dict[str, Any], int, Any], T | "Suspension"]
    origin: parser.Function[P, T]

    def __init__(
        self, environment: "Environment", origin: parser.Function[P, T]
    ) -> None:
        self.environment = environment
        self.origin = origin
        compiled_fn = compile_worflow(origin, environment=environment)
        self._compiled = _mk_runnable(  # type: ignore
            compiled_fn, env=environment.data
        )

    def _resume(
        self, variables: dict[str, Any], position: int, value: Any
    ) -> T | "Suspension":
        env_token = _OtfEnv.set(self.environment)
        workflow_token = _OtfWorkflow.set(self)
        try:
            return self._compiled(  # type: ignore
                _otf_variables=variables, _otf_pos=position, _otf_val=value
            )
        finally:
            _OtfEnv.reset(env_token)
            _OtfWorkflow.reset(workflow_token)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T | "Suspension":
        ba = self.origin.signature.bind(*args, **kwargs)
        ba.apply_defaults()
        return self._resume(
            variables=ba.arguments,
            position=0,
            value=None,
        )


class ExplodedSuspension(TypedDict, total=True):
    """A serializable representation of a Suspension"""

    environment: "Environment"
    variables: dict[str, Any]
    awaiting: Any
    code: str
    position: int


@dataclasses.dataclass
class Suspension:
    position: int
    variables: dict[str, Any]
    awaiting: Any
    workflow: "Workflow[Any, Any]"

    def resume(self, value: Any) -> Any:
        return self.workflow._resume(
            position=self.position, variables=self.variables, value=value
        )

    def __getstate__(self) -> ExplodedSuspension:
        return _explode_suspension(self)

    def __setstate__(self, state: ExplodedSuspension) -> None:
        other = _implode_suspension(state)
        self.__dict__ = other.__dict__


def _explode_suspension(suspension: Suspension) -> ExplodedSuspension:
    return {
        "environment": suspension.workflow.environment,
        "variables": suspension.variables,
        "awaiting": suspension.awaiting,
        "code": suspension.workflow.origin.body,
        "position": suspension.position,
    }


def _implode_suspension(exploded: ExplodedSuspension) -> Suspension:
    function = parser._gen_function(
        name="workflow", body=exploded["code"], signature=inspect.Signature()
    )
    workflow = Workflow[Any, Any](
        environment=exploded["environment"], origin=function
    )
    return Suspension(
        position=exploded["position"],
        variables=exploded["variables"],
        awaiting=exploded["awaiting"],
        workflow=workflow,
    )


def _otf_suspend(
    position: int, variables: dict[str, Any], awaiting: Any
) -> Suspension:
    return Suspension(
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
        self, fn: Optional[Callable[P, T]] = None, /, *, lazy: bool = False
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
            wrapped = _OtfFunWrapper[P, T](origin=parsed)
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
        parsed = parser.Function[P, T].from_function(fn)
        return Workflow[P, T](environment=self, origin=parsed)
