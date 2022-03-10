import ast
import collections
import contextvars
import inspect
import typing
from typing import Any, Callable, Generic, Optional, TypeVar

from otf import parser

FunctionType = TypeVar("FunctionType", bound=Callable[..., Any])


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
        arg = ast.arg(arg=param.name)
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


def _mk_function_def(fn: parser.Function) -> ast.FunctionDef:
    fun_def = ast.FunctionDef(
        fn.name,
        args=_mk_arguments(fn.signature),
        body=list(fn.statements),
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    fun_def.lineno = fn.lineno
    fun_def.col_offset = fn.col_offset
    ast.fix_missing_locations(fun_def)
    return fun_def


_value_missing = object()


def _mk_runnable(fn: parser.Function, env: dict[str, Any]) -> FunctionType:
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


_OtfEnv: contextvars.ContextVar["Environment"] = contextvars.ContextVar(
    "OtfEnvironement"
)


class _OtfFunWrapper(Generic[FunctionType]):
    """A wrapper around a compiled function"""

    # This allows us to do things like having a defined "__reduce__" function.
    _fun: Optional[FunctionType]
    _origin: parser.Function

    def __init__(self, origin: parser.Function) -> None:
        self._fun = None
        self._origin = origin

    def _compile(self, env: "Environment") -> None:
        self._fun = _mk_runnable(self._origin, env.data)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._fun is None:
            self._compile(_OtfEnv.get())
            assert self._fun is not None
        return self._fun(*args, **kwargs)


class FunctionReference:
    """A callable otf function

    This is technically a pointer to a function in an ``Environement`` but it
    can be called like the function it points to.

    """

    env: "Environment"
    name: str

    def __init__(self, env: "Environment", name: str) -> None:
        self.name = name
        self.env = env

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        token = _OtfEnv.set(self.env)
        try:
            return self.env[self.name](*args, **kwargs)
        finally:
            _OtfEnv.reset(token)

    @property
    def origin(self) -> parser.Function:
        return typing.cast(_OtfFunWrapper[Any], self.env[self.name])._origin


# TODO: maybe use a module instead of a userdict?
# TODO:
#   Environement can maybe be created as classes:
#    class MyEnv(Environment):
#
#       i = 5
#
#       def f():
#          ...
#
# https://docs.python.org/3/library/importlib.html#importlib.util.module_from_spec
class Environment(collections.UserDict[str, Any]):
    """An otf environement contains functions and values."""

    @typing.overload
    def function(
        self, *, lazy: bool = False
    ) -> Callable[[FunctionType], FunctionType]:  # pragma: no cover
        ...

    @typing.overload
    def function(
        self, fn: FunctionType, *, lazy: bool = False
    ) -> FunctionType:  # pragma: no cover
        ...

    def function(
        self, fn: Optional[FunctionType] = None, *, lazy: bool = False
    ) -> FunctionType | Callable[[FunctionType], FunctionType]:
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

        def bind(fn: FunctionType) -> FunctionType:
            parsed = parser.Function.from_function(fn)
            wrapped: _OtfFunWrapper[FunctionType] = _OtfFunWrapper(
                origin=parsed
            )
            if not lazy:
                wrapped._compile(self)
            self.data[parsed.name] = wrapped
            return typing.cast(
                FunctionType, FunctionReference(name=parsed.name, env=self)
            )

        if fn is None:
            return bind
        return bind(fn)
