import builtins
import inspect
import typing
from typing import Any, Callable, Optional, ParamSpec, TypeVar

from otf import analyze, compiler, utils

__all__ = ("function", "environment")

T = TypeVar("T")
P = ParamSpec("P")

FunctionType = TypeVar("FunctionType", bound=Callable[..., Any])


@typing.overload
def function(f: FunctionType) -> FunctionType:  # pragma: no cover
    ...


@typing.overload
def function(
    *, strict: bool = True
) -> Callable[[FunctionType], FunctionType]:  # pragma: no cover
    ...


def function(
    f: Optional[FunctionType] = None, *, strict: bool = True
) -> Callable[[Callable[P, T]], Callable[P, T]] | FunctionType:

    r"""Wraps the decorated function in its own portable environment.

    The function be will be compiled in its own module and will only have access
    to globals passed in via the :func:`environment` decorator::

      @function
      @environment(i=0)
      def counter() -> int:
        "A counter that increments by 1 every time it's called"
        global i
        i+=1
        return i

    By default the decorator will vet the function for issues like using globals
    that aren't declared in its environment::

      >>> @function
      ... def counter() -> int:
      ...  global i
      ...  i+=1
      ...  return i
      Traceback (most recent call last):
      ...
      SyntaxError: variable 'i' not found in the environment

    This can be turned off with the ``strict`` argument::

      @function(strict=False)
      def counter() -> int:
         ...

    Args:

       strict: do not check the wrapped function.

    Returns:
       ~otf.compiler.Closure:
    """

    def wrapper(f: Callable[P, T]) -> Callable[P, T]:
        env: Optional[compiler.Environment] = getattr(f, "_otf_env", None)
        if env is None:
            env = compiler.Environment()
        wrapped: compiler.Closure[P, T] = env.function(f)
        if strict:
            universe = {*dir(builtins), *env}
            parsed = wrapped.origin
            infos = analyze.visit_function(parsed)
            for name, origin in infos.free_vars.items():
                if name not in universe:
                    utils.syntax_error(
                        f"variable {name!r} not found in the environment",
                        filename=parsed.filename,
                        node=origin,
                    )
        return wrapped

    if f is None:
        return wrapper
    return wrapper(f)


# We could use __closure__ (resp: .__globals__) and __code__.co_freevars
# (resp. co_names) and then filter out all the builtins to automatically create
# the environment for closures. The resulting functions would still not behave
# exactly like we expect (e.g.: rebinding a global within an otf function would
# only change the binding in the otf environment).


def environment(**kwargs: Any) -> Callable[[FunctionType], FunctionType]:
    """Attach an environment to the function.

    All keyword arguments will be declared as variables in the function's
    globals.

    """

    def wrapper(f: FunctionType) -> FunctionType:
        if not inspect.isfunction(f):
            raise TypeError(f"Argument is not a function (type={type(f)})")
        f._otf_env = compiler.Environment(  # type: ignore[attr-defined]
            **kwargs
        )
        # We seem to have a bug in mypy:
        #
        # error: Incompatible return value type (got "FunctionType", expected
        # "FunctionType")
        return f  # type: ignore[return-value]

    return wrapper
