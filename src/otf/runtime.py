"""Various helper functions to support otf runtimes
"""
import dataclasses
from typing import Any, Generic, ParamSpec, TypeVar

from . import compiler

P = ParamSpec("P")
T = TypeVar("T")

__all__ = (
    "Task",
    "task",
)


@dataclasses.dataclass
class Task(Generic[T]):
    """A deferred function application"""

    closure: compiler.Closure[Any, T]
    args: list[Any]
    kwargs: dict[str, Any]

    def run(self) -> T:
        return self.closure(*self.args, **self.kwargs)


def task(
    fn: compiler.Closure[P, T] | compiler._OtfFunWrapper[P, T],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> Task[T]:
    """Create a :class:`Task`.

    This is useful when implementing schedulers. We capture the function, the
    arguments and the environment it runs in.

    """
    if isinstance(fn, compiler._OtfFunWrapper):
        closure = compiler.Closure[P, T](
            environment=compiler._OtfEnv.get(), target=fn
        )
    else:
        closure = fn
    return Task(closure=closure, args=[*args], kwargs=dict(**kwargs))
