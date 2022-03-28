"""A toy local scheduler to demonstrate how to use OTF
"""
import concurrent.futures
import contextlib
import contextvars
import multiprocessing
from typing import Callable, Generator, ParamSpec, TypeVar

from . import compiler, runtime

__all__ = ("defer", "run")

T = TypeVar("T")
P = ParamSpec("P")

_CurrentContext: contextvars.ContextVar[
    concurrent.futures.ProcessPoolExecutor
] = contextvars.ContextVar("OtfLocalSchedulerContext")


@contextlib.contextmanager
def _run_ctx() -> Generator[None, None, None]:
    with contextlib.ExitStack() as es:
        executor = es.enter_context(
            concurrent.futures.ProcessPoolExecutor(
                mp_context=multiprocessing.get_context("spawn")
            )
        )
        token = _CurrentContext.set(executor)
        es.callback(_CurrentContext.reset, token)
        yield


def defer(
    fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
) -> concurrent.futures.Future[T]:
    "Schedules ``fn(*args, **kwargs)`` to be executed in a separate process."
    assert isinstance(fn, (compiler._OtfFunWrapper, compiler.Closure))
    context = _CurrentContext.get()
    task = runtime.task(fn, *args, **kwargs)
    return context.submit(task.run)


def run(wf: compiler.Workflow[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
    """Run an async workflow.

    All the futures (ie: the values we "await" on) must be created via
    :func:`dispatch`.

    """
    with _run_ctx():
        step = wf(*args, **kwargs)
        while isinstance(step, compiler.Suspension):
            print(f"step:{step.awaiting}")
            step = step.resume(step.awaiting.result())
        return step
