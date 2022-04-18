"""
``otf.local_scheduler``: Run workflows locally
==============================================

A toy local scheduler to demonstrate how to use OTF
"""
from __future__ import annotations

import concurrent.futures
import contextvars
import dataclasses
import multiprocessing
import types
import uuid
from typing import Any, Callable, Generic, ParamSpec, Type, TypeVar

from . import compiler, pack, runtime

__all__ = ("defer", "Future", "Scheduler", "Checkpoint", "Result")

T = TypeVar("T")
P = ParamSpec("P")

_CurrentContext: contextvars.ContextVar[Scheduler] = contextvars.ContextVar(
    "OtfLocalSchedulerContext"
)


@dataclasses.dataclass
class Future(Generic[T]):
    "Handle on the result of a deferred computation"

    scheduler_id: str
    local_id: int

    @property
    def uid(self) -> str:
        return f"{self.scheduler_id}:{self.local_id}"

    @staticmethod
    def _otf_reconstruct(uid: str) -> Future[Any]:
        scheduler_id, local_id = uid.split(":")
        return Future(scheduler_id, int(local_id))


@dataclasses.dataclass
class TracePoint:
    parent: Checkpoint | None


@dataclasses.dataclass
class Result(TracePoint, Generic[T]):
    """The result of running a workflow

    Attributes:
      value(T): The result of the computation
      parent(Checkpoint): The parent step in the computation
    """

    parent: Checkpoint
    value: T


@dataclasses.dataclass
class Checkpoint(TracePoint):
    """
    Attributes:
      suspension(Suspension): A snapshot of the running workflow
      parent(Checkpoint): The parent step in the computation
    """

    suspension: compiler.Suspension


@pack.register
def _explode_future(fut: Future[Any]) -> tuple[Type[Future[Any]], str]:
    return Future, fut.uid


class Scheduler:
    """Class to run workflows locally."""

    uuid: str
    executor: concurrent.futures.ProcessPoolExecutor | None
    futures: list[concurrent.futures.Future[Any]]
    token: contextvars.Token[Scheduler] | None

    def __init__(self) -> None:
        self.executor = None
        self.futures = []
        self.uuid = str(uuid.uuid4())
        self.token = None

    def __enter__(self) -> Scheduler:
        assert self.executor is None
        assert self.token is None
        self.executor = concurrent.futures.ProcessPoolExecutor(
            mp_context=multiprocessing.get_context("spawn")
        )
        self.token = _CurrentContext.set(self)
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_val: BaseException | None,
        exc_traceback: types.TracebackType | None,
    ) -> None:
        assert self.executor is not None
        assert self.token is not None
        _CurrentContext.reset(self.token)
        self.executor.shutdown(wait=True)
        self.executor = None
        self.token = None
        self.futures.clear()

    def submit(self, task: runtime.Task[T]) -> Future[T]:
        assert self.executor is not None
        conc_fut = self.executor.submit(task.run)
        res: Future[T] = Future(
            scheduler_id=self.uuid, local_id=len(self.futures)
        )
        self.futures.append(conc_fut)
        return res

    def wait(self, fut: Future[T]) -> T:
        # Make sure we are getting values from other schedulers
        assert fut.scheduler_id == self.uuid
        conc_fut: concurrent.futures.Future[T] = self.futures[fut.local_id]
        return conc_fut.result()

    def run(
        self, wf: compiler.Workflow[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> Result[T]:
        """Run an async workflow.

        All the futures (ie: the values we "await" on) must be created via
        :func:`dispatch`.
        """
        step = wf.freeze(*args, **kwargs)
        trace: Checkpoint | None = None
        while isinstance(step, compiler.Suspension):
            trace = Checkpoint(parent=trace, suspension=step)
            step = pack.copy(step)
            fut = step.awaiting
            if fut is None:
                value = None
            else:
                value = self.wait(fut)
            step = step.resume(value)
        return Result(
            parent=trace,
            value=step,
        )


def defer(
    fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
) -> Future[T]:
    """Schedules ``fn(*args, **kwargs)`` to be executed in a separate process.

    Must be called from within a :class:`Scheduler` context.
    """
    assert isinstance(fn, compiler.Function | compiler.Closure)
    context = _CurrentContext.get()
    task = runtime.task(fn, *args, **kwargs)
    return context.submit(task)
