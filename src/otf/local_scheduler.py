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
import typing
import uuid
from typing import Any, Callable, Generic, ParamSpec, Type, TypedDict, TypeVar

if typing.TYPE_CHECKING:  # pragma: no cover
    import ipywidgets  # type: ignore[import]

from . import compiler, pack, runtime

__all__ = ("defer", "Future", "Scheduler", "Checkpoint", "Result")

T = TypeVar("T")
P = ParamSpec("P")

_CurrentContext: contextvars.ContextVar[Scheduler] = contextvars.ContextVar(
    "OtfLocalSchedulerContext"
)


class ExplodedFuture(TypedDict, total=True):
    uid: str
    task: runtime.Task[Any]


@dataclasses.dataclass
class Future(Generic[T]):
    "Handle on the result of a deferred computation"

    scheduler_id: str
    local_id: int
    task: runtime.Task[T]

    @property
    def uid(self) -> str:
        return f"{self.scheduler_id}:{self.local_id}"


def future(uid: str, task: runtime.Task[T]) -> Future[T]:
    """Function used by pack to recreate the futures..."""
    scheduler_id, local_id = uid.split(":")
    return Future(scheduler_id, int(local_id), task=task)


@pack.register
def _explode_future(
    fut: Future[Any],
) -> tuple[Callable[..., Future[Any]], tuple[()], dict[str, Any]]:
    kwargs = ExplodedFuture({"uid": fut.uid, "task": fut.task})
    return future, (), typing.cast(dict[str, Any], kwargs)


@dataclasses.dataclass
class TracePoint:
    parent: Checkpoint | None

    # Display as a nice widget in ipython
    def _ipython_display_(self) -> None:
        import ipywidgets
        from IPython import display  # type: ignore

        from otf import _ipy_utils

        trace = []
        curs: TracePoint | None = self
        while curs is not None:
            trace.append(curs)
            curs = curs.parent

        trace.reverse()

        tab = ipywidgets.Tab(
            children=[pt._ipy_widget() for pt in trace],
            # titles = ... doesn't work
        )
        for i, pt in enumerate(trace):
            tab.set_title(i, pt._ipy_title_(i))
        style = _ipy_utils.get_highlight_style()
        display.display(display.HTML(f"<style>{style}</style>"))
        display.display(tab)

    def _ipy_title_(self, index: int) -> str:
        return f"{index} [{type(self).__name__}]"

    def _ipy_widget(self) -> ipywidgets.Widget:  # pragma: no cover
        raise NotImplementedError


@dataclasses.dataclass
class Result(TracePoint, Generic[T]):
    """The result of running a workflow

    Attributes:
      value(T): The result of the computation
      parent(Checkpoint): The parent step in the computation
    """

    parent: Checkpoint
    value: T

    def _ipy_widget(self) -> ipywidgets.Widget:
        import ipywidgets

        from . import _ipy_utils

        html = _ipy_utils.highlight(pack.dump_text(self.value))
        return ipywidgets.HTML(html)

    def _ipy_title_(self, index: int) -> str:
        return "Result"


@dataclasses.dataclass
class Checkpoint(TracePoint):
    """
    Attributes:
      suspension(Suspension): A snapshot of the running workflow
      parent(Checkpoint): The parent step in the computation
    """

    suspension: compiler.Suspension

    def _ipy_widget(self) -> ipywidgets.Widget:
        import ipywidgets

        from . import _ipy_utils

        lineno = self.suspension.lineno
        code = ipywidgets.HTML(
            _ipy_utils.highlight(
                self.suspension.code,
                hl_lines=() if lineno is None else (lineno,),
            )
        )
        elts = [code]
        if self.suspension.awaiting is not None:
            core = _ipy_utils.highlight(
                pack.dump_text(self.suspension.awaiting, indent=4, width=120)
            )
            awaiting = ipywidgets.HTML(f"<b>Awaiting:</b>{core}")
            elts.append(awaiting)

        # Printing the locals
        #
        # Stack overflow has some interesting discussions on the subject:
        #
        # https://stackoverflow.com/questions/37718907/\
        # variable-explorer-in-jupyter-notebook
        local_rows = []
        for k, v in self.suspension.variables.items():
            value = _ipy_utils.summarize(pack.dump_text(v))
            local_rows.append(f"<tr><td><b>{k}</b></td><td>{value}</td></tr>")
        tbody = "".join(local_rows)
        local_table = (
            "<b>Locals:</b><br>"
            '<div class="rendered_html jp-RenderedHTMLCommon"><table>'
            f"{tbody}"
            "</table></div>"
        )
        elts.append(ipywidgets.HTML(local_table))
        vbox = ipywidgets.VBox(elts)
        return vbox


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
            scheduler_id=self.uuid, local_id=len(self.futures), task=task
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
    task = runtime.Task.make(fn, *args, **kwargs)
    return context.submit(task)
