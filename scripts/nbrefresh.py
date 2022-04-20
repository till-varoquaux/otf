#!/usr/bin/env python3
from __future__ import annotations

import difflib
import glob
import shutil
import sys
from typing import Iterator

import click
import nbformat  # type: ignore[import]
from nbconvert.preprocessors import ExecutePreprocessor  # type: ignore[import]


def diff(v1: str, v2: str, filename: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            v1.splitlines(keepends=True),
            v2.splitlines(keepends=True),
            fromfile=filename,
        )
    )


MAKE_DET = """

def _make_det():
   import os, random
   r = random.Random(0)
   os.urandom = r.randbytes

_make_det()
del _make_det
"""


def run(orig: str) -> str:
    nb = nbformat.reads(orig, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    # Make sure the run is deterministic by inserting a cell that monkeypatches
    # urandom
    det_cell = nbformat.notebooknode.NotebookNode(
        {
            "cell_type": "code",
            "id": "4f25faa9",
            "metadata": {},
            "outputs": [],
            "source": MAKE_DET,
        }
    )
    nb.cells.insert(0, det_cell)
    ep.preprocess(nb)
    nb.cells.pop(0)
    for cell in nb.cells:
        if "execution_count" in cell:
            cell["execution_count"] -= 1
        metadata = cell["metadata"]
        if "execution" in metadata:
            del metadata["execution"]
    del nb["metadata"]["language_info"]["version"]
    cnt: str = nbformat.writes(nb)
    return cnt + "\n"


def _expand(filenames: list[str]) -> Iterator[str]:
    """Basic support for globs

    Since tox doesn't support globs we add a very rudimentary expansion here.

    """
    for f in filenames:
        if "*" in f:
            yield from glob.glob(f)
        else:
            yield f


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.argument("files", nargs=-1)
def check(files: list[str]) -> None:
    """Check that ipynb files are up to date"""
    columns, _ = shutil.get_terminal_size()
    retcode = 0
    for filename in _expand(files):
        click.echo(f"Checking {filename}".ljust(columns - 5), nl=False)
        with open(filename, "r") as fd:
            orig = fd.read()
        new = run(orig)
        if new == orig:
            click.secho("OK", fg="green")
        else:
            click.secho("FAILED", fg="red")
            click.echo(diff(orig, new, filename=filename), err=True)
            retcode = 1
    sys.exit(retcode)


@cli.command()
@click.argument("files", nargs=-1)
def refresh(files: list[str]) -> None:
    """Re-generate the output cells in ipynb files"""
    columns, _ = shutil.get_terminal_size()
    for filename in _expand(files):
        click.echo(f"Regenerating {filename}".ljust(columns - 10), nl=False)
        with open(filename, "r") as fd:
            orig = fd.read()
        new = run(orig)
        if new == orig:
            click.secho("No change", fg="green")
        else:
            with open(filename, "w") as fd:
                fd.write(new)
            click.secho("Updated", fg="cyan")


if __name__ == "__main__":
    cli()
