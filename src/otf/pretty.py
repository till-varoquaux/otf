"""``otf.pretty``: A pretty printing library
=========================================

The library is based on Christian Lindig's "strictly pretty" [`pdf
<https://lindig.github.io/papers/strictly-pretty-2000.pdf>`_] article.

"""

from __future__ import annotations

import dataclasses
import enum
import io
from typing import Callable, TextIO

__all__ = (
    "Doc",
    "EMPTY",
    "text",
    "BREAK",
    "nest",
    "group",
)


class Mode(enum.Enum):
    "Specify the layout of a group"
    Flat = enum.auto()
    Break = enum.auto()


# doc =
# | DocNil
# | DocCons of doc * doc
# | DocText of string
# | DocNest of int * doc
# | DocBreak of string
# | DocGroup of doc


class Doc:
    """Type used to represent documents

    This constructor should never be called directly

    Documents can be concatenated via the ``+`` operator.
    """

    def __add__(self, other: Doc) -> Doc:
        return DocCons(self, other)

    def to_string(self, width: int = 80) -> str:
        """Render this document to a string"""
        return to_string(width, self)


@dataclasses.dataclass(slots=True)
class DocNil(Doc):
    pass


@dataclasses.dataclass(slots=True)
class DocCons(Doc):
    left: Doc
    right: Doc


@dataclasses.dataclass(slots=True)
class DocText(Doc):
    text: str


@dataclasses.dataclass(slots=True)
class DocNest(Doc):
    indent: int
    doc: Doc


@dataclasses.dataclass(slots=True)
class DocBreak(Doc):
    text: str


@dataclasses.dataclass(slots=True)
class DocGroup(Doc):
    doc: Doc


# let (^^) x y = DocCons(x,y)
# let empty = DocNil
# let text s = DocText(s)
# let nest i x = DocNest(i,x)
# let break = DocBreak(" ")
# let breakWith s = DocBreak(s)
# let group d = DocGroup(d)

#: The empty document
EMPTY: Doc = DocNil()


#: Turns a string into a document
#:
#: Args:
#:  s(str):
def text(s: str) -> Doc:
    """
    Turns a string into a document

    Args:
      s(str)

    Returns:
      Doc:
    """
    return DocText(s)


def nest(indentation: int, doc: Doc) -> Doc:
    """Set the indentation level for a document.

    Args:
      indentation(int):
      doc(Doc):

    Returns:
      Doc:
    """
    return DocNest(indentation, doc)


#: A break is either rendered as a space or a newline followed by a spaces (the
#: number of spaces is determined by the indentation level)
BREAK: Doc = DocBreak(" ")

break_with: Callable[[str], Doc] = DocBreak


def group(doc: Doc) -> Doc:
    """
    Breaks inside the group are either turned into spaces or newline.

    Args:
      doc(Doc):

    Returns
      Doc:
    """
    return DocGroup(doc)


# type sdoc =
# | SNil
# | SText of string * sdoc
# | SLine of int * sdoc (* newline + spaces *)


class SDoc:
    pass


class SNil(SDoc):
    pass


SNIL: SDoc = SNil()


@dataclasses.dataclass(slots=True)
class SText(SDoc):
    text: str
    doc: SDoc


@dataclasses.dataclass(slots=True)
class SLine(SDoc):
    indent: int
    doc: SDoc


# let rec sdocToString = function
#   | SNil -> ""
#   | SText(s,d) -> s ^ sdocToString d
#   | SLine(i,d) -> let prefix = String.make i ’ ’
#                   in nl ^ prefix ^ sdocToString d


def render_sdoc(sdoc: SDoc, out: TextIO) -> None:
    while True:
        match sdoc:
            case SText(s, sdoc):
                out.write(s)
            case SLine(i, sdoc):
                out.write("\n")
                out.write(" " * i)
            case _:
                assert isinstance(sdoc, SNil)
                return


# let rec fits w = function
#   | _ when w < 0 -> false
#   | [] -> true
#   | (i,m,DocNil) :: z -> fits w z
#   | (i,m,DocCons(x,y)) :: z -> fits w ((i,m,x)::(i,m,y)::z)
#   | (i,m,DocNest(j,x)) :: z -> fits w ((i+j,m,x)::z)
#   | (i,m,DocText(s)) :: z -> fits (w - strlen s) z
#   | (i,Flat, DocBreak(s)) :: z -> fits (w - strlen s) z
#   | (i,Break,DocBreak(_)) :: z -> true (* impossible *)
#   | (i,m,DocGroup(x)) :: z -> fits w ((i,Flat,x)::z)


# NOTE: OCaml's list are linked list. The algorithm does a lot of
# deconstructing/reconstructing of head::tail. If we used normal python lists
# we'd convert a lot of O(1) operation in O(n) operations.
@dataclasses.dataclass(slots=True)
class LL:
    width: int
    mode: Mode
    doc: Doc
    _succ: LL | None = None


def fits(w: int, elts: LL | None) -> bool:
    if w < 0:
        return False
    match elts:
        case None:
            return True
        case LL(i, m, DocNil(), z):
            return fits(w, z)
        case LL(i, m, DocCons(x, y), z):
            return fits(w, LL(i, m, x, LL(i, m, y, z)))
        case LL(i, m, DocNest(j, x), z):
            return fits(w, LL(i + j, m, x, z))
        case LL(i, m, DocText(s), z):
            return fits(w - len(s), z)
        case LL(i, Mode.Flat, DocBreak(s), z):
            return fits(w - len(s), z)
        case LL(i, Mode.Break, DocBreak(_), z):
            # The article is wrong: this case is possible, the code written by
            # Lindig for qc-- has the fix:
            # https://github.com/nrnrnr/qc--/blob/ec56191b669/cllib/pp.nw#L413
            return True
        case LL(i, m, DocGroup(x), z):
            return fits(w, LL(i, Mode.Flat, x, z))
    # unreachable
    assert False, elts  # pragma: no cover


# let rec format w k = function
#   | [] -> SNil
#   | (i,m,DocNil) :: z -> format w k z
#   | (i,m,DocCons(x,y)) :: z -> format w k ((i,m,x)::(i,m,y)::z)
#   | (i,m,DocNest(j,x)) :: z -> format w k ((i+j,m,x)::z)
#   | (i,m,DocText(s)) :: z -> SText(s,format w (k + strlen s) z)
#   | (i,Flat, DocBreak(s)) :: z -> SText(s,format w (k + strlen s) z)
#   | (i,Break,DocBreak(s)) :: z -> SLine(i,format w i z)
#   | (i,m,DocGroup(x)) :: z -> if fits (w-k) ((i,Flat,x)::z)
#                               then format w k ((i,Flat ,x)::z)
#                               else format w k ((i,Break,x)::z)


def format(w: int, k: int, elts: LL | None) -> SDoc:
    match elts:
        case None:
            return SNIL
        case LL(i, m, DocNil(), z):
            return format(w, k, z)
        case LL(i, m, DocCons(x, y), z):
            return format(w, k, LL(i, m, x, LL(i, m, y, z)))
        case LL(i, m, DocNest(j, x), z):
            return format(w, k, LL(i + j, m, x, z))
        case LL(i, m, DocText(s), z):
            return SText(s, format(w, k + len(s), z))
        case LL(i, Mode.Flat, DocBreak(s), z):
            return SText(s, format(w, k + len(s), z))
        case LL(i, Mode.Break, DocBreak(s), z):
            return SLine(i, format(w, i, z))
        case LL(i, m, DocGroup(x), z):
            if fits(w - k, LL(i, Mode.Flat, x, z)):
                return format(w, k, LL(i, Mode.Flat, x, z))
            else:
                assert not isinstance(x, DocBreak)
                return format(w, k, LL(i, Mode.Break, x, z))
    # unreachable
    assert False  # pragma: no cover


def to_string(width: int, doc: Doc) -> str:
    sdoc = format(width, 0, LL(0, Mode.Flat, DocGroup(doc)))
    out = io.StringIO()
    render_sdoc(sdoc, out)
    return out.getvalue()
