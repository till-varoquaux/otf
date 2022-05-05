"""``otf.pretty``: A pretty printing library
=========================================

The library is based on Christian Lindig's "strictly pretty" [`pdf
<https://lindig.github.io/papers/strictly-pretty-2000.pdf>`_] article.

The code was modified to add the features from the `QuickC-- implementation
<https://github.com/nrnrnr/qc--/blob/master/cllib/pp.nw>`_

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
    "fgrp",
    "hgrp",
    "vgrp",
)


class Mode(enum.Enum):
    "Specify the layout of a group"
    FLAT = enum.auto()
    BREAK = enum.auto()
    FILL = enum.auto()
    # Only valid in groups
    AUTO = enum.auto()


# doc =
# | DocNil
# | DocCons of doc * doc
# | DocText of string
# | DocNest of int * doc
# | DocBREAK of string
# | DocGroup of gmode * doc


class Doc:
    """Type used to represent documents

    This constructor should never be called directly

    Documents can be concatenated via the ``+`` operator.
    """

    def __add__(self, other: Doc) -> Doc:
        return DocCons(self, other)

    def to_string(self, width: int = 80) -> str:
        """Render this document to a string

        args:
          width(int):
        """
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
class DocBREAK(Doc):
    text: str


@dataclasses.dataclass(slots=True)
class DocGroup(Doc):
    mode: Mode
    doc: Doc


# let (^^) x y = DocCons(x,y)
# let empty = DocNil
# let text s = DocText(s)
# let nest i x = DocNest(i,x)
# let break = DocBREAK(" ")
# let breakWith s = DocBREAK(s)
# let hgrp d = DocGroup(GFlat, d)
# let vgrp d = DocGroup(GBreak,d)
# let agrp d = DocGroup(GAuto, d)
# let fgrp d = DocGroup(GFill, d)

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
BREAK: Doc = DocBREAK(" ")

break_with: Callable[[str], Doc] = DocBREAK


def hgrp(doc: Doc) -> Doc:
    """
    BREAKs inside the group are never always into newline.

    Args:
      doc(Doc):

    Returns
      Doc:
    """
    return DocGroup(Mode.BREAK, doc)


def vgrp(doc: Doc) -> Doc:
    """
    BREAKs inside the group are never turned into newline.

    Args:
      doc(Doc):

    Returns
      Doc:
    """
    return DocGroup(Mode.FLAT, doc)


def agrp(doc: Doc) -> Doc:
    """
    BREAKs inside the group are either turned into spaces or newline.

    Args:
      doc(Doc):

    Returns
      Doc:
    """
    return DocGroup(Mode.AUTO, doc)


def fgrp(doc: Doc) -> Doc:
    """
    BREAKs inside the group are considered individually.

    Args:
      doc(Doc):

    Returns
      Doc:
    """
    return DocGroup(Mode.FILL, doc)


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


# NOTE: OCaml's list are linked list. The algorithm does a lot of
# deconstructing/reconstructing of head::tail. If we used normal python lists
# we'd convert a lot of O(1) operation in O(n) operations.
@dataclasses.dataclass(slots=True)
class LL:
    width: int
    mode: Mode
    doc: Doc
    _succ: LL | None = None


# let rec fits w = function
#     | _ when w < 0                   -> false
#     | []                             -> true
#     | (i,m,DocNil)              :: z -> fits w z
#     | (i,m,DocCons(x,y))        :: z -> fits w ((i,m,x)::(i,m,y)::z)
#     | (i,m,DocNest(j,x))        :: z -> fits w ((i+j,m,x)::z)
#     | (i,m,DocText(s))          :: z -> fits (w - strlen s) z
#     | (i,Flat, DocBreak(s))     :: z -> fits (w - strlen s) z
#     | (i,Fill, DocBreak(_))     :: z -> true
#     | (i,Break,DocBreak(_))     :: z -> true
#     | (i,m,DocGroup(_,x))       :: z -> fits w ((i,Flat,x)::z)


def fits(w: int, elts: LL | None) -> bool:
    if w < 0:
        return False
    match elts:
        case None:
            return True
        case LL(_, m, DocNil(), z):
            return fits(w, z)
        case LL(i, m, DocCons(x, y), z):
            return fits(w, LL(i, m, x, LL(i, m, y, z)))
        case LL(i, m, DocNest(j, x), z):
            return fits(w, LL(i + j, m, x, z))
        case LL(_, m, DocText(s), z):
            return fits(w - len(s), z)
        case LL(_, Mode.FLAT, DocBREAK(s), z):
            return fits(w - len(s), z)
        case LL(_, Mode.FILL | Mode.BREAK, DocBREAK(_), _):
            return True
        case LL(i, _, DocGroup(_, x), z):
            return fits(w, LL(i, Mode.FLAT, x, z))
    # unreachable
    assert False, elts  # pragma: no cover


# Note: This reference code is from the qc-- code. It's nearly the same as the
# one in the article but it has a couple of extra cases (to handle the fgrp,
# vgrp and hgrp). It's also written in CPS form to avoid stack overflows.
#
# CPython does not have tail call optimisation so it makes no sense to use
# CPS. We could eventually rewrite `format` to use a stack and avoid the stack
# blow outs.
#
# let cons  s post z = post (SText (s, z))
# let consl i post z = post (SLine (i, z))
# let rec format w k l post = match l with
#     | []                             -> post SNil
#     | (i,m,DocNil)              :: z -> format w k z post
#     | (i,m,DocCons(x,y))        :: z -> format w k ((i,m,x)::(i,m,y)::z) post
#     | (i,m,DocNest(j,x))        :: z -> format w k ((i+j,m,x)::z) post
#     | (i,m,DocText(s))          :: z ->
#                                        format w (k + strlen s) z (cons s post)
#     | (i,Flat, DocBreak(s))     :: z ->
#                                        format w (k + strlen s) z (cons s post)
#     | (i,Fill, DocBreak(s))     :: z -> let l = strlen s in
#                                         if   fits (w - k - l) z
#                                         then format w (k+l) z (cons s post)
#                                         else format w  i    z (consl i post)
#     | (i,Break,DocBreak(s))     :: z -> format w i z (consl i post)
#     | (i,m,DocGroup(GFlat ,x))  :: z -> format w k ((i,Flat ,x)::z) post
#     | (i,m,DocGroup(GFill ,x))  :: z -> format w k ((i,Fill ,x)::z) post
#     | (i,m,DocGroup(GBreak,x))  :: z -> format w k ((i,Break,x)::z) post
#     | (i,m,DocGroup(GAuto, x))  :: z -> if fits (w-k) ((i,Flat,x)::z)
#                                         then format w k ((i,Flat ,x)::z) post
#                                         else format w k ((i,Break,x)::z) post
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
        case LL(i, Mode.FLAT, DocBREAK(s), z):
            return SText(s, format(w, k + len(s), z))
        case LL(i, Mode.FILL, DocBREAK(s), z):
            if fits(w - k - len(s), z):
                return SText(s, format(w, (k + len(s)), z))
            else:
                return SLine(i, format(w, i, z))
        case LL(i, Mode.BREAK, DocBREAK(s), z):
            return SLine(i, format(w, i, z))
        case LL(i, _, DocGroup(Mode.FLAT | Mode.FILL | Mode.BREAK as m, x), z):
            return format(w, k, LL(i, m, x, z))
        case LL(i, _, DocGroup(Mode.AUTO, x), z):
            if fits(w - k, LL(i, Mode.FLAT, x, z)):
                return format(w, k, LL(i, Mode.FLAT, x, z))
            else:
                assert not isinstance(x, DocBREAK)
                return format(w, k, LL(i, Mode.BREAK, x, z))
    # unreachable
    assert False  # pragma: no cover


def to_string(width: int, doc: Doc) -> str:
    out = io.StringIO()
    sdoc = format(width, 0, LL(0, Mode.FLAT, DocGroup(Mode.AUTO, doc)))
    render_sdoc(sdoc, out)
    return out.getvalue()
