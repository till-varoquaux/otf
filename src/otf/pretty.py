"""``otf.pretty``: A pretty printing library
=========================================

The library is based on Christian Lindig's "strictly pretty" [`pdf
<https://lindig.github.io/papers/strictly-pretty-2000.pdf>`_] article.

The code was modified to add the features from the `QuickC-- implementation
<https://github.com/nrnrnr/qc--/blob/master/cllib/pp.nw>`_

"""

# IPython has an implementation of linding's pretty printer which a lot further
# from the article:
#
#   https://github.com/ipython/ipython/blob/master/IPython/lib/pretty.py
#
# In turn it's based on:
#
#   https://github.com/ruby/ruby/blob/master/lib/prettyprint.rb
#
# The nice thing about this implementation is that greedily pretty prints the
# document as it is being created. It doesn't have the vgrp and hgrp and it's
# far enough from Linding's article that it's not obvious the semantic has been
# preserved.

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
    while w >= 0:
        match elts:
            case None:
                return True
            case LL(_, m, DocNil(), z):
                elts = z
                continue
            case LL(i, m, DocCons(x, y), z):
                elts = LL(i, m, x, LL(i, m, y, z))
                continue
            case LL(i, m, DocNest(j, x), z):
                elts = LL(i + j, m, x, z)
                continue
            case LL(_, m, DocText(s), z):
                w -= len(s)
                elts = z
                continue
            case LL(_, Mode.FLAT, DocBREAK(s), z):
                w -= len(s)
                elts = z
                continue
            case LL(_, Mode.FILL | Mode.BREAK, DocBREAK(_), _):
                return True
            case LL(i, _, DocGroup(_, x), z):
                elts = LL(i, Mode.FLAT, x, z)
                continue
        # unreachable
        assert False, elts  # pragma: no cover
    return False


# Note: This reference code is from the qc-- code. It's nearly the same as the
# one in the article but it has a couple of extra cases (to handle the fgrp,
# vgrp and hgrp). It's also written in CPS form to avoid stack overflows.
#
# CPython does not have tail call optimisation so it makes no sense to use
# CPS. Instead we rewrote the function as a loop and replaced `cons` and `consl`
# with direct writes to out
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
def format(w: int, k: int, elts: LL | None, out: TextIO) -> None:
    def sline(i: int) -> None:
        out.write("\n")
        out.write(" " * i)

    stext = out.write

    while elts is not None:
        match elts:
            case LL(i, m, DocNil(), z):
                elts = z
                continue
            case LL(i, m, DocCons(x, y), z):
                elts = LL(i, m, x, LL(i, m, y, z))
                continue
            case LL(i, m, DocNest(j, x), z):
                elts = LL(i + j, m, x, z)
                continue
            case LL(i, m, DocText(s), z):
                stext(s)
                k += len(s)
                elts = z
                continue
            case LL(i, Mode.FLAT, DocBREAK(s), z):
                stext(s)
                k += len(s)
                elts = z
                continue
            case LL(i, Mode.FILL, DocBREAK(s), z):
                if fits(w - k - len(s), z):
                    stext(s)
                    k += len(s)
                    elts = z
                    continue
                else:
                    sline(i)
                    k = i
                    elts = z
                    continue
            case LL(i, Mode.BREAK, DocBREAK(s), z):
                sline(i)
                k = i
                elts = z
                continue
            case LL(
                i, _, DocGroup(Mode.FLAT | Mode.FILL | Mode.BREAK as m, x), z
            ):
                elts = LL(i, m, x, z)
                continue
            case LL(i, _, DocGroup(Mode.AUTO, x), z):
                if fits(w - k, LL(i, Mode.FLAT, x, z)):
                    elts = LL(i, Mode.FLAT, x, z)
                    continue
                else:
                    assert not isinstance(x, DocBREAK)
                    elts = LL(i, Mode.BREAK, x, z)
                    continue
        # unreachable
        assert False  # pragma: no cover


def to_string(width: int, doc: Doc) -> str:
    out = io.StringIO()
    format(width, 0, LL(0, Mode.FLAT, DocGroup(Mode.AUTO, doc)), out)
    return out.getvalue()
