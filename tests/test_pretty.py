from __future__ import annotations

import dataclasses
from typing import Any

from otf import pretty

NULL_BREAK = pretty.break_with("")


@dataclasses.dataclass
class Add:
    left: Any
    right: Any


def mk_doc(v):
    match v:
        case []:
            return pretty.text("[]")
        case list(z):
            sep = pretty.text(",") + pretty.BREAK
            acc = NULL_BREAK
            first = True
            for x in z:
                if not first:
                    acc += sep
                else:
                    first = False
                acc += mk_doc(x)
            return (
                pretty.text("[")
                + pretty.nest(4, acc)
                + NULL_BREAK
                + pretty.text("]")
                + pretty.EMPTY
            )
        case int(i):
            return pretty.text(str(i))
        case Add(l, r):
            return (
                pretty.agrp(mk_doc(l) + pretty.BREAK + pretty.text("+"))
                + pretty.BREAK
                + mk_doc(r)
            )


def pp(v, width=20):
    return mk_doc(v).to_string(width)


L10 = """\
[
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9
]\
"""

# This is intentionally weird. If we wanted add to come out nicely we should
# have put it in a grp.
ADD = """\
[
    1232341234145345634643657,
    1 +
    2
]\
"""


def test_nested():
    assert pp(list(range(3))) == "[0, 1, 2]"
    assert pp(Add(1, 2)) == "1 + 2"
    assert pp([*range(10)]) == L10
    assert pp([1232341234145345634643657, Add(1, 2)]) == ADD


QUICK_BROWN_FOX = "The quick brown fox jumps over the lazy dog"


def test_groups():
    words = QUICK_BROWN_FOX.split(" ")
    doc = None
    for w in words:
        wdoc = pretty.text(w)
        if doc is None:
            doc = wdoc
        else:
            doc += pretty.BREAK + wdoc
    as_lines = ("\n").join(words)
    v = pretty.vgrp(doc)
    h = pretty.hgrp(doc)
    a = pretty.agrp(doc)
    f = pretty.fgrp(doc)
    assert v.to_string(10) == v.to_string(100) == QUICK_BROWN_FOX
    assert h.to_string(10) == h.to_string(100) == as_lines
    assert a.to_string(10) == as_lines
    assert a.to_string(100) == QUICK_BROWN_FOX

    assert f.to_string(20) == "The quick brown fox\njumps over the lazy\ndog"
