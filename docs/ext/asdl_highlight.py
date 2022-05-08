# taken from: https://github.com/python/cpython/blob/3.10/Doc/tools/\
# extensions/asdl_highlight.py
from __future__ import annotations

from typing import Any, Callable

bygroups: Callable[..., Any]

from pygments.lexer import RegexLexer, bygroups, include, words  # noqa E402
from pygments.token import (  # noqa E402
    Comment,
    Keyword,
    Name,
    Operator,
    Punctuation,
    Text,
)
from sphinx.highlighting import lexers  # noqa E402

builtin_types = {"identifier", "string", "int", "constant"}


class ASDLLexer(RegexLexer):
    name = "ASDL"
    aliases = ["asdl"]
    filenames = ["*.asdl"]
    _name = r"([^\W\d]\w*)"
    _text_ws = r"(\s*)"

    tokens = {
        "ws": [
            (r"\n", Text),
            (r"\s+", Text),
            (r"--.*?$", Comment.Singleline),
        ],
        "root": [
            include("ws"),
            (
                r"(module)" + _text_ws + _name,
                bygroups(Keyword, Text, Name.Tag),
            ),
            (
                r"(\w+)(\*\s|\?\s|\s)(\w+)",
                bygroups(Name.Builtin.Pseudo, Operator, Name),
            ),
            (words(builtin_types), Name.Builtin),
            (r"attributes", Name.Builtin),
            (
                _name + _text_ws + "(=)",
                bygroups(Name, Text, Operator),
            ),
            (_name, Name.Class),
            (r"\|", Operator),
            (r"{|}|\(|\)", Punctuation),
            (r".", Text),
        ],
    }


def setup(app: Any) -> dict[str, Any]:
    lexers["asdl"] = ASDLLexer()
    return {"version": "1.0", "parallel_read_safe": True}
