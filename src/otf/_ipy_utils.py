from __future__ import annotations

import functools
import html
from typing import Any, Iterable, Iterator

import pygments
import pygments.formatters
import pygments.lexers


@functools.lru_cache()
def get_highlight_style() -> str:
    formatter = pygments.formatters.HtmlFormatter(cssclass="otf-highlight")
    styles: str = formatter.get_style_defs(".otf-highlight")
    return styles


class InlineHtmlFormatter(
    pygments.formatters.HtmlFormatter  # type: ignore[type-arg]
):
    def wrap(
        self, source: Iterable[tuple[int, str]], *args: Any, **kwargs: Any
    ) -> Iterator[tuple[int, str]]:
        yield 0, (
            f'<span class="{self.cssclass}" '
            'style="background-color: transparent"><tt>'
        )
        yield from source
        yield 0, "</tt></span>"


def highlight(
    code: str, hl_lines: tuple[int, ...] = (), inline: bool = False
) -> str:
    lexer = pygments.lexers.PythonLexer()
    format_cls = (
        pygments.formatters.HtmlFormatter if not inline else InlineHtmlFormatter
    )
    formatter = format_cls(cssclass="otf-highlight", hl_lines=hl_lines)
    res: str = pygments.highlight(code, lexer, formatter)
    return res


LINE_LEN = 120


def summarize(rep: str) -> str:
    if len(rep) > LINE_LEN:
        trim_size = LINE_LEN // 2 - 2
        cnt = f"{rep[:trim_size]}...f{rep[-trim_size:]}"
        return f"<tt>{html.escape(cnt)}</tt>"
    return highlight(rep, inline=True)
