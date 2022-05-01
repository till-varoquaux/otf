from __future__ import annotations

import dataclasses
from typing import Any

from otf import pack


@dataclasses.dataclass
class Sig:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


@pack.register
def _Sig(a: Sig):
    return Sig, a.args, a.kwargs
