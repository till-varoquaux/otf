from __future__ import annotations

import math

import otf
from otf import local_scheduler
from otf.local_scheduler import defer


def test(monkeypatch):
    e = otf.Environment(
        math=otf.NamedReference(math),
        defer=otf.NamedReference(local_scheduler.defer),
    )

    @e.function
    def is_prime(n: int) -> bool:
        if n < 2:
            return False

        if n == 2:
            return True

        if n % 2 == 0:
            return False

        sqrt_n = int(math.floor(math.sqrt(n)))
        for i in range(3, sqrt_n + 1, 2):
            if n % i == 0:
                return False
        return True

    @e.workflow
    async def check(*vals: int) -> list[bool]:
        """Check for prime numbers in a list of values"""
        res = []
        futures = [defer(is_prime, x) for x in vals]
        while futures:
            elt = await futures.pop(0)
            res.append(elt)
            del elt
        return res

    assert e["is_prime"](5)
    assert not e["is_prime"](4)

    with local_scheduler.Scheduler() as schd:
        fut = local_scheduler.defer(is_prime, 5)
        assert schd.wait(fut) is True

    with local_scheduler.Scheduler() as schd:
        trace = schd.run(check, 1, 2, 3, 4, 5)
    assert trace.value == [False, True, True, False, True]

    monkeypatch.setattr("IPython.display.display", lambda _: None)
    trace._ipython_display_()
