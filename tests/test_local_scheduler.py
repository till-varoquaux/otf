from math import floor, sqrt

import otf
from otf import local_scheduler
from otf.local_scheduler import defer


def test():
    e = otf.Environment(floor=floor, sqrt=sqrt, defer=local_scheduler.defer)

    @e.function
    def is_prime(n: int) -> bool:
        if n % 2 == 0:
            return False

        sqrt_n = int(floor(sqrt(n)))
        for i in range(3, sqrt_n + 1, 2):
            if n % i == 0:
                return False
        return True

    @e.workflow
    async def check_five():
        f1, f2, f3, f4, f5 = (defer(is_prime, x) for x in (1, 2, 3, 4, 5))
        v1 = await f1
        v2 = await f2
        v3 = await f3
        v4 = await f4
        v5 = await f5
        return v1, v2, v3, v4, v5

    assert e["is_prime"](5)
    assert not e["is_prime"](4)

    with local_scheduler._run_ctx():
        t = local_scheduler.defer(is_prime, 5)
        assert t.result() is True

    r = local_scheduler.run(check_five)
    assert r == (True, False, True, False, True)
