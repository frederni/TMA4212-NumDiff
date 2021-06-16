from typing import Callable
import time

from objects import Params, Result, Method


def add_timer(func: Callable[[Params], Result]) -> Callable[[Params], Result]:

    def timed_func(params: Params) -> Result:
        start_time = time.perf_counter_ns()
        res = func(params)
        res.time = (time.perf_counter_ns() - start_time) / 1E9
        return res

    return timed_func
