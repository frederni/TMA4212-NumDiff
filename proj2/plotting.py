import numpy as np
import matplotlib.pyplot as plt

import algorithm as alg
from objects import Method, Params, Result
from typing import Tuple, Union, Callable

try:
    import tqdm
except ModuleNotFoundError:
    tqdm = lambda x: x  # noqa

def trueFunc(xx, yy):
    pass

def get_time_error(N_arr: np.ndarray, method: Method, f: Callable, solution: Callable) -> Tuple[np.ndarray, np.ndarray]:
    """Get the time and error array for the given Ns and the given method"""
    error_arr = np.ones_like(N_arr) * np.inf
    time_arr = np.ones_like(N_arr) * np.inf

    # Set test parameters

    test_params = Params(method, None, f)

    for idx, N in enumerate(tqdm(N_arr)):
        test_params.N = N

        # Calculate the expected solution
        x = np.linspace(0, 1, test_params.N + 1)[1:-1]
        xx, yy = np.meshgrid(x, x)
        expected_sol = solution(xx, yy)

        # Solve problem
        res = alg.solve(test_params)

        solution = res.sol

        error_arr[idx] = np.linalg.norm(solution - expected_sol) / np.sqrt(solution.size)
        time_arr[idx] = res.time

    return error_arr, time_arr

def

