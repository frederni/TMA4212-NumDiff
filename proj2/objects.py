import numpy as np

import enum
import dataclasses as dc
from typing import Callable, Union


class Method(enum.Enum):
    abdul_special_5 = 1
    abdul_special_9 = 2
    fast_2d_5 = 3
    fast_2d_9 = 4

    @property
    def stencil(self):
        return 5 if self.name[-1] == "5" else 9

    @property
    def dst_dim(self):
        return 2 if self in [Method.fast_2d_5, Method.fast_2d_9] else 1


@dc.dataclass
class Params:
    """Parameters defining the problem to solve

    :param method: Method to use for solver
    :param N: Number of points (x_0, ..., x_N) where x_0 = 0 and x_N = 1
    :param f: Function defining f = nabla^4 u (and f = nabla^2 v)
    :param h: Step lengths (read only). Equivalent to 1/N
    :param dims: Problem dimension (read only). Equivalent to (N-1)**2
    :param F: Evaluated f in the evaluation points
    """
    method: Method
    N: int
    f: Callable
    F: Union[np.ndarray, None] = None

    @property
    def h(self):
        """Step-lengths"""
        return 1/self.N

    @property
    def dims(self):
        """Problem dimensions"""
        return (self.N - 1)**2

    @property
    def method_str(self):
        return str(self.method.name)


@dc.dataclass
class Result:
    """Result of solving the biharmonic equation

    :param params: Problem parameters
    :param sol: Soulution u. (f = nabla^4 u)
    :param v: Intermediate solution v (f = nabla^2 v)
    :param time: Time spent computing the solution
    """
    params: Params
    sol: np.ndarray
    v: np.ndarray
    time: Union[float, None]
