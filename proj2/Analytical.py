import numpy as np
from sympy import diff, symbols, sin, pi, exp, lambdify

class Analytical:
    '''
    Contains u and f from task h) as analytical numpy functions
    '''

    class _Backend:
        def u(self, x, y):
            return (np.sin(np.pi * x) * np.sin(np.pi * y)) ** 4 * np.exp(-(x - 0.5) ** 2 - (y - 0.5) ** 2)

        def __init__(self):
            x, y = symbols('x y')
            u_symbolic = (sin(pi * x) * sin(pi * y)) ** 4 * exp(-(x - 0.5) ** 2 - (y - 0.5) ** 2)
            f_symbolic = diff(u_symbolic, x, 4) + 2 * diff(u_symbolic, x, 2, y, 2) + diff(u_symbolic, y, 4)
            self.f = lambdify([x, y], f_symbolic, 'numpy')

    _B = _Backend()

    @classmethod
    def u(cls, x, y):
        return cls._B.u(x, y)

    @classmethod
    def f(cls, x, y):
        return cls._B.f(x, y)
