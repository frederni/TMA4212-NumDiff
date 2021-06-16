import numpy as np
import scipy as scp
import scipy.fft
import scipy.sparse as spr
import scipy.sparse.linalg

from typing import Tuple, Union
import functools

from objects import Params, Result, Method
import utilities as utils


def solve(params: Params) -> Result:
    """Solve the biharmonic problem given
    
    Thin wrapper around _Solver.solve()

    :param params: Parameters defining the clamped biharmonic problem
    :return: Result of problem
    """
    # Get points on which to calculate
    N = params.N
    x = np.linspace(0, 1, N+1)  # Linspace from x_0=0 to x_{N} = 1
    xx, yy = np.meshgrid(x, x)
    # Get F-array
    params.F = params.f(xx, yy)

    return _Solver.solve(params)


class _Solver:
    """Handling solving of problem"""
    @staticmethod
    @utils.add_timer
    def solve(params: Params) -> Result:
        """Solve the biharmoniq equation for the given parameters

        :param params: Parameters defining the clamped biharmonic problem
        :return: Result of problem
        """
        res = _Solver.base_solver(params)
        return res

    @staticmethod
    def base_solver(params: Params) -> Result:
        """Solve with the 'smart' method, 5-point stencil

        1. Get F
        2. Fourier sine of each column: H
        3. H' = H.T
        4. Get the gamma-matrix for each different j: G_j
        5. Where h_j is a column in H', solve for y_j: G_j @ y_j = h_j
        6. Let y_j be each column in Y': Y = Y'.T
        7. Inverse Fourier sine of each column in Y: v
        Do twice to get u
        """
        if params.method.dst_dim == 1:
            solveHarmonic = _Solver.solveHarmonic
        elif params.method.dst_dim == 2:
            solveHarmonic = _Solver.solveHarmonic2d
        else:
            raise NotImplementedError

        res = Result(params, None, None, None)

        F = params.F

        G = _Solver.makeG(params, F)

        res.v, dst_v = solveHarmonic(params, G=G)

        F = np.pad(res.v, 1, mode='constant', constant_values=0)
        G = _Solver.makeG(params, F)

        res.sol, _ = solveHarmonic(params, G=G)

        return res

    @staticmethod
    def makeG(params: Params, F: np.ndarray) -> np.ndarray:
        """Create the G-matrix from the F-matrix for both 5 and 9-stencil

        F must be of size (N+1)x(N+1)

        :param params: Problem parameters
        :param F: The function evaluated at the points
        :return:
        """
        N = params.N
        assert F.shape == (N+1, N+1)
        if params.method.stencil == 5:
            return F[1:-1, 1:-1] / params.N**2
        elif params.method.stencil == 9:
            G = (1 / 12) * (1 / N ** 2) * (8 * F[1:-1, 1:-1]
                                           + F[1:-1, 0:-2]
                                           + F[1:-1, 2:]
                                           + F[0:-2, 1:-1]
                                           + F[2:, 1:-1]
                                           )
            return G

    @staticmethod
    def solveHarmonic2d(params: Params, G: np.ndarray = None, dst2d_G: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Solve a single iteration of the harmonic equation

        Note that dst2d_G must be the 2d transform, not row-wise 1d.
        Half a bi-harmonic :D

        :param params: Biharmonic problem parameters. Note that this specifies stencil method.
        :param G: Vector defining nabla^2 solution = G. Can not be given if H is given. (Should already be multiplied by h^2)
        :param dst2d_G: NOTE! 2D transform! Discrete sine transform of G. Can not be given if G is given. (Should already be multiplied by h^2)
        :return: Tuple containing (solution, scp.fft.dst(solution).
        """
        assert G is not None or dst2d_G is not None, f"Both G and H is None, at least one mus be given."
        if dst2d_G is None:
            dst2d_G = scp.fft.dstn(G, type=1, norm='ortho')
        else:
            raise NotImplementedError("Sure you called the right function now? This is the 2d harmonic solver.")

        N = params.N
        j_arr = np.arange(1, N)

        if params.method.stencil == 5:
            eigval_arr = -2 + 2*np.cos(j_arr*np.pi / N)
            eigvals = np.add.outer(eigval_arr, eigval_arr)
        elif params.method.stencil == 9:
            eigval_arr = -2 + 2*np.cos(j_arr*np.pi / N)
            eigvals = (1/6) * (np.multiply.outer(eigval_arr, eigval_arr)) + np.add.outer(eigval_arr, eigval_arr)
        else:
            raise NotImplementedError()

        dst2d_G = dst2d_G / eigvals

        # dst2d_G = (1/2)*(res1 + res2)
        # dst2d_G = (dst2d_G / eigval_arr)#  / eigval_arr.T

        return scp.fft.idstn(dst2d_G, type=1, norm='ortho'), dst2d_G

    @staticmethod
    def solveHarmonic(params: Params, G: np.ndarray = None, dst_G: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Solve a single iteration of the harmonic equation

        Half a bi-harmonic :D

        :param params: Biharmonic problem parameters. Note that this specifies stencil method.
        :param G: Vector defining nabla^2 solution = G. Can not be given if H is given. (Should already be multiplied by h^2)
        :param dst_G: Discrete sine transform of G. Can not be given if G is given. (Should already be multiplied by h^2)
        :return: Tuple containing (solution, scp.fft.dst(solution).
        """
        assert G is not None or dst_G is not None, f"Both G and H is None, at least one mus be given."
        if dst_G is None:
            dst_G = scp.fft.dst(G, axis=0, type=1, norm='ortho')

        Y = np.ones_like(dst_G)*np.inf

        for idx, h in enumerate(dst_G):
            # h is row in h, idx is j-1,
            j = idx + 1

            Y[idx] = _Solver.solveTST(_Solver.getEigvals(params, j), h)
            continue

        return scp.fft.idst(Y, axis=0, type=1, norm='ortho'), Y

    @staticmethod
    def solveTST(diags: Tuple[float, float], b: np.ndarray) -> np.ndarray:
        """Solve the TST problem with diagonals diag[0] (on diag) and diag[1] (off diag) defining T in Tx=b

        :param diags: Diagonals (diag_on, diag_off)
        :param b: Vector b in Tx=b
        :return x: Solution to Tx=b
        """
        diag_on, diag_off = diags
        N = b.size
        n = np.arange(1, N+1)
        eigval_factor = diag_on + diag_off*2*np.cos(n*np.pi/(N+1))
        assert eigval_factor.shape == b.shape

        b_dst = scipy.fft.dst(b, type=1, norm='ortho')
        x_dst = b_dst / eigval_factor

        return scipy.fft.idst(x_dst, type=1, norm='ortho')

    @staticmethod
    def getEigvals(params: Params, j: Union[int, np.ndarray]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Get the eigenvalues of the TST matrices

        :param params: Problem parameters
        :param j: Eigvals index
        :return: Eigenvalues i tuple (l_diag, l_off) for diagonal-eigenvalue and off-diagonal eigenvalue respectively"""
        if params.method.stencil == 5:
            # Eigval of central TST matrix with central band (-4) and off-diag bands 1
            # Eigval of off-diag TST matrix which is the identity
            l_diag = -4 + 2*np.cos(j*np.pi/params.N)
            l_off = 1
            return l_diag, l_off
        elif params.method.stencil == 9:
            l_diag = (-10 + 4*np.cos(j * np.pi / params.N)) / 3
            l_off = (2 + np.cos(j * np.pi / params.N)) / 3
            return l_diag, l_off
        else:
            raise NotImplementedError()


def _plot_method_convergence(N_arr: np.ndarray, method: Method, w_x: int = 9, w_y: int = 2, rel_error: bool = False):
    """Plot the convergence of the given method in the current figure"""

    error_arr = np.ones_like(N_arr) * np.inf
    time_arr = np.ones_like(N_arr) * np.inf
    if rel_error:
        rel_error_arr = np.ones_like(N_arr) * np.inf

    w_factor = -(w_x ** 2 + w_y ** 2) * np.pi ** 2

    # Set test parameters

    test_params = Params(method, None,
                         lambda x, y: w_factor**2 * np.sin(w_x*np.pi*x) * np.sin(w_y*np.pi*y))

    from Analytical import Analytical as An
    test_params.f = An.f

    for idx, N in enumerate(tqdm(N_arr)):
        test_params.N = N

        # Calculate the expected solution
        x = np.linspace(0, 1, test_params.N + 1)[1:-1]
        xx, yy = np.meshgrid(x, x)
        expected_sol = np.sin(w_x * np.pi * xx) * np.sin(w_y * np.pi * yy)
        expected_sol = An.u(xx, yy)

        # Solve problem
        res = solve(test_params)

        solution = res.sol

        error_arr[idx] = np.linalg.norm(solution - expected_sol) / np.sqrt(solution.size)
        time_arr[idx] = res.time

        # Get relative error, absolute error and time taken
        if rel_error:
            rel_acc = solution / expected_sol
            rel_error_arr[idx] = (np.mean(np.abs(rel_acc)))  # / (N+1)**2 - 1
            if np.max(rel_acc) / np.min(rel_acc) > 1.2:
                warnings.warn(f"Relative accuracy off for N={N}")
                rel_error_arr[idx] = 0

    plt.loglog(N_arr, error_arr, "x", label=f"Absolute error in $l_2$ norm for {test_params.method_str}")
    # plt.loglog(N_arr, time_arr, "o", markerfacecolor='None', label=f"Time taken in seconds for  {test_params.method_str}")
    if rel_error:
        plt.plot(N_arr, rel_error_arr, "^", label=f"Relative value for {test_params.method_str}")


def _test():
    """Run for nice plots :)"""
    # Plot error convergence
    w_x, w_y = 2, 3
    N_arr = np.unique(np.logspace(3, 9, base=2, num=20, dtype=int))[::-1]
    # N_arr = 2 ** np.arange(3, 10)
    # N_arr = np.arange(10, 200, 1)

    _plot_method_convergence(N_arr, Method.abdul_special_5, w_x, w_y, False)
    _plot_method_convergence(N_arr, Method.fast_2d_5, w_x, w_y, False)
    _plot_method_convergence(N_arr, Method.abdul_special_9, w_x, w_y, False)
    _plot_method_convergence(N_arr, Method.fast_2d_9, w_x, w_y, False)

    plt.grid()
    plt.ylabel(r"calc/expected   or   $||u_{an} - u_{num}||_2$,   or   time [s]")
    plt.xlabel("N")
    plt.legend()
    plt.title("Plot of error of numerical solution from analytical $u$ \n"
              fr"where $f = \nabla^4 u$ and $u = \sin({w_x}\pi x) \sin({w_y} \pi y)$")
    plt.tight_layout()
    plt.show()


def _test_raw():
    """Run for profiling, no plotting"""
    N_arr = (2 ** (np.ones(10, dtype=int) * 13))
    method = Method.fast_2d_9

    w_x, w_y = 2, 4
    w_factor = -(w_x ** 2 + w_y ** 2) * np.pi ** 2

    test_params = Params(method, None,
                         lambda x, y: w_factor ** 2 * np.sin(w_x * np.pi * x) * np.sin(w_y * np.pi * y))

    for N in tqdm(N_arr):
        test_params.N = N
        solve(test_params)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import warnings
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        tqdm = lambda x: x  # noqa

    _test()
    # _test_raw()  # Approx time:
