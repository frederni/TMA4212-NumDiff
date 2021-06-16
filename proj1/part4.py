import numpy as np
import numpy.fft
from typing import Tuple, Union
import warnings

import matplotlib.pyplot as plt
import scipy.sparse as spr
import scipy.sparse.linalg

def getEigvals(k: float, h: float, M: int, method: str) -> np.ndarray:
    """Get the eigenvalues of the matrix C in the iteration scheme

    Are the elements along the diagonal d in C = (1/m) F^*_m diag(d) F_m,

    The eigvals are the dft of the first row-vector of the circulant matrix
    Golub, Gene H.; Van Loan, Charles F. (1996), "§4.7.7 Circulant Systems", Matrix Computations (3rd ed.), Johns Hopkins, ISBN 978-0-8018-5414-9

    :param k: Time step-size k
    :param h: Spatial step-size h
    :param M: Number of steps in spatial direction
    :param method: Solver method. Either "CN" or "FE" for Crank-Nicolson or forwards euler
    :return: Array of eigenvalues in order of increasing frequency in th fourier transform. Shape (M,)
    """
    assert method.lower() in ['cn', 'fe'], f"Method not defined: {method}"
    assert M >= 7, f"The cyclic matrix generation fails for M<7, M={M} given."

    r = k/h**3

    if method.lower() == 'cn':
        # Setting the first row-vector of the cyclic matrix b
        b = np.zeros(M)
        b[0] = 1
        b[1] = -(r/16) * (3 - 4*h**2 * (1 + np.pi**2))
        b[3] = r/16
        b[-1] = -b[1]
        b[-3] = -b[3]

        d_b = np.fft.fft(b)

        # Setting the first row-vector of the cyclic matrix a
        a = np.zeros(M)
        a[0] = 1
        a[1] = (r/16) * (3 - 4*h**2 * (1 + np.pi**2))
        a[3] = -r / 16
        a[-1] = -a[1]
        a[-3] = -a[3]

        d_a = np.fft.fft(a)

        # Return the result D_a^-1 D_b = D_c, which element-wise for the diagonal is d_b / d_a
        return d_b / d_a

    elif method.lower() == 'fe':
        # Setting the first row-vector of the cyclic matrix
        c = np.zeros(M)
        c[0] = 1
        c[1] = -(r/8) * (3 - 4*h**2 * (1 + np.pi**2))
        c[3] = r/8
        c[-1] = -c[1]
        c[-3] = -c[3]

        # The eigvals are the dft of the first row-vector of the circulant matrix
        # Golub, Gene H.; Van Loan, Charles F. (1996), "§4.7.7 Circulant Systems", Matrix Computations (3rd ed.), Johns Hopkins, ISBN 978-0-8018-5414-9
        ret = np.fft.fft(c)
        return ret


def solveKdV(initial_cond: callable, stop_time: float, N: int, M: int,
             method="CN", interval: Tuple[float] = (-1, 1),
             return_at_its: Union[np.ndarray, None, bool] = None) -> np.ndarray:
    """Solve the Korteveg-deVries equation returning the result at stop_time

    :param initial_cond: Initial condition. A function returning the initial condition on the interval. Should take np-arrays
    :param stop_time: End time for numerical simulation (from t=0)
    :param N: Time-steps from t=0 to (and including) t=stop_time
    :param M: Spatial steps from (and including) interval[0] to interval[1] (periodic bc.)
    :param method: Solver method. Either "CN" or "FE" for Crank-Nicolson or forwards euler. Default: "CN"
    :param interval: Interval for the periodic boundary condition. Default: (-1, 1)
    :param return_at_its: At what iteration steps to return the result. True->All its. Default: (N,)
    :return: Found solution to the problem at time t=stop_time. Shape (len(return_at_its), M). Defaults to (1, M)
    """
    if return_at_its is None:
        return_at_its = np.array((N,))
    elif return_at_its is True:
        return_at_its = np.arange(0, N)
    elif return_at_its is False:
        raise NotImplementedError("return_at_its is False. What would you think no returning would do??")

    assert method.lower() in ['cn', 'fe'], f"Method given ({method}) not implemented. Expected 'CN' or 'FE'."
    assert M >= 7, f"This method must have at least 7 points in the spatial direction. {M} was given."

    # Set up initial array U0
    x = np.linspace(*interval, num=M, endpoint=False)
    U0 = initial_cond(x)

    # Setting step-sizes
    k = stop_time / N
    h = (interval[1] - interval[0]) / M

    # Get the diagonal d in C = (1/m) F^*_m diag(d) F_m, which are the eigenvalues of C
    diag = getEigvals(k, h, M, method)

    # See report for formula
    res = np.zeros((return_at_its.size, diag.size))
    U0_fft = np.fft.fft(U0)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='overflow encountered in power')
        warnings.filterwarnings('ignore', message='invalid value encountered in power')
        warnings.filterwarnings('ignore', message='invalid value encountered in multiply')

        for idx, n in enumerate(return_at_its):
            res[idx] = np.real(np.fft.ifft(
                (diag ** n) * U0_fft
            ))  # Note that the "**" operator can work in O(log(n)) time

    # Return only the real part
    return np.real(res)


def makeConvergencePlot():
    """Make convergence plots for t=1 for different M-s"""
    # Set up what Ms to calculate for
    M_arr = np.concatenate((np.arange(8, 16), np.logspace(4, 13, 50, base=2, dtype=int)))

    # Set up initial conditions and parameters
    stop_time = 1
    timesteps_arr = 2**np.arange(5, 12)
    initial_cond, interval = lambda x_: np.sin(np.pi*x_), (-1, 1)
    end_cond = lambda x_: np.sin(np.pi*(x_ - stop_time))

    # Common plotting parameters
    plt.figure(figsize=(8, 6))
    colors = "viridis"
    cmap = plt.get_cmap(colors)

    # Run through both methods
    for method, name in [("FE", "Forwards Euler"), ("CN", "Crank-Nicolson")]:
        plt.subplot(1, 2, (1 if method == "FE" else 2))

        for t_idx, timesteps in enumerate(timesteps_arr):
            # Calculate errors for each M
            errors = np.empty_like(M_arr, dtype=float)
            for idx, M in enumerate(M_arr):
                x = np.linspace(*interval, M, endpoint=False)

                errors[idx] = np.linalg.norm(
                    end_cond(x) - solveKdV(initial_cond, stop_time, timesteps, M, method=method)[0]
                ) / np.sqrt(x.size) * (interval[1] - interval[0])

            plt.loglog(M_arr, errors, color=cmap(t_idx / len(timesteps_arr)), label=f"{method} error, N={timesteps}")

        if method == "FE":
            plt.ylim((1E0, 1E7))

        elif method == "CN":
            polygon_pts = 10 ** (np.array(((2, -1), (3, -1), (3, -3)), dtype=float) + np.array(((0.1, 0.1),)))
            plt.gca().add_patch(plt.Polygon(polygon_pts, fill=False, color="xkcd:red"))
            plt.text(*(10**np.array((2.6, -0.8))), "1")
            plt.text(*(10**np.array((3.2, -1.9))), "-2")

        plt.xlabel("Spatial points $M$ which is $\propto N^{1/6}$")
        plt.ylabel("Discrete $l_2$ norm of error from analytical solution")
        plt.title(f"Error of {name}\nscheme at t=1")
        plt.grid()
        plt.legend()

    plt.tight_layout()
    plt.savefig("figs/task4b_error_plot.pdf")
    plt.show()


def makeEulerConvergencePlot():
    """Make convergence plots for t=1 for different M-s"""
    # Set up what Ms to calculate for
    M_arr = np.concatenate((np.arange(8, 16), np.logspace(4, 9, 10, base=2, dtype=int)))

    # Set up initial conditions and parameters
    stop_time = 1
    mu_arr = np.logspace(0, 2, 5)
    initial_cond, interval = lambda x_: np.sin(np.pi*x_), (-1, 1)
    end_cond = lambda x_: np.sin(np.pi*(x_ - stop_time))

    # Common plotting parameters
    plt.figure(figsize=(8, 6))
    colors = "viridis"
    cmap = plt.get_cmap(colors)

    # Run through both methods
    for method, name in [("FE", "Forwards Euler"), ("CN", "Crank-Nicolson")]:
        plt.subplot(1, 2, (1 if method == "FE" else 2))

        for mu_idx, mu in enumerate(mu_arr):
            # Calculate errors for each M
            errors = np.empty_like(M_arr, dtype=float)
            for idx, M in enumerate(M_arr):
                x = np.linspace(*interval, M, endpoint=False)

                # mu = k / (2h**6)
                # k = 1/N, M = 2/h
                timesteps = 1 + int((M/2)**6 / (mu * 2))

                errors[idx] = np.linalg.norm(
                    end_cond(x) - solveKdV(initial_cond, stop_time, timesteps, M, method=method)[0]
                ) / np.sqrt(x.size) * (interval[1] - interval[0])

            plt.loglog(M_arr, errors, color=cmap(mu_idx / len(mu_arr)), label=f"{method} error, $\mu={mu:.2G}$")

        if method == "FE":
            plt.ylim((None, 1E2))
            polygon_pts = 10 ** (np.array(((1.5, 0), (2, 0), (2, -1)), dtype=float) + np.array(((0.1, 0.1),)))
            plt.gca().add_patch(plt.Polygon(polygon_pts, fill=False, color="xkcd:red"))
            plt.text(*(10**np.array((1.85, 0.2))), "1")
            plt.text(*(10**np.array((2.2, -0.4))), "-1")
            pass
            pass

        elif method == "CN":
            plt.ylim((None, 1E2))
            polygon_pts = 10 ** (np.array(((1.5, 0), (2, 0), (2, -1)), dtype=float) + np.array(((0.1, 0.1),)))
            plt.gca().add_patch(plt.Polygon(polygon_pts, fill=False, color="xkcd:red"))
            plt.text(*(10**np.array((1.85, 0.2))), "1")
            plt.text(*(10**np.array((2.2, -0.4))), "-1")
            pass

        plt.xlabel("Spatial points $M$ which is $\propto N^{1/6}$")
        plt.ylabel("Discrete $l_2$ norm of error from analytical solution")
        plt.title(f"Error of {name} scheme\nat t=1 with constant "r"$\mu = k/2h^6$")
        plt.grid()
        plt.legend()

    plt.tight_layout()
    plt.savefig("figs/task4b_error_plot_euler.pdf")
    plt.show()


def makeNormPlot():
    """Make a plot of the norm of the numerical solution over time"""
    # Set up initial conditions and parameters
    stop_time, space_steps = 1, 20
    timesteps_arr = 2**np.arange(9, 16)
    initial_cond, interval = lambda x_: np.sin(np.pi*x_), (-1, 1)
    true_norm = np.sqrt(2)

    # Common plotting parameters
    plt.figure(figsize=(8, 6))
    colors = "viridis"
    cmap = plt.get_cmap(colors)

    # Run through both methods
    for method, name in [("FE", "Forwards Euler"), ("CN", "Crank-Nicolson")]:
        plt.subplot(1, 2, (1 if method == "FE" else 2))

        for t_idx, timesteps in enumerate(timesteps_arr):
            x = np.linspace(*interval, space_steps, endpoint=False)

            norms = np.linalg.norm(
                solveKdV(initial_cond, stop_time, timesteps, space_steps, method=method, return_at_its=True),
                axis=1
            ) / np.sqrt(x.size) * (interval[1] - interval[0])

            plt.plot(np.linspace(0, stop_time, norms.size), norms,
                     color=cmap(t_idx / len(timesteps_arr)),
                     label=f"{method} error, $\mu=${(1/timesteps)/(2*(2/space_steps)**6):.2G}")

        plt.ylabel("Discrete $l_2$ norm of numerical solution")
        if method == "FE":
            plt.yscale('log')
            plt.ylim((1, 1E12))

        elif method == "CN":
            plt.ylim((1, 2))
            # plt.ylim(true_norm + 1E-12*np.array((-1, 1)))
            pass

        plt.xlabel("Time (t)")
        plt.title(f"Norm of solution to {name}\nscheme with M={space_steps}")
        plt.grid()
        plt.legend()

    plt.tight_layout()
    plt.savefig("figs/task4c_norm_plot.pdf")
    plt.show()


def main():
    # Task 4b convergence of methods when increasing M
    makeConvergencePlot()
    # Task 4b convergence of methods when increasing M with \mu = k/2h^6 constant
    makeEulerConvergencePlot()

    # Task 4c showing constant norm of methods (and convergence towards constant norm for the euler method)
    makeNormPlot()


if __name__ == "__main__":
    main()
