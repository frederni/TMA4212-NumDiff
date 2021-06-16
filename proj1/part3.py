import numpy as np
import scipy.sparse as spr
import scipy.sparse.linalg
from typing import Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import cm
from matplotlib.ticker import LinearLocator


def makeStencilMatrix(n: int, m: int, g: callable, f: callable = lambda x, y: np.zeros_like(x)) -> Tuple[spr.csr_matrix, np.ndarray]:
    """Create the matrix for solving the poisson eq. with the 5-point stencil method

    Creates matrix L for solving the poisson eq. (u_xx + u_yy = f) for the area (0<=x<=1, 0<=y<=1) with
    boundary conditions g at the border.
    The final poisson eq. will thus be L@U=f, where U are the points in the internal
    area with U[(x, y)] = U[x + n*y]

    :param n: Points in x-direction (internal)
    :param m: Points in y-direction (internal)
    :param g: Boundary conditions (Dirichlet) for all the intervals where x=0, y=0, x=1 or y=1. Should have signature f(x: np.ndarray, y: np.ndarray)
    :param f: Function parameter defined at all points (x, y). Default: f=0

    :return: Tuple of (L-matrix, f_vector) of size (m*n, m*n) and of size (m, 1),  for solving L@U=f where U[(x, y)] = U[x + n*y].
        Note that the f_vector is input f with added relevant boundary conditions g
    """
    h, k = 1/(n+1), 1/(m+1)  # Defining the step-size
    x_arr, y_arr = np.linspace(0, 1, n+2), np.linspace(0, 1, m+2)

    ## L-matrix
    # Define diagonals with offsets from the 0-diagonal
    diagonals = [-2*(1/h**2 + 1/k**2), 1/h**2, 1/h**2, 1/k**2, 1/k**2]
    offsets = [0, 1, -1, n, -n]

    # Inputting into L
    L = spr.diags(diagonals, offsets, shape=(n*m, n*m))

    ## f-vector
    # Adding values of f fcn. to output f-vector
    xx, yy = np.meshgrid(x_arr[1:-1], y_arr[1:-1])
    f_out = f(xx.flatten(), yy.flatten())

    # Adding boundary conditions:
    x_0 = g(x_arr[1:-1], np.zeros(n)) * (1/k**2)  # Along (x, 0)
    x_1 = g(x_arr[1:-1], np.ones(n)) * (1/k**2)   # Along (x, 1)
    y_0 = g(np.zeros(m), y_arr[1:-1]) * (1/h**2)  # Along (0, y)
    y_1 = g(np.ones(m), y_arr[1:-1]) * (1/h**2)   # Along (1, y)
    # Getting the indexes of the boundary conditions in the f-vector (and thus U-vector) and inserting values
    f_out[np.arange(n)] += x_0
    f_out[-np.arange(n)-1] += x_1
    f_out[np.arange(m)*n] += y_0
    f_out[-np.arange(m)*n - 1] += y_1

    return L.tocsc(), f_out


def solveLaplace(n, m):
    def boundaryCondition(x, y):
        boundary_values = np.zeros_like(x)
        mask = np.logical_and(np.logical_and(x != 0, x != 1), y != 0)
        boundary_values[mask] = np.sin(2*np.pi*x[mask])
        return boundary_values

    L, f = makeStencilMatrix(n, m, boundaryCondition)

    sol = spr.linalg.spsolve(L, f).reshape((m, n))

    return sol.reshape((m, n))


def analyticalSol(x, y):
    return 1/np.sinh(2*np.pi)*np.sinh(2*np.pi*y)*np.sin(2*np.pi*x)

def errorCalculation(loglist):
    errorlist = []
    for m in tqdm(loglist):
        x = np.linspace(0, 1, m)
        xx, yy = np.meshgrid(x, x)
        sol = analyticalSol(xx, yy)
        errorlist.append((1 / m) * np.linalg.norm(sol - solveLaplace(m, m)))
    return errorlist

def errorX(loglist):
    errorlist = []
    for i in tqdm(loglist):
        x = np.linspace(0, 1, i)
        y = np.linspace(0, 1, 499) #can be an arbitrary number, as long as it is constant for all iterations here
        xx, yy = np.meshgrid(x, y)
        sol = analyticalSol(xx, yy)
        errorlist.append(1/(np.sqrt(i)*np.sqrt(499)) * np.linalg.norm(sol - solveLaplace(i, 499)))
    return errorlist

def errorY(loglist):
    errorlist = []
    for i in tqdm(loglist):
        x = np.linspace(0, 1, 499) #can be an arbitrary number, as long as it's constant for all iterations here
        y = np.linspace(0, 1, i)
        xx, yy = np.meshgrid(x, y)
        sol = analyticalSol(xx, yy)
        errorlist.append(1/(np.sqrt(499)*np.sqrt(i)) * np.linalg.norm(sol - solveLaplace(499, i)))
    return errorlist

# Makes the different convergence
def convergencePlots(loglist, type):
    if type == "xy":
        # Convergence plot for discretizations in both directions
        plt.loglog(loglist, errorCalculation(loglist), "-", label="Error for discretization of both x- and y-directions")
        plt.title("Discretization in both x- and y-directions")
        plt.grid()
        plt.legend()
        triangle = plt.Polygon([[20, 4 * 10 ** -2], [20, 4 * 10 ** -3], [190, 4 * 10 ** -3]], fill=False, linewidth=1)
        plt.text(16, 1.2e-2, "-1")
        plt.text(50, 3e-3, "1")
        plt.gca().add_patch(triangle)
        plt.xlabel("Number of points in both directions")
        plt.ylabel("Discretization error")
        plt.savefig("task3_convergence_complete.pdf")
        plt.show()

    elif type == "x":
        # For x-direction
        plt.loglog(loglist, errorX(loglist), "-", label="Discretization error for x-direction", color="b")
        plt.title("Discretization in the x-direction")
        plt.grid()
        plt.legend()
        triangle = plt.Polygon([[20, 2 * 10 ** -2], [20, 3 * 10 ** -3], [150, 3 * 10 ** -3]], fill=False, linewidth=1)
        plt.text(16, 0.0071, "-1")
        plt.text(50, 2.5e-3, "1")
        plt.gca().add_patch(triangle)
        plt.xlabel("Number of discretization points in x-direction")
        plt.ylabel("Discretization error")
        plt.savefig("task3_convergence_x.pdf")
        plt.show()
    else:
        # for y-direction
        plt.loglog(loglist, errorY(loglist), "-", label="Discretization error for y-direction", color="b")
        plt.title("Discretization in the y-direction")
        plt.grid()
        plt.legend()
        triangle = plt.Polygon([[20, 4 * 10 ** -2], [20, 4 * 10 ** -3], [180, 4 * 10 ** -3]], fill=False, linewidth=1)
        plt.text(16, 1.2e-2, "-1")
        plt.text(50, 3e-3, "1")
        plt.gca().add_patch(triangle)
        plt.xlabel("Number of discretization points in y-direction")
        plt.ylabel("Discretization error")
        plt.savefig("task3_convergence_y.pdf")
        plt.show()


def _main():
    loglist = np.unique(np.logspace(1, 3, 11, dtype=int))

    convergencePlots(loglist, "x") # convergence plot for discretization of x-direction
    convergencePlots(loglist, "y") # convergence plot for discretization of y-direction
    convergencePlots(loglist, "xy") # convergence plot where both x- and y-directions are discretized

    # Makes 3d plot of the analytical solution
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    X = np.arange(0, 1, 1/499)
    XX, YY = np.meshgrid(X, X)
    ZZ = analyticalSol(XX, YY)
    plt.title("Analytical solution of the 2D Laplace equation")
    surf = ax.plot_surface(XX, YY, ZZ, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-1.0, 1.0)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.grid()
    plt.savefig("task3_analytical.pdf")
    plt.show()

if __name__ == "__main__":
    _main()

