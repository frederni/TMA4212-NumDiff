import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from typing import Callable
import scipy.sparse as spr
import scipy.sparse.linalg as spr_linalg
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    pass #noqa

class Parameters():
    def __init__(self, task:str) -> None:
        """Constructor for Parameters, assigning all necessary information

        :param task: String signifying which task/case we're working on
        :type task: str
        """
        self.task = task 
        legal_tasks = 'bcde'
        assert task in legal_tasks and len(task)==1, "Invalid task number."
        if task=='b' or task=='e': self.a = 0 
        else: self.a = -1
        self.b = 1
        
        # Function value and inital values
        values = {
            'function': [
            lambda x: -2,
            lambda x: -(40_000 * x**2 - 200)*np.exp(-100*x**2),
            lambda x: -(4_000_000 * x**2 - 2_000)*np.exp(-1_000*x**2),
            lambda x: 2/9 * x**(-4/3)
            ],
            'initials': [
            (0, 1),
            (np.exp(-100), np.exp(-100)),
            (np.exp(-1_000), np.exp(-1_000)),
            (0, 1)
            ],
            'analytical': [
            lambda x : x**2,
            lambda x : np.exp(-100*x**2),
            lambda x : np.exp(-1_000*x**2),
            lambda x : x**(2/3)
            ],
            'analytical_txt': [
                '$x^2$', '$\exp(-100 x^2)$', '$\exp(-1 000 x^2)$', '$x^{2/3}$'
            ]
        }

        selection = legal_tasks.index(task)
        self.f = values['function'][selection]
        self.d_i = values['initials'][selection]
        self.exact = values['analytical'][selection]
        self.plot_text = values['analytical_txt'][selection]

def L2_norm(func:Callable, domain_a, domain_b): return np.sqrt( integrate.quad( lambda x : func(x)**2, domain_a, domain_b)[0])
def L2_norm_difference(param:Parameters, approx:Callable, subspace=None):
    domain = (param.a, param.b) if subspace is None else subspace
    return np.sqrt( integrate.quad( lambda x : (param.exact(x)-approx(x))**2, *domain, limit=100)[0])

def get_L2_error(param:Parameters, U:Callable) -> float:
    """Finds relative L2 error from FEM-approximation and analytical solution

    :param param: Parameters-class containing analytical expression and its domain
    :type param: Parameters 
    :param U: Functional from FEM
    :type U: Callable
    :return: || u - U ||_2 / || u ||_2, i.e. relative error in L2-norm
    :rtype: float
    """
    difference_norm = L2_norm_difference(param, U) #np.sqrt( integrate.quad(lambda x: (param.exact(x)-U(x))**2, param.a, param.b)[0] )
    exact_norm = L2_norm(param.exact, param.a, param.b)
    return difference_norm / exact_norm


def plot_error(params:Parameters, N_list:np.ndarray, error_list:list, AFEM_avg, AFEM_max, useTeX:bool=True) -> None:
    if useTeX:
        plt.rcParams.update({
        "text.usetex": True,
        "font.size": 14, 
        "font.family": "serif",
        "font.sans-serif": ["CMU Serif Roman"]})

    N_list = np.array(N_list, dtype=np.float64)
    plt.loglog(N_list, error_list, '-o', label="FEM error with uniform grid")
    # plt.loglog(N_list, error_list[0]/N_list[0]*(N_list)**(-2), '--', label=r"$\mathcal{O}(Ndof^2)$")
    plt.loglog(N_list, np.array((error_list[0]*N_list[0]**2)/N_list**2), '--', label=r"$\mathcal{O}(Ndof^2)$")
    

    plt.loglog(*AFEM_avg, label=r"AFEM error, $\alpha=1.0$ average")
    plt.loglog(*AFEM_max, label=r"AFEM error, $\alpha=0.7$ max")
    

    plt.title(f"Error plot for case {params.task})")
    plt.ylabel(r"Relative $L_2$-norm error")
    plt.xlabel(r"Ndof, Degrees of freedom")
    plt.grid()
    plt.legend(loc=3)
    plt.savefig(f"figs/5{params.task}-AFEM.pdf")
    plt.show()

def plot_ux(params:Parameters, approx:Callable, N:int, label='', useTeX:bool=True):
    if useTeX:
        plt.rcParams.update({
        "text.usetex": True,
        "font.size": 14, 
        "font.family": "serif",
        "font.sans-serif": ["CMU Serif Roman"]})
    x = np.linspace(params.a, params.b, num=200)
    FEM_approx = [approx(x_i) for x_i in x]
    plt.plot(x, FEM_approx, label="FEM approx" + label)
    plt.plot(x, params.exact(x), label="Analytical solution")
    plt.title(f"u(x)={params.plot_text} with {N} elements")
    plt.legend(loc=3)
    plt.savefig(f"figs/5{params.task}-approx-N{N}.pdf")
    plt.show()
# <<<<< Helpers

def phi_first(i, x, XX):
    """Basis function part 1"""
    return (x - XX[i - 1]) / (XX[i] - XX[i - 1])

def phi_second(i, x, XX):
    """Basis function part 2"""
    return (XX[i + 1] - x) / (XX[i + 1] - XX[i])

def FEM(param:Parameters, XX:np.ndarray):
    """Finite element method

    :param param: Parameter-object storing all info about task number, analytical solution etc
    :type param: Parameters
    :param XX: Grid
    :type XX: np.ndarray
    :return: Returns the same grid (to make AFEM easier), approximated u and the corresponding functional
    :rtype: (np.ndarray, np.ndarray, Callable)
    """
    N = XX.shape[0] - 1
    num_pts = XX.shape[0]
    HH = 1 / (XX[1:] - XX[:-1])

    A_diag = np.append(HH, 0) + np.append(0, HH)
    A = spr.diags( (-HH, A_diag, -HH), (-1, 0, 1), shape=(num_pts, num_pts), format="lil", dtype=np.float64 )

    F = np.zeros(num_pts)

    for i in range(1, N - 1):
        F[i] = (integrate.quad(lambda x: phi_first(i, x, XX) * param.f(x), XX[i - 1], XX[i])[0]
        + integrate.quad(lambda x: phi_second(i, x, XX) * param.f(x), XX[i], XX[i + 1])[0])

    u = np.zeros(num_pts)
    u[0] = param.d_i[0]
    u[-1] = param.d_i[1]
    F = F - A.tocsr().dot(u) # F-A\tilde{u}
    # Exclude first and last entry
    F = F[1:-1]
    A = A[1:-1, 1:-1]
    
    u_bar = spr_linalg.spsolve(A.tocsc(), F)
    u[1:-1] = u_bar # Impose B.C.

    return XX, u, interp1d(XX, u)

def AFEM(param: Parameters, tol: float, N: int = 20, strategy: str = 'avg'):
    """Adaptive FEM

    :param param: Parameter object
    :type param: Parameters
    :param tol: User-given tolerance
    :type tol: float
    :param N: Number of starting Ndofs, defaults to 20
    :type N: int, optional
    :param strategy: refinement strategy, `avg` or `max, defaults to 'avg'
    :type strategy: str, optional
    :return: Same as `FEM` but also returns the plotting list
    :rtype: (np.ndarray, np.ndarray, Callable, list)
    """
    x = np.linspace(param.a, param.b, N + 1)
    plotting = [[], []]

    while len(x)-1 < 2048:
        to_insert = []
        x, u, u_func = FEM(param, x)
        plotting[0].append(len(x)-1) #ndof
        plotting[1].append(get_L2_error(param, u_func))

        error = []
        for i in range(len(x)-1):
            error.append(
                np.sqrt(
                    integrate.quad(
                        lambda x : (param.exact(x)-u_func(x))**2, x[i], x[i+1]
                    )[0]
                )
            )
        
        for i, err_i in enumerate(error):
            criteria = 1.0 * np.mean(error) if strategy=='avg' else 0.7 *  np.max(error)
            if err_i > criteria:
                to_insert.append( (x[i+1] + x[i]) / 2 )
        x = np.sort( np.concatenate((x, to_insert)) )
        
    return x, u, u_func, plotting

def test_FEM(task: str, skip_func_plot: bool=False):
    param = Parameters(task)
    N_list = np.logspace(3,11,num=9, base=2, dtype=np.int32)
    errors = []
    for N in N_list:
        XX = np.linspace(param.a, param.b, N+1)
        XX, u, u_func = FEM(param, XX)
        if N < 100 and not skip_func_plot: plot_ux(param, u_func, N)
        errors.append(get_L2_error(param, u_func))
    return param, N_list, errors

def test_AFEM(task: str, skip_func_plot: bool=False):
    param = Parameters(task)
    x, u, u_func, AFEM_max = AFEM(param, 1e-8, strategy='max')
    x, u, u_func, AFEM_avg = AFEM(param, 1e-8, strategy='avg')
    return AFEM_avg, AFEM_max


def main():
    tasks = 'bcde'
    for task in tasks:
        plot_error(*test_FEM(task, skip_func_plot=False), *test_AFEM(task, skip_func_plot=True))

if __name__ == '__main__':
    main()