## GLOBAL IMPORTS ##
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spr
import scipy.sparse.linalg as spr_linalg #Specify submodule linalg in sparse
from scipy.interpolate import interp1d # For cont L2 norm
import scipy.integrate as integrate # For cont L2 norm
from functools import partial # For a simple selection menu
from collections import deque # Unused currently
from part5 import L2_norm, L2_norm_difference
from tqdm import tqdm

class BVP(object):
    """
    A class to hold the parameters of the BVP
    :param f: given function f(x)
    :param bc_a: Boundary condition at start
    :param bc_b: Boundary condition at end
    :param bcType: Tuple storing whether B.C. is Dirichlet ('d') or Neumann ('n')
    :param a: Starting point, ususally 0
    :param b: Ending point, ususally 1
    :param analytic: Analytic expression, lambda function
    """
    def __init__(self, f, bc_a, bc_b, bcType=('d','n'), a=0, b=1, analytic=None):
        self.f = f 
        self.bc_a = bc_a 
        self.bc_b = bc_b
        self.dirichlet_a = bcType[0] == 'd' # Bool if a is Dirichlet
        self.dirichlet_b = bcType[1] == 'd' # Bool if b is Dirichlet
        self.a = a 
        self.b = b 
        self.analytic = analytic 

class Poisson(object):
    """
    Class containing relevant info to calculate a numerical approximation of the Poisson equation
    :param BVP: BVP-object from class defined above.
    :param M: Number of points 
    :param AMR: Whether or not we wish to preform AMR on scheme 
    """
    def __init__(self, BVP, M, AMR=False):
        self.BVP = BVP
        self.M = M
        self.AMR = AMR

        # Derived properties
        self.XX_full = [i/(M+1) for i in range(M+1)] + [1.0] # Shape (M+2)
        self.XX = np.array(self.XX_full[1:]) # Shape (M+1), omits x0
        self.HH = np.array([self.XX[1]-self.XX[0]]*len(self.XX)) # Constant h across grid by default
        self.A, self.F = self.prepare_system()
        self.A_LU = None # Before calculating LU decomposition, replaced in `insert_BC()`
    
    def prepare_system(self) -> (np.ndarray, np.ndarray):
        """Create containers for difference scheme w/o boundary conditions"""
        F = np.array([self.BVP.f(x) for x in self.XX])

        # This segment produces the [1, -2, 1] tridiagonal shape
        A = np.diag(-2.0*np.ones(self.XX.shape[0])) # -2 along diagonal (M+1)x(M+1) matrix
        #debug: self.M + 1 -> self.XX.shape[0]
        np.fill_diagonal(A[1:], 1) # Fills offsets with 1
        np.fill_diagonal(A[:,1:], 1)
        return A, F
    
    def insert_BC(self):
        """Modify arrays based on the given Boundary Conditions"""
        h0, hend = self.HH[0], self.HH[-1]
        
        # Modify A and F according to B.C.
        if self.BVP.dirichlet_a and not self.BVP.dirichlet_b:
            # First case (A)
            self.A[-1, -3] = -1/2 * hend
            self.A[-1, -2] = 2 * hend
            self.A[-1, -1] = -3/2 * hend
            self.F[-1] = self.BVP.bc_b
        
        elif self.BVP.dirichlet_a and self.BVP.dirichlet_b:
            # Second case (B) and (D)
            self.F[-1] = self.F[-1] - self.BVP.bc_b/(hend*hend)
        
        # Modifications applying to all B.C.
        self.F[0] = self.F[0] - self.BVP.bc_a/(h0*h0)
        if not self.AMR: # We don't divide by h^2 if we're using Poisson 3/4 stencil
            self.A = 1/(self.HH**2) * self.A

        # Convert to sparse scipy matrix to optimize efficiency
        A_spr = spr.csc_matrix(self.A)
        try:
            self.A_LU = spr_linalg.splu(A_spr) # with LU decomp.
        except RuntimeError:
            self.A_LU = spr_linalg.spilu(A_spr) # incomplete LU decomp.

    

    def liu_stencil(self, order:int=2) -> list:
        """
        AMR / Poisson 4/3-stencil from Liu1997.
        :param second_order: Bool determining whether or not to use second order stencil
        :return: approximated and analytical solution using stencils
        """
        n = self.XX.shape[0]
        # Start off with central difference, need to divide by h^2
        self.A[0, 0] /= self.HH[0]**2
        self.A[0, 1] /= self.HH[0]**2
        self.A[0, 2] /= self.HH[0]**2

        for i in range(1, n-1):
            d_i = np.zeros(4) # [0, i+1, i-2, i-1]
            d_i[-2] = np.abs( self.XX[i] - self.XX[i-2] )
            d_i[-1] = np.abs( self.XX[i] - self.XX[i-1] )
            d_i[+1] = np.abs( self.XX[i+1] - self.XX[i] )
            if order == 2:
                # Equation (3) in Liu1995
                alpha = (2*(d_i[+1] - d_i[-1])
                    / (d_i[-2] * (d_i[-2] + d_i[+1]) * (d_i[-2] - d_i[-1])))

                beta = (2*(d_i[-2] - d_i[+1])
                    / (d_i[-1] * (d_i[-2] - d_i[-1]) * (d_i[-1] + d_i[+1])))

                gamma = (2* (d_i[-2] + d_i[-1])
                    / (d_i[+1] * (d_i[-1] + d_i[+1]) * (d_i[-2] + d_i[+1])))
            else:
                # Three-point central-difference scheme from Table 1 in Liu1995
                delta_x_i  = 1/2 * ( d_i[-1] + d_i[+1] ) # Gridblock size
                alpha = 0

                beta = (1
                    / (d_i[-1]*delta_x_i) )
                
                gamma = (1
                    / (d_i[+1]*delta_x_i) )
            
            # Same prodecure for first and second order
            self.A[i, i-2] = alpha
            self.A[i, i-1] = beta
            self.A[i, i] = -(alpha+beta+gamma)
            self.A[i, i+1] = gamma

        self.insert_BC()
        return self.get_solution() 


    def get_solution(self) -> (np.ndarray, np.ndarray):
        """Helper to get exact and approximate solution based on A and F"""
        assert self.A_LU is not None, "Cannot calculate solution before inserting B.C."
        U = self.A_LU.solve(self.F)
        uexact = np.array([self.BVP.analytic(x) for x in self.XX])
        return U, uexact
        
    def get_errors(self, U, uexact, get_cont_only:bool=False) -> (float, float):
        """
        Helper to find discrete/continuous error
        :param AMRpts: Index set of points to preform AMR on (unless it's None)
        :return: Relative error with discrete norm and continuous norm
        """
        err_cont = cont_L2_norm(uexact-U, self.XX)/cont_L2_norm(uexact, self.XX)

        if get_cont_only:
            # When doing AMR we don't need discrete
            return err_cont
        
        err_discrete = np.linalg.norm(uexact-U)/np.linalg.norm(uexact)
        return err_discrete, err_cont

    def do_AMR_old(self, U, uexact, order, refine_method:str='max') -> (float, float):
        # I'm proud of the implementation even if it's wrong so i dont wanna delete it ok
        """
        We get the approximation and exact solution from UMR and find indices where we need to refine h
        Then we divide h by 2 in appropriate indices and solve
        """
        U_spline = interp1d(self.XX, U)
        deviations = np.array([np.abs(U_spline(x)-self.BVP.analytic(x)) for x in self.XX])

        if refine_method == 'max':
            refine_criteria = 0.7 * np.max(deviations)
        else:
            refine_criteria = 1.0 * np.mean(deviations)
        
        def get_refinement(ind, dev_lst, side='left', current=None): # TODO Maybe we have to take in self
            if current is None:
                current = self.XX[ind]
            if side=='left':
                if ind == 0:
                    insert_left = 0.5 * (self.BVP.a + current)
                else:
                    insert_left = 0.5 * (self.XX[ind-1] + current)
                return insert_left
            elif side=='right':
                if ind == len(dev_lst)-1:
                    insert_right = 0.5 * (self.BVP.b + current)
                else:
                    insert_right = 0.5 * (current + self.XX[ind+1])
                return insert_right
        
        for ind, dev in enumerate(deviations):
            if dev > refine_criteria:
                left_list = [get_refinement(ind, deviations, side='left')]
                left_err = np.abs(U_spline(left_list[0])-self.BVP.analytic(left_list[0])) # First error
                
                count = 0
                while left_err > refine_criteria:
                    left_list.append(get_refinement(ind, deviations, side='left', current=left_list[count]))
                    count += 1
                    left_err = np.abs(U_spline(left_list[count]) - self.BVP.analytic(left_list[count]))
                
                right_list = [get_refinement(ind, deviations, side='right')]
                right_err = np.abs(U_spline(right_list[0])-self.BVP.analytic(right_list[0]))

                count = 0
                while right_err > refine_criteria:
                    right_list.append(get_refinement(ind, deviations, side='right', current=right_list[count]))
                    count += 1
                    right_err = np.abs(U_spline(right_list[count]) - self.BVP.analytic(right_list[count]))
                
                # say XX [4,8,12,16], ind=1. left_list could be [6, 5, 5.5, ...]
                # thus we need to reverse before insertion
                left_list.reverse()
                right_list.reverse()

                self.XX = np.insert(self.XX, ind+1, right_list) # This doesn't give us IndexError, based NumPy <3
                self.XX = np.insert(self.XX, ind, left_list)
        
        # Reinitiate object so we can solve again
        self.A, self.F = self.prepare_system()

        return self.liu_stencil(second_order=order) # Using liu stencil then 


    def do_AMR(self, U, uexact, order, refine_method:str='max') -> (float, float):
        """
        We get the approximation and exact solution from UMR and find indices where we need to refine h
        Then we divide h by 2 in appropriate indices and solve
        """
        U_spline = interp1d(self.XX, U)
        # deviations = np.array([np.abs(U_spline(x)-self.BVP.analytic(x)) for x in self.XX]) #old
        self.BVP.exact = self.BVP.analytic
        deviations = np.array([L2_norm_difference(self.BVP, U_spline, subspace=(self.XX[i], self.XX[i+1])) for i in range(len(self.XX)-1)])

        if refine_method == 'max':
            refine_criteria = 0.7 * np.max(deviations)
        else:
            refine_criteria = 1.0 * np.mean(deviations)
        
        # def get_refinement(ind, dev_lst, side='left'):
        #     if side=='left':
        #         if ind == 0:
        #             insert_left = 0.5 * (self.BVP.a + self.XX[ind])
        #         else:
        #             insert_left = 0.5 * (self.XX[ind-1] + self.XX[ind])
        #         return insert_left
        #     elif side=='right':
        #         if ind == len(dev_lst)-1:
        #             insert_right = 0.5 * (self.BVP.b + self.XX[ind])
        #         else:
        #             insert_right = 0.5 * (self.XX[ind] + self.XX[ind+1])
        #         return insert_right
        insert_values = []
        # exclude_values = []
        for ind, dev in enumerate(deviations):
            if dev > refine_criteria:
                # insert_values.append(get_refinement(ind, deviations, side='left'))
                # insert_values.append(get_refinement(ind, deviations, side='right'))
                # exclude_values.append(self.XX[ind])
                insert_values.append(0.5*(self.XX[ind] + self.XX[ind+1]))
        # Quick explanation of one-liner below. We first remove duplicates of refinement point and concatenate it 
        #   with the original grid, excluding those from exclude_values. Then we finally sort the values
        #   so that XX[i-1] < XX[i] \forall i != 0
        # self.XX = np.sort(np.concatenate((self.XX[~np.in1d(self.XX, exclude_values)], np.unique(insert_values))))
        self.XX = np.sort(np.concatenate((self.XX, insert_values)))
        self.HH = np.diff(self.XX)
        
        # Reinitiate object so we can solve again
        self.A, self.F = self.prepare_system()
        return self.liu_stencil(order) 


def cont_L2_norm(V:np.ndarray, XX, interpKind:str='linear', subspace=None) -> float:
    """
    Returns continuous L2 norm with interpolation and integration
    :param V: Vector of approximated values 
    :param BVP: BVP-class containing integration limits, though usually 0 and 1
    :param XX: linear space form PoissonSystem
    :param interpKind: Type of interpolation, passed directly into interp1d. `linear` or `cubic`
    :return: Continuous L2-norm as float
    """
    if subspace is not None:
        try:
            V = V[subspace[0]:subspace[1]+1]
            XX = XX[subspace[0]:subspace[1]+1]
        except IndexError:
            V = V[-2:]
            XX = XX[-2:]
    f = interp1d(XX, V, kind=interpKind)
    # f is the interpolation for the values V based on some grid XX
    g = lambda x : f(x)**2
    return np.sqrt(integrate.quad(g, XX[0], XX[-1])[0])

def poisson_plot(p:Poisson, M:int, U:np.ndarray):
    xls = np.linspace(p.BVP.a, p.BVP.b, num=len(p.XX))
    title_bc = "Dirichlet-Neumann" if (p.BVP.dirichlet_a and not p.BVP.dirichlet_b) else ("pure Dirichlet" if (p.BVP.dirichlet_a and p.BVP.dirichlet_b) else "")
    plt.plot(xls, U, label=f"Approximation with M={M}")
    plt.plot(xls, p.BVP.analytic(xls), label="Exact")
    plt.title(r"$u_{xx}=f(x)$ with " + title_bc)
    plt.xlabel("$x$")
    plt.ylabel("$u(x)$")
    plt.legend()
    plt.grid()
    plt.savefig(f"figs/poisson_{M}_pts_{title_bc}.pdf")
    plt.show()

def rel_error_plot(errD:list, errC:list, Mlst:list, task:str):
    """
    Plots relative discrete/continuous error over different number of points

    :param errD: Python list of relative discrete errors as y axis
    :param errC: Python list of relative continuous errors as y axis
    :param Mlst: List of "M" number of points corresponding to error
    :param task: Either '1a' or '1b'
    """
    plt.loglog(Mlst, errD, '*', label="Discrete error")
    plt.loglog(Mlst, errC, label="Continuous error")
    #Draw convergence triangle
    if task == '1a':
        # Draw order 2
        triangle = plt.Polygon([[20,10**-3], [20,1.2*10**-5], [120,1.2*10**-5]], fill=False, linewidth=1)
        plt.text(15, 7e-5, "-2")
        plt.text(42, 5e-6, "1")
    else:
        triangle = plt.Polygon([[20,10**-3], [20,1.2*10**-4], [120,1.2*10**-4]], fill=False, linewidth=1)
        plt.text(14, 2.5e-4, "-1")
        plt.text(50, 7e-5, "1")
    plt.gca().add_patch(triangle)

    plt.xlabel("Number of points M")
    plt.ylabel("Relative error")
    plt.title(f"Case {task[-1]}): Convergence plot with \ncontinuous and discrete relative error norm")
    plt.grid()
    plt.legend()
    plt.savefig(f"figs/{task}_rel_error.pdf")
    plt.show()


def rel_err_plot_AMR(UMR, AMR_max, AMR_avg, Mlst, order):
    """
    Plots the error when using AMR, only plotting relative continuous L2-error
    :param UMR: list of approximations using UMR for each `M`
    :param AMR_max: list of approximations using AMR, refining when error > alpha * max_error
    :param AMR_avg: list of approximations using AMR, refining when error > alpha * avg_error
    :param Mlst: List of "M" number of points corresponding to error
    :param order: AMR-order, either 1 (using Poisson 3 stencil) or 2 (using Poisson 4 stencil)
    """

    plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14, 
    "font.family": "serif",
    "font.sans-serif": ["CMU Serif Roman"]})

    plt.loglog(Mlst, UMR, '*', label="UMR (discrete norm)")
    plt.loglog(Mlst, AMR_max, label="AMR with max criteria")
    plt.loglog(Mlst, AMR_avg, label="AMR with avg. criteria") 
    plt.xlabel("Number of points M")
    plt.ylabel(r"Relative error $e^r_{(\cdot)}$")
    plt.title(f"Poission equation using \nUMR and AMR of order {order}")

    #Draw convergence triangle
    # Draw order 2
    triangle = plt.Polygon([[12,10**-3], [12,1.2*10**-5], [120,1.2*10**-5]], fill=False, linewidth=1)
    plt.text(8, 7e-5, "-2")
    plt.text(30, 2e-6, "1")
    plt.gca().add_patch(triangle)


    plt.grid()
    plt.legend()
    plt.savefig(f"figs/AMR_error_order_{order}.pdf")
    plt.show()

def analyzePoisson(BVP:BVP, AMR:bool=False, export:bool=False, AMR_order=2, task:str=''):
    """
    Testing function to check convergence of methods in Poisson-class.
    :param BVP: BVP-object storing necessary information
    :param AMR: Bool determining wether or not we wish to preform AMR
    """
    M_lst = [2**n for n in range(2,13)] # unique logspace [4, 8192]
    # Initiate containers
    if AMR:
        list_UMR, list_AMR_max, list_AMR_avg = [], [], []
    else:
        err_lst_discr, err_lst_cont = [], []
    for m in tqdm(M_lst):
        p = Poisson(BVP, m, AMR=AMR)
        if AMR:
            U, uex = p.liu_stencil(order=2)
            err_UMR, _ = p.get_errors(U, uex)
            for N in range(5):
                U, uex = p.do_AMR(U, uex, AMR_order, refine_method='max')
            err_AMR_max = p.get_errors(U, uex, get_cont_only=True)
            # Reinitialize
            p.__init__(BVP, m, AMR=AMR)
            U, uex = p.liu_stencil(order=2)
            for N in range(5):
                U, uex = p.do_AMR(U, uex, AMR_order, refine_method='avg')
            err_AMR_avg = p.get_errors(U, uex, get_cont_only=True)
            
            list_UMR.append(err_UMR)
            list_AMR_max.append(err_AMR_max)
            list_AMR_avg.append(err_AMR_avg)

        else:
            p.insert_BC()
            U, uex = p.get_solution()
            if m == M_lst[0] or m == M_lst[-1]:
                poisson_plot(p, m, U)
            err_d_i, err_c_i = p.get_errors(U, uex)
            err_lst_discr.append(err_d_i)
            err_lst_cont.append(err_c_i)

    # Plot natively with matplotlib
    if AMR:
        rel_err_plot_AMR(list_UMR, list_AMR_max, list_AMR_avg, M_lst, order=AMR_order)
    else:
        rel_error_plot(err_lst_discr, err_lst_cont, M_lst, task)



def main():
    
    def f(x): # This applies for all of task 1, except d
        return np.cos(2*np.pi*x) + x

    BVPa = BVP(f=f, bc_a=0, bc_b=0, a=0, b=1,
        analytic=lambda x: -1/(4*np.pi*np.pi)*np.cos(2*np.pi*x)+1/6 * x**3 + -1/2*x + 1/(4*np.pi*np.pi)
        )
    BVPb = BVP(
        f=f, bc_a=1, bc_b=1, bcType=('d', 'd'), a=0, b=1,
        analytic= lambda x: -1/(4*np.pi*np.pi)*np.cos(2*np.pi*x)+1/6 * x**3 - 1/6*x + 1+1/(4*np.pi*np.pi)
        ) 

    # Task 1d
    def task1d():
        eps = 1/1000
        def u_d(x): return np.exp(-1/eps * np.square(x - 1/2))
        def f_d(x): return u_d(x) * (4*x**2 - 4*x - 2*eps + 1) /eps**2
        BVPd = BVP(
            f = f_d, bc_a = 0, bc_b = 0, bcType=('d', 'd'),
            a=0, b=1, analytic= u_d)
        # `bc_a` and `bc_b` are actually ~ 2e-109, i.e. practically 0.
        
        analyzePoisson(BVPd, AMR=True, AMR_order=2)
        analyzePoisson(BVPd, AMR=True, AMR_order=1)
    
    def select_menu():
        """Selection menu so user may choose tasks easily"""
        menu_selection = ''
        menu = {
            "1": partial(analyzePoisson, BVPa, task='1a'),
            "2": partial(analyzePoisson, BVPb, task='1b'),
            "3": partial(task1d),
            "q": partial(print, "")
        }
        while menu_selection != 'q':
            print("Selection menu", "1\tTask 1a)", "2\tTask 1b)", "3\tTask 1d)", "q\tExit", sep='\n')
            menu_selection = input("Choose task: ").lower()
            try:
                menu[menu_selection]()
            except KeyError:
                print("Invalid selection")
    
    select_menu()
    
    


if __name__ == "__main__":
    main()
