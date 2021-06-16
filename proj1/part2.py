#!/Users/Bendik/opt/anaconda3/bin/python
#from part1 import discr_l2_norm, relErrorPlot
import utilities as utils 
import numpy as np 
import numpy.linalg as lin 
import matplotlib.pyplot as plt 
from timeit import default_timer as timer 
from numba import njit
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation
from matplotlib import cm

################################################################################
########     2 a)                                                       ########
################################################################################

def error(U_ref, U_approx):
    ulen_star = len(U_ref)
    ulen = len(U_approx)
    k = int((ulen_star-1)/(ulen-1))

    # The slice is a view, so constant time
    U_ref_adjusted = U_ref[::k]
    return lin.norm(U_ref_adjusted - U_approx) / lin.norm(U_ref_adjusted)

def make_convergence_plot(errors, lab, ax, Ms = [2**i + 1 for i in range(2, 9)]):
    ax.plot(Ms, errors, 'o-', label=lab)
    ax.set_yscale('log')
    ax.set_xscale('log')
    
def add_triangle(points, ax):
    triangle = plt.Polygon(points, fill=False, linewidth=0.6)
    ax.add_patch(triangle)

def make_errors(init_fnc, DE_method, **kwargs):
    '''
    Make a wrapper function 'DE_method' for the method that only takes the points (here M) as an argument,
    alternatively one that takes M and some kwargs. If you have a reference solution function,
    pass it in as ref_fnc kwarg. Otherwise one will be made from the DE_method. You can also pass in M_star
    or the list of Ms to evaluate. Be careful to make sure the points match up!
    '''
    M_star = kwargs.get('M_star', 2**10 + 1)
    Ms = [2**i + 1 for i in range(2, 9)]
    Ms = kwargs.get('Ms', Ms)
    errors = np.zeros(len(Ms))

    if 'ref_fnc' in kwargs:
        ut_ref = kwargs['ref_fnc'](np.linspace(0,1,M_star))
    else:
        print('Computing reference solution with {} nodes...'.format(M_star))
        start = timer()
        ut_ref = DE_method(init_fnc(np.linspace(0,1,M_star)))
        end = timer()
        minutes = (end-start)//60
        print('Computed reference solution in', minutes, 'minutes,', (end-start)%60, 'seconds.\n')    
    
    for i in range(len(Ms)):
        u0_M = init_fnc( np.linspace(0,1, Ms[i]) )
        start = timer()
        ut_M = DE_method(u0_M)
        end = timer()
        print('Computed approximation with', Ms[i], 'nodes in', end-start, 'seconds.')
        errors[i] = error(ut_ref, ut_M)
    print('\n')
    return errors

def plot_h_refinement(save=False):
    N = 1000
    t = 0.1
    wrap1 = lambda x: utils.CN_o1(x, t, N=N)
    wrap2 = lambda x: utils.CN_o2(x, t, N=N)

    print('Order 1 approximation:\n')
    errors_o1 = make_errors(utils.f, wrap1)
    print('Order 2 approximation:\n')
    errors_o2 = make_errors(utils.f, wrap2)

    fig, ax = plt.subplots(1,1)
    ax.set_axisbelow(True)
    ax.grid()
    make_convergence_plot(errors_o1, "First order BC", ax)
    make_convergence_plot(errors_o2, "Second order BC", ax)
    add_triangle([[10,0.1], [100,0.1], [100,0.01]], ax)
    add_triangle([[10,10**-3], [10,10**-5], [100,10**-5]], ax)

    ax.text(30, 0.2, "1")
    ax.text(120, 0.03, "-1")
    ax.text(30, 5e-6, "1")
    ax.text(7.5, 7e-5, "-2")

    ax.legend()
    ax.set_xlabel(r'$M$')
    ax.set_ylabel('Relative error')
    ax.set_ylim(10**(-7), 1)
    if save:
        plt.savefig("2a_h_refinement.pdf")
    else:
        plt.show()

def plot_k_refinement():
    M = 2**13+1
    t = 0.1
    N_star = 2**13
    Ns = [2**i for i in range(5, 12)]
    u0 = utils.f(np.linspace(0,1,M))

    print('Computing reference solution with {} timesteps...'.format(N_star))
    start = timer()
    ut_ref = utils.CN_o2(u0, t, N=N_star)
    end = timer()
    minutes = (end-start)//60
    print('Computed reference solution in', minutes, 'minutes,', (end-start)%60, 'seconds.') 
    
    errors_o1 = np.zeros(len(Ns))
    errors_o2 = np.zeros(len(Ns))

    for i in range(2):
        print('\nOrder {} approximation'.format(i+1))
        if i==0:
            method = utils.CN_o1
            errors = errors_o1
        else:
            method = utils.CN_o2
            errors = errors_o2
        for i, N in enumerate(Ns):
            start = timer()
            ut_K = method(u0, t, N=N)
            end = timer()
            print('Computed approximation with {} timesteps in {} seconds.'.format(N, end-start))
            errors[i] = error(ut_ref, ut_K)
    
    # Plotting
    fig, ax = plt.subplots(1,1)
    make_convergence_plot(errors_o1, "First order BC", ax, Ns)
    make_convergence_plot(errors_o2, "Second order BC", ax, Ns)
    
    add_triangle([[60,0.003], [60,0.0003], [600,0.0003]], ax)
 
    ax.text(43, 0.001, "-1")
    ax.text(150, 0.0002, "1")
    
    ax.legend()
    ax.set_xlabel(r'$N$')
    ax.set_ylabel('Relative error')
    #ax.set_ylim(10**(-7), 1)
    plt.show()



################################################################################
########     2 b)                                                       ########
################################################################################

# Manufactured solution is u(x,t) = e^(-4pi^2 t) sin(2 pi x - pi/2)
# u(x,0) = sin(2 pi x - pi/2)
# u'(0,t) = u'(1,t) = 0

def u(x, t):
    return np.exp(-4*np.pi**2 * t) * np.sin(2*np.pi*x - np.pi/2)

def plot_u(save=False):
    M,N = 200,300
    X,T = np.linspace(0,1,M), np.linspace(0,0.2,N)
    X,T = np.meshgrid(X,T)
    Z = u(X,T)
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot_surface(X,T,Z,cmap=cm.jet,linewidth=0,antialiased=False)
    ax.view_init(21,-55)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    #ax.set_zlabel('$u(x,y)$')
    if save:
        plt.savefig('Plots/2b_u.pdf')
    else:
        plt.show()

def plot_u_slice(t=0.2):
    x = np.linspace(0,1,200)
    plt.plot(x, u(x,t))
    plt.show()

# Error functions
def disc_l2_error(u_exact_fnc, u_approx):
    '''
    Remember to use a wrapper for u(x,t) with the correct t input!
    '''
    u_exact = u_exact_fnc(np.linspace(0,1,len(u_approx)))
    denom = np.sqrt(np.sum((u_exact-u_approx)**2))
    return denom/np.sqrt(np.sum(u_exact**2))

def cont_L2_error(u_exact_fnc, u_approx, interpolate_type='cubic'):
    x = np.linspace(0,1,len(u_approx))
    approx_fnc = interp1d(x, u_approx, kind=interpolate_type)
    numer_fnc = lambda x: (u_exact_fnc(x) - approx_fnc(x))**2
    denom_fnc = lambda x: (u_exact_fnc(x))**2
    return np.sqrt(quad(numer_fnc, 0, 1)[0]) / np.sqrt(quad(denom_fnc, 0, 1)[0])

# Plots actual solutions
def check_solutions():
    init_fnc = lambda x: u(x,0)
    M = 1001
    Ns = 2**np.arange(5, 9) + 1
    Tend = 1
    exact = lambda x: u(x, Tend)
    U_exact = exact(np.linspace(0,1,M))

    fig, axs = plt.subplots(2,2, figsize=(8, 10))
    axs = [ax for sub in axs for ax in sub]
    for i, ax in enumerate(axs):
        U_approx_BE = utils.BE_dirichlet(init_fnc, Tend, M, Ns[i])
        U_approx_CN = utils.CN_dirichlet(init_fnc, Tend, M, Ns[i])
        ax.plot(U_approx_BE, label='BE')
        ax.plot(U_approx_CN, label='CN')
        ax.plot(U_exact, label='Exact')
        ax.grid()
        ax.legend()
        ax.set_title(f'N exponent: {np.log2(Ns[i]-1)}')
    
    plt.show()

# Refinements
def h_ref(cont=False, save=False):
    init_fnc = lambda x: u(x,0)
    N = 10000
    Ms = 2**np.arange(2,13)
    Tend = 0.2
    if cont:
        get_error = cont_L2_error
        lab = '($L_2$)'
        filename = 'Plots/2b_h_ref_cont.pdf'
    else:
        get_error = disc_l2_error
        lab = '($\ell_2$)'
        filename = 'Plots/2b_h_ref.pdf'

    exact = lambda x: u(x, Tend)

    CN_errors = np.zeros(len(Ms))
    BE_errors = np.zeros(len(Ms))
    for i, M in enumerate(Ms):
        CN_approx = utils.CN(init_fnc, Tend, M, N)
        BE_approx = utils.BE(init_fnc, Tend, M, N)
        CN_errors[i] = get_error(exact, CN_approx)
        BE_errors[i] = get_error(exact, BE_approx)
    
    fig, ax = plt.subplots(figsize=(5,6))
    ax.plot(Ms, CN_errors, 'o-', label='CN '+lab)
    ax.plot(Ms, BE_errors, 'o-', label='BE '+lab)
    # Makes the stipled order line
    O2 = 1/Ms**2
    O2 *= CN_errors[0] / O2[0]
    ax.plot(Ms, O2, '--', label=r'$\mathcal{O}(M^{-2})$')

    #ax.set_title('$h$ refinement')
    ax.legend()
    ax.set_xlabel(r'$M$')
    ax.set_ylabel('Relative error')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid()
    if save:
        plt.savefig(filename)
    else:
        plt.show()

def k_ref(cont=False, save=False):
    init_fnc = lambda x: u(x,0)
    M = 10000
    Ns = 2**np.arange(5,15)
    Tend = 0.2
    if cont:
        get_error = cont_L2_error
        lab = '($L_2$)'
        filename = 'Plots/2b_k_ref_cont.pdf'
    else:
        get_error = disc_l2_error
        lab = '($\ell_2$)'
        filename = 'Plots/2b_k_ref.pdf'

    exact = lambda x: u(x, Tend)

    CN_errors = np.zeros(len(Ns))
    BE_errors = np.zeros(len(Ns))
    for i, N in enumerate(Ns):
        CN_approx = utils.CN(init_fnc, Tend, M, N)
        BE_approx = utils.BE(init_fnc, Tend, M, N)
        CN_errors[i] = get_error(exact, CN_approx)
        BE_errors[i] = get_error(exact, BE_approx)
    
    fig, ax = plt.subplots(figsize=(5,6))
    ax.plot(Ns, CN_errors, 'o-', label='CN '+lab)
    ax.plot(Ns, BE_errors, 'o-', label='BE '+lab)
    # Makes the stipled order line
    O1 = 1/Ns
    O1 *= BE_errors[2] / O1[2]
    O2 = 1/Ns**2
    O2 *= CN_errors[2] / O2[2]
    ax.plot(Ns, O1, '--', label=r'$\mathcal{O}(N^{-1})$')
    ax.plot(Ns, O2, '--', label=r'$\mathcal{O}(N^{-2})$')

    #ax.set_title('$k$ refinement')
    ax.legend()
    ax.set_xlabel(r'$N$')
    ax.set_ylabel('Relative error')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid()
    if save:
        plt.savefig(filename)
    else:
        plt.show()

def kh_ref(cont=False, save=False):
    init_fnc = lambda x: u(x,0)
    Tend = 0.2
    Ms = 2**np.arange(5,12)
    Ns = 100*Tend * Ms
    Ndof = Ns*Ms
    if cont:
        get_error = cont_L2_error
        lab = '($L_2$)'
        filename = 'Plots/2b_kh_ref_cont.pdf'
    else:
        get_error = disc_l2_error
        lab = '($\ell_2$)'
        filename = 'Plots/2b_kh_ref.pdf'
    
    exact = lambda x: u(x, Tend)

    CN_errors = np.zeros(len(Ns))
    BE_errors = np.zeros(len(Ns))
    for i, N in enumerate(Ns):
        CN_approx = utils.CN(init_fnc, Tend, Ms[i], N)
        BE_approx = utils.BE(init_fnc, Tend, Ms[i], N)
        CN_errors[i] = get_error(exact, CN_approx)
        BE_errors[i] = get_error(exact, BE_approx)
    
    fig, ax = plt.subplots(figsize=(5,6))
    ax.plot(Ndof, CN_errors, 'o-', label='CN '+lab)
    ax.plot(Ndof, BE_errors, 'o-', label='BE '+lab)
    # Makes the stipled order line
    Oh = Ndof**(-.5)
    Oh *= BE_errors[2] / Oh[2]
    ax.plot(Ndof, Oh, '--', label=r'$\mathcal{O}(\mathrm{Ndof}^{-0.5})$')
    O1 = 1/Ndof
    O1 *= CN_errors[2] / O1[2]
    ax.plot(Ndof, O1, '--', label=r'$\mathcal{O}(\mathrm{Ndof}^{-1})$')


    #ax.set_title('$k=h$ refinement')
    ax.legend()
    ax.set_xlabel(r'$M\cdot N$')
    ax.set_ylabel('Relative error')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid()
    if save:
        plt.savefig(filename)
    else:
        plt.show()

def krh2_ref(cont=False, save=False):
    init_fnc = lambda x: u(x,0)
    # Random choice of r because CN is stable anyways
    r = 0.5
    Tend = 0.2
    Ms = 2**np.arange(4,11)
    Ns = Tend*Ms**2/r
    Ndof = Ns*Ms
    if cont:
        get_error = cont_L2_error
        lab = '($L_2$)'
        filename = 'Plots/2b_krh2_ref_cont.pdf'
    else:
        get_error = disc_l2_error
        lab = '($\ell_2$)'
        filename = 'Plots/2b_krh2_ref.pdf'
    
    exact = lambda x: u(x, Tend)

    CN_errors = np.zeros(len(Ns))
    BE_errors = np.zeros(len(Ns))
    for i, N in enumerate(Ns):
        CN_approx = utils.CN(init_fnc, Tend, Ms[i], N)
        BE_approx = utils.BE(init_fnc, Tend, Ms[i], N)
        CN_errors[i] = get_error(exact, CN_approx)
        BE_errors[i] = get_error(exact, BE_approx)
    
    fig, ax = plt.subplots(figsize=(5,6))
    ax.plot(Ndof, CN_errors, 'o-', label='CN '+lab)
    ax.plot(Ndof, BE_errors, 'o-', label='BE '+lab)
    # Makes the stipled order line
    O23 = Ndof**(-2/3) * 100
    #Oh *= BE_errors[2] / Oh[2]
    ax.plot(Ndof, O23, '--', label=r'$\mathcal{O}(\mathrm{Ndof}^{-\frac{2}{3}})$')


    #ax.set_title('$k=rh^2$ refinement')
    ax.legend()
    ax.set_xlabel(r'$M\cdot N$')
    ax.set_ylabel('Relative error')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid()
    if save:
        plt.savefig(filename)
    else:
        plt.show()

################################################################################
########     2 c)                                                       ########
################################################################################

# Idea is to solve v_m, m = 1, ... , M first, and then embed into vector of
# dimension M+2 in the end

# Semidiscretization ODE function
@njit
def F(t,v):
    M = len(v)
    h = 1/(M+1)
    v_ret = np.zeros(M)
    v_ret[0] = -1/(2*h) * v[0]*v[1]
    v_ret[-1] = 1/(2*h) * v[-1]*v[-2]
    for i in range(1, M-1):
        v_ret[i] = -1/(2*h) * v[i]*(v[i+1] - v[i-1])
    return v_ret

# Handling the system of ODEs
def solve_burgers(Tend, t_eval):
    '''
    Tend:       When to end calculation
    t_eval:     Times for which to store solution
    '''
    n = 1002
    print('solve_burgers: n =',n)
    n_points = len(t_eval)
    x = np.linspace(0,1, n)
    init_fnc = np.exp(-400 * (x[1:-1] - 1/2)**2)
    x = np.zeros((n,n_points))
    sol = solve_ivp(F, (0,Tend), init_fnc, t_eval=t_eval)
    
    x[1:-1,:] = sol.y
    x[0,:] = np.zeros(n_points)
    x[-1,:] = np.zeros(n_points)
    sol.y = x
    
    return sol

# Doesn't work in script, only .ipynb?
def make_Burgers_gif(sol):
    t = np.arange(50)
    fig, ax = plt.subplots(figsize=(15,10))
    ax.set_ylim(-0.1, 2)
    ax.set_xlim(0,1)
    line, = ax.plot([],[])
    timepoint = ax.text(0.8,0.8,'', fontsize=15)

    def init():
        line.set_data([],[])
        timepoint.set_text('')
        return line,timepoint

    def animate(i):
        x = np.linspace(0,1,1000)
        y = sol.y[:,i]
        line.set_data(x,y)
        timepoint.set_text('$t = {:.4f}$'.format(sol.t[i]))
        return line, timepoint

    anim = FuncAnimation(fig, animate, frames=t, interval=200)
    anim.save('VSCode_test.gif', writer='imagemagick')

def show_burgers_break(plot_times=False):
    sol = solve_burgers(0.8, [0.04, 0.05, 0.06])
    fig, axs = plt.subplots(1,3, figsize=(14,4))
    #axs = [ax for sub in axs for ax in sub]
    x = np.linspace(0,1,len(sol.y[:,0]))
    for ind, ax in enumerate(axs):
        plt.figure()
        plt.plot(x, sol.y[:,ind], linewidth=1)
        if plot_times: plt.text(0.65, 0.6, f'$t = {sol.t[ind]}$')
        plt.xlim(0.2,0.8)
        plt.grid()
        plt.savefig(f'Plots/burgers_t-{sol.t[ind]}.pdf')
        plt.show()


################################################################################
########     Testing                                                    ########
################################################################################

if __name__ == '__main__':
    #path = '/Users/bendikw/Documents/TMA4212-2021V/proj1/part2.py'
    #plot_h_refinement(save=False)
    #plot_convergence('equal', save=False)
    #show_burgers_break(plot_times=False)
    #check_solutions()
    #kh_ref(cont=False, save=True)
    #krh2_ref(cont=False, save=True)
    #h_ref(cont=False, save=True)
    #k_ref(cont=False, save=True)
    #k_ref(cont=True, save=True)
    #plot_u(save=True)
    