import numpy as np 
import matplotlib.pyplot as plt 
import scipy.sparse as spr
from scipy.sparse.linalg import splu


def CN_o1(u0, Tend, N=None, r=None):
    '''
    Computes matrix U of shape (Tend/k, M), where each row i
    is the space discretization of u(x, t_i). Von Neuman boundary
    conditions hard coded in.
    
    u0  : disc. of u(x, 0)
    N   : number of nodes in time disc.
    Tend: end time (potentially not included)
    '''
    # Verifying shape and creating parameters
    M = len(u0)-2
    h = 1/(M+1)
    if N:
        k = Tend/N
        r = k/h**2
    elif r:
        k = r * h**2
        N = Tend // k

    # Initiating U
    res = u0.copy()
    
    # Construction of linear systems
    # Matrix Q in semi-discretization theory
    main = np.ones(M+2) * (-2)
    main[0] = -1
    main[-1] = -1

    over = np.ones(M+1)
    under = np.ones(M+1)
    offsets = [-1,0,1]
    Q = spr.diags([under, main, over], offsets, format="csc")

    leftMatrix_LU = splu(spr.eye(M+2, format="csc") - r/2*Q)
    rightMatrix = spr.eye(M+2, format="csc") + r/2*Q
    # Solving the system
    for i in range(1, N):
        res = leftMatrix_LU.solve(rightMatrix*res)
    
    return res


def CN_o2(u0, Tend, N=None, r=None):
    '''
    u0  : disc. of u(x, 0)
    N   : number of nodes in time disc.
    Tend: end time (potentially not included)
    '''
    M = len(u0)-2
    h = 1/(M+1)
    if N:
        k = Tend/N
        r = k/h**2
    elif r:
        k = r * h**2
        N = Tend // k

    # Initiating U
    res = u0.copy()
    
    # Construction of linear systems
    # Matrix Q in semi-discretization theory
    main = np.ones(M+2) * (-2)

    over = np.ones(M+1)
    over[0] = 2
    under = np.ones(M+1)
    under[-1] = 2
    offsets = [-1,0,1]
    Q = spr.diags([under, main, over], offsets, format="csc")

    leftMatrix_LU = splu(spr.eye(M+2, format="csc") - r/2*Q)
    rightMatrix = spr.eye(M+2, format="csc") + r/2*Q
    # Solving the system
    for i in range(1, int(N+1)):
        res = leftMatrix_LU.solve(rightMatrix*res)
        #if i==int(ntsteps/2): print('Halfway there!')
    
    return res

def f(x):
    return 2*np.pi*x - np.sin(2*np.pi*x)


############################################################
#######  For 2b)
#######  Manufactured solution as in part2.py
############################################################

# Solvers for the manufactured Neumann system
def CN(init_fnc, Tend, M, N):
    # Using M as total length of v vector here
    res = init_fnc(np.linspace(0,1,M+2))
    h = 1/(M+1)
    k = Tend/N
    r = k/h**2
    # Construction of linear systems
    # Matrix Q as in semi-discretization theory
    main = np.ones(M+2) * (-2)
    over = np.ones(M+1)
    over[0] = 2
    under = np.ones(M+1)
    under[-1] = 2
    offsets = [-1,0,1]
    Q = spr.diags([under, main, over], offsets, format="csc")

    leftMatrix_LU = splu(spr.eye(M+2, format="csc") - r/2*Q)
    rightMatrix = spr.eye(M+2, format="csc") + r/2*Q
    # Solving the system
    for i in range(1, int(N+1)):
        res = leftMatrix_LU.solve(rightMatrix*res)
        #if i==int(ntsteps/2): print('Halfway there!')
    
    return res

def BE(init_fnc, Tend, M, N):
    # Using M as total length of v vector here
    res = init_fnc(np.linspace(0,1,M+2))
    h = 1/(M+1)
    k = Tend/N
    r = k/h**2
    # Construction of linear systems
    # Matrix Q as in semi-discretization theory
    main = np.ones(M+2) * (-2)
    over = np.ones(M+1)
    over[0] = 2
    under = np.ones(M+1)
    under[-1] = 2
    offsets = [-1,0,1]
    Q = spr.diags([under, main, over], offsets, format="csc")

    leftMatrix_LU = splu(spr.eye(M+2, format="csc") - r*Q)
    # Solving the system
    for i in range(1, int(N+1)):
        res = leftMatrix_LU.solve(res)
        #if i==int(ntsteps/2): print('Halfway there!')
    
    return res

############################################################
#######  Testing
############################################################

if __name__== '__main__':
    r = 0.01
    M = 100
    k = r * 1/(M+1)**2
    print(k)
    x = np.linspace(0,1,M+2)
    u0 = f(x)
    ut_1 = CN_o1(u0, r, 0.1)
    ut_2 = CN_o2(u0, r, 0.1)
    plt.plot(u0, label='Initial')
    plt.plot(ut_1, label='Order 1')
    plt.plot(ut_2, label='Order 2')
    plt.legend()
    plt.show()
