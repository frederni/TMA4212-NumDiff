import numpy as np
import matplotlib.pyplot as plt 
import numpy.linalg as lin
from matplotlib import cm 
import matplotlib.colors as mcolors
# Differentials
# DJ Marius
from algorithm import solve 
from objects import Method, Params, Result

from tqdm import tqdm
from Analytical import Analytical


class MeshRefinement:
    '''
    Manufactured solution with u and f as specified
    '''

    method_str = {
        Method.abdul_special_5: 'Rowwise Fourier (5-point)',
        Method.abdul_special_9: 'Rowwise Fourier (9-point)',
        Method.fast_2d_5: '2D Fourier (5-point)',
        Method.fast_2d_9: '2D Fourier (9-point)'
    }

    def f(self, x, y):
        return np.sin(np.pi*x)*np.sin(np.pi*y)
    
    def u(self,x,y):
        return self.f(x,y) / (4*np.pi**4)
    
    def __init__(self, method1: Method, method2: Method):
        self.M = 2**np.arange(3,9)
        self.method = method1
        self.method2 = method2
        self.task_string = 'Part2_g'
    
    def plot_u(self, save=False):
        M = 1000
        X,Y = np.linspace(0,1,M), np.linspace(0,1,M)
        X,Y = np.meshgrid(X,Y)
        Z = self.u(X,Y)
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.plot_surface(X,Y,Z,cmap=cm.jet,linewidth=0,antialiased=False)
        #ax.view_init(21,-55)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        #ax.set_zlabel('$u(x,y)$')
        if save:
            plt.savefig(f'Plots_pt2/{self.task_string}_u.pdf')
        else:
            plt.show()

    def plot_f(self, save=False):
        M = 1000
        X,Y = np.linspace(0,1,M), np.linspace(0,1,M)
        X,Y = np.meshgrid(X,Y)
        Z = self.f(X,Y)
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.plot_surface(X,Y,Z,cmap=cm.jet,linewidth=0,antialiased=False)
        #ax.view_init(21,-55)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        #ax.set_zlabel('$u(x,y)$')
        if save:
            plt.savefig(f'Plots_pt2/{self.task_string}_f.pdf')
        else:
            plt.show()
        
    def compute_error(self, approx):
        '''
        approx: approximated u of shape (N-1)x(N-1)
        return: nd.array of length 6
        '''
        inner_points = approx.shape[0]
        x = np.linspace(0,1,inner_points+2)[1:-1]
        y = np.linspace(0,1,inner_points+2)[1:-1]
        x,y = np.meshgrid(x,y)
        exact = self.u(x,y)
        numerator = lin.norm(exact-approx)
        return numerator / lin.norm(exact)
    
    def get_results(self, method, pbar = None):
        params = []
        results = []
        for m in self.M:
            params.append(Params(method, m, self.f))
        for p in params:
            results.append(solve(p))
            print(self.method_str[method], 'with M =', p.N, ':\texec_time =', results[-1].time, 's')
            if pbar: pbar.update(1)

        return results
    
    def plot_refinement(self, save=False):
        results = self.get_results(self.method)
        results2 = self.get_results(self.method2)
        errors = np.zeros(len(self.M))
        errors2 = errors.copy()
        Ndof = self.M**2
        for i in range(len(errors)):
            errors[i] = self.compute_error(results[i].sol)
            errors2[i] = self.compute_error(results2[i].sol)
        
        if self.method == Method.abdul_special_5:
            labstring = '5-point stencil'
            labstring2 = '9-point stencil'
        else:
            labstring = '9-point stencil'
            labstring2 = '5-point stencil'
        
        fig, ax = plt.subplots(figsize=(5,6))
        O1 = 1/Ndof
        O2 = 1/Ndof**2
        ax.plot(Ndof, O1, '--', label=r'$\mathcal{O}(\mathrm{Ndof}^{-1})$')
        ax.plot(Ndof, O2, '--', label=r'$\mathcal{O}(\mathrm{Ndof}^{-2})$')
        ax.plot(Ndof, errors, 'o-', label=labstring)
        ax.plot(Ndof, errors2, 'o-', label=labstring2)
        self.prep_axis(ax, 'DOF')
        if save:
            plt.savefig(f'Plots_pt2/{self.task_string}_UMR.pdf')
        else:
            plt.show()
    
    def prep_axis(self, ax, yax):
        ax.legend()
        ax.set_xlabel('Ndof')
        if yax == 'DOF':
            ax.set_ylabel('Relative error')
            ax.set_title('Convergence of relative $\ell_2$ error')
        elif yax == 'time':
            ax.set_ylabel('Time consumption')
            ax.set_title('Execution time')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid()




class ComplicatedBVP(MeshRefinement):
    '''
    Create instance by passing the method. Remember to use .run() on instance
    BEFORE the plotting! It stores the results internally
    '''

    colorcycle = ['maroon', 'orangered', 'cadetblue', 'orchid']

    def reset_colorcycle(self):
        self.color_counter = 0

    def get_next_CSS_color(self):
        l = len(self.colorcycle)
        colorname = self.colorcycle[ self.color_counter % l ]
        self.color_counter += 1
        return mcolors.CSS4_COLORS[ colorname ]

    def get_style(self, method):
        if method.dst_dim == 1:
            return '^', 7
        return '+', 10

    def __init__(self, *args):
        self.methods = list(args)
        self.task_string = 'Part2_h'
        self.M = 2**np.arange(3,13)
        self.color_counter = 0
    
    def u(self,x,y):
        return Analytical.u(x,y)
    
    def f(self,x,y):
        return Analytical.f(x,y)
    
    # NB! Run before plotting
    def run(self):
        self.results = []
        with tqdm(total=len(self.M)*len(self.methods)) as pbar:
            for method in self.methods:
                self.results.append(self.get_results(method, pbar=pbar))

    def plot_DOF_refinement(self, save=False):
        self.reset_colorcycle()
        errors = np.zeros((len(self.methods),len(self.M)))
        Ndof = self.M**2
        for i in range(len(errors)):
            for j in range(len(errors[i])):
                errors[i][j] = self.compute_error(self.results[i][j].sol)

        fig, ax = plt.subplots(figsize=(5,7))
        O1 = 1/Ndof
        O1 *= errors[0][0] / O1[0]
        O2 = 1/Ndof**(2)
        O2 *= errors[-1][0] / O2[0]
        ax.plot(Ndof, O1, '--', linewidth=1, label=r'$\mathcal{O}(\mathrm{Ndof}^{-1})$')
        ax.plot(Ndof, O2, '--', linewidth=1, label=r'$\mathcal{O}(\mathrm{Ndof}^{-2})$')
        for i, meth_error in enumerate(errors):
            method = self.methods[i]
            point, size = self.get_style(method)
            ax.plot(Ndof, meth_error, point, markersize=size, fillstyle='none', color=self.get_next_CSS_color(), label=self.method_str[method])
        self.prep_axis(ax, 'DOF')
        if save:
            plt.tight_layout()
            plt.savefig(f'Plots_pt2/{self.task_string}_DOF_refinement.pdf')
        else:
            plt.tight_layout()
            plt.show()
    
    def plot_time_consumption(self, save=False):
        self.reset_colorcycle()
        times = np.zeros( (len(self.methods),len(self.M)) )
        Ndof = self.M**2
        for i in range(len(times)):
            for j in range(len(times[i])):
                times[i][j] = self.results[i][j].time
        fig, ax = plt.subplots(figsize=(5,7))
        O = Ndof*np.log2(Ndof)
        O *= times[0][-1] / O[-1]
        ax.plot(Ndof[2:], O[2:], '--', linewidth=1, label=r'$\mathcal{O}(\mathrm{Ndof}\log_2(\mathrm{Ndof}))$')
        for i, meth_time in enumerate(times):
            method = self.methods[i]
            point, size = self.get_style(method)
            col = self.get_next_CSS_color()
            ax.plot(Ndof, meth_time, '-', linewidth=0.5, color=col)
            ax.plot(Ndof, meth_time, point, markersize=size, fillstyle='none', color=col, label=self.method_str[method])
        self.prep_axis(ax, 'time')
        if save:
            plt.tight_layout()
            plt.savefig(f'Plots_pt2/{self.task_string}_time_consumption.pdf')
        else:
            plt.tight_layout()
            plt.show()

def main():
    method1 = Method.abdul_special_5
    method2 = Method.abdul_special_9
    method3 = Method.fast_2d_5
    method4 = Method.fast_2d_9
    mr = MeshRefinement(method1, method2)
    #mr.plot_f(save=False)
    #mr.plot_u(save=True)
    #mr.plot_refinement(save=True)
    cm = ComplicatedBVP(method1, method2, method3, method4)
    cm.run()
    cm.plot_DOF_refinement(save=True)
    cm.plot_time_consumption(save=True)

if __name__ == '__main__':
    main()