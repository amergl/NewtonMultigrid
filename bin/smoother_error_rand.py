# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as LA
from matplotlib import rc

from project.gauss_seidel import GaussSeidel
from project.poisson1d import Poisson1D
from project.weighted_jacobi import WeightedJacobi

if __name__ == "__main__":

    rc('font', family='sans-serif', size=32)
    rc('legend', fontsize='small')
    rc('xtick', labelsize='small')
    rc('ytick', labelsize='small')

    prob = Poisson1D(ndofs=63)
    urand = np.random.rand(prob.ndofs)

    niter = 100

    smoother_list = [(WeightedJacobi(prob=prob, omega=2.0 / 3.0), 'Jacobi_w23'),
                     (GaussSeidel(prob=prob), 'GaussSeidel')]

    for smoother, fname in smoother_list:

        u = urand
        uex = prob.u_exact
        err = [LA.norm(u - uex, np.inf)]

        for i in range(niter):
            u = smoother.do_smoothing(u_old=u)
            err.append(LA.norm(u - uex, np.inf))

        fig = plt.subplots(figsize=(10, 7))

        plt.plot(range(niter + 1), err, lw=3)

        plt.axis([0, niter, 0, 1])
        plt.xlabel('Iteration')
        plt.ylabel('Error')

        plt.grid()
        plt.tight_layout()

        plt.savefig(fname + '_rand.pdf', rasterized=True, transparent=True, bbox_inches='tight')

    for smoother, fname in smoother_list:

        xvalues = np.array([(i + 1) * prob.dx for i in range(prob.ndofs)])

        fig = plt.subplots(figsize=(10, 7))

        colors = ['Blue', 'Red', 'Green']

        ind = 0
        for mode in [1, 3, 6]:

            u = np.sin(np.pi * mode * xvalues)

            uex = prob.u_exact
            err = [LA.norm(u - uex, np.inf)]

            for i in range(niter):
                u = smoother.do_smoothing(u_old=u)
                err.append(LA.norm(u - uex, np.inf))

            label = 'k=' + str(mode)
            plt.plot(range(niter + 1), err, color=colors[ind], lw=3, label=label)
            ind += 1

        plt.axis([0, niter, 0, 1])
        plt.xlabel('Iteration')
        plt.ylabel('Error')

        plt.legend(loc=3)
        plt.grid()
        plt.tight_layout()

        plt.savefig(fname + '_modes.pdf', rasterized=True, transparent=True, bbox_inches='tight')

        # plt.show()
