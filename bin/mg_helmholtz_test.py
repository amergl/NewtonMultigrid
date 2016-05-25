#!/usr/bin/python
import numpy as np
import scipy.linalg as LA
import scipy.sparse.linalg as spLA

from matplotlib.pyplot import *

#from project.poisson1d import Poisson1D
from project.helmholtz1d import Helmholtz1D
from project.weighted_jacobi import WeightedJacobi
#from project.gauss_seidel import GaussSeidel
from project.linear_transfer import LinearTransfer

from project.mymultigrid import MyMultigrid

if __name__ == "__main__":
    n=10
    sigmas = np.linspace(-50,50,2*n)
    ndofs = 15
    niter_list = []
    for sigma in sigmas:
        nlevels = int(np.log2(ndofs+1))

        prob = Helmholtz1D(ndofs=ndofs,sigma=sigma)

        mymg = MyMultigrid(ndofs=ndofs, nlevels=nlevels)
        mymg.attach_transfer(LinearTransfer)
        mymg.attach_smoother(WeightedJacobi,prob.A,omega=2.0/3.0)
#       mymg.attach_smoother(GaussSeidel,prob.A)

        k = 6
        xvalues = np.array([(i+1) * prob.dx for i in range(prob.ndofs)])
        prob.rhs = (np.pi*k)**2 * np.sin(np.pi*k*xvalues)

        uex = spLA.spsolve(prob.A, prob.rhs)

        res = 1
        niter = 0
        err = []
        u = np.zeros(uex.size)
        while res > 1E-10 and niter < 100:
            niter += 1
            u = mymg.do_v_cycle(u, prob.rhs, 2, 2, 0)
            res = LA.norm(prob.A.dot(u)-prob.rhs, np.inf)
            err.append(LA.norm(u-uex, np.inf))
            

        niter_list.append(niter)

    n2=len(sigmas)/2
    subplot(1,2,1)
    xlabel("Sigma")
    ylabel("Iterations")
    title("Negative Sigma")
    plot(sigmas[:n2],niter_list[:n2],"b-")
    subplot(1,2,2)
    xlabel("Sigma")
    ylabel("Iterations")
    title("Positive Sigma")
    plot(sigmas[n2:],niter_list[n2:],"b-")

    show()
    close()
        
