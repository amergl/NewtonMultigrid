import scipy.sparse.linalg as sLA
import numpy as np

#from pymg.multigrid_base import MultigridBase
from project.jacobi import generalJacobi,specificJacobi
from scipy.sparse import *

from pymg.multigrid_base import MultigridBase
from project.linear_transfer import LinearTransfer
from project.weighted_jacobi import WeightedJacobi
from project.mymultigrid import MyMultigrid

from project.nonlinear2d import Nonlinear2D

class Newton(MultigridBase):
    """Implementation of a newton multigrid solver
    """

    def __init__(self, ndofs, nlevels):
        """Initialization routine
        """
        assert np.log2(ndofs+1) >= nlevels
        super(Newton, self).__init__(ndofs, nlevels)
        self.ndofs=ndofs

    def do_newton_lu_cycle(self,prob,max_outer=20,eps=1e-10):
        """Newton iteration using sparse lu decomposition"""
        #compatibility for linear problems
        v=np.ones(prob.rhs.shape)
        F = prob.A
        if type(prob.A) is csc_matrix:
            F = lambda x: prob.A.dot(x)
            
        r=prob.rhs - F(v)
        while max_outer > 0 and np.linalg.norm(r,np.inf) > eps:
            Jv = specificJacobi(prob.ndofs,prob.gamma, v)
            r=prob.rhs-F(v)
            e=sLA.splu(csc_matrix(Jv)).solve(r)
            v+=e
            max_outer -= 1
        return v

    #nu0 No. pre smoothing
    #nu1 No. post smoothing
    def do_newton_cycle(self, prob, nu1, nu2, n_v_cycles,max_outer=20, eps=1e-10):

        v=np.ones(prob.rhs.shape[0])
        #compatibility for linear problems
        F = prob.A
        if type(prob.A) is csc_matrix:
            #wenn prob.A nicht linear -> dann Funktion ; Ansonsten: Matrix Vektor Produkt
            F = lambda x: prob.A.dot(x)

        r=prob.rhs-F(v)
        
        while max_outer > 0 and np.linalg.norm(r,np.inf) > eps:
            Jv = specificJacobi(prob.ndofs,prob.gamma, v)
            r = prob.rhs - F(v)

            #approximate error using linear multigrid
            mgrid = MyMultigrid(prob.ndofs,int(np.log2(prob.ndofs+1)))
            mgrid.attach_transfer(LinearTransfer)
            mgrid.attach_smoother(WeightedJacobi,Jv,omega=2.0/3.0)
            e=np.ones(prob.ndofs)
            for i in range(n_v_cycles):
                e=mgrid.do_v_cycle(e,r,nu1,nu2,0)

            v += e
            max_outer -= 1
            
        return v

    def do_newton_fmg_cycle(self, prob, rhs, level, nu0, nu1, nu2, max_inner=1,max_outer=20):
        self.fh[0] = rhs

        mgrid = MyMultigrid(prob.ndofs,2**(prob.ndofs)-1)
        mgrid.attach_transfer(LinearTransfer)


        # downward cycle
        if (level < self.nlevels - 1):
            self.fh[level + 1] = mgrid.trans[level].restrict(self.fh[level])
            # plt.plot(level, self.flevel[level])
            self.vh[level + 1] = self.do_newton_fmg_cycle(prob,self.fh[level + 1], level + 1, nu0, nu1, nu2)
        else:
            self.vh[-1] = sLA.spsolve(self.Acoarse, self.fh[-1])
            return self.vh[-1]

        for i in range(nu0):
            self.vh[level] = self.do_newton_cycle(prob,self.vh[level], self.fh[level], nu1, nu2, level, max_inner, max_outer)

        return self.vh[level]
