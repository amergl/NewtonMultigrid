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

    def do_v_cycle(self, v0, rhs, nu1, nu2, lstart):
        """Straightforward implementation of a V-cycle

        This can also be used inside an FMG-cycle!

        Args:
            v0 (numpy.array): initial values on finest level
            rhs (numpy.array): right-hand side on finest level
            nu1 (int): number of downward smoothing steps
            nu2 (int): number of upward smoothing steps
            lstart (int): starting level

        Returns:
            numpy.array: solution vector on finest level
        """

        assert self.nlevels >= lstart >= 0
        assert v0.size == self.vh[lstart].size

        # set intial conditions (note: resetting vectors here is important!)
        self.reset_vectors(lstart)
        self.vh[lstart] = v0
        self.fh[lstart] = rhs

        # downward cycle
        for l in range(lstart, self.nlevels - 1):
            # print('V-down: %i -> %i' %(l,l+1))
            # pre-smoothing
            for i in range(nu1):
                self.vh[l] = self.smoo[l].smooth(self.fh[l], self.vh[l])

            # restrict
            self.fh[l + 1] = self.trans[l].restrict(self.fh[l] - self.smoo[l].A.dot(self.vh[l]))

        # solve on coarsest level
        self.vh[-1] = sLA.spsolve(self.Acoarse, self.fh[-1])

        # upward cycle
        for l in reversed(range(lstart, self.nlevels - 1)):
            # print('V-up: %i -> %i' %(l+1,l))
            # correct
            self.vh[l] += self.trans[l].prolong(self.vh[l + 1])

            # post-smoothing
            for i in range(nu2):
                self.vh[l] = self.smoo[l].smooth(self.fh[l], self.vh[l])

        return self.vh[lstart]

    def do_newton_fmg_cycle_recursive(self, rhs, h, nu0, nu1, nu2):
        # set intial conditions (note: resetting vectors here is important!)
        self.fh[0] = rhs

        # downward cycle
        if (h < self.nlevels - 1):
            self.fh[h + 1] = self.trans[h].restrict(self.fh[h])
            # plt.plot(h, self.fh[h])
            self.vh[h + 1] = self.do_fmg_cycle_recursive(self.fh[h + 1], h + 1, nu0, nu1, nu2)
        else:
            self.vh[-1] = sLA.spsolve(self.Acoarse, self.fh[-1])
            return self.vh[-1]

        # correct
        self.vh[h] = self.trans[h].prolong(self.vh[h + 1])

        # one v-cycle
        self.vh[h] = self.do_v_cycle(self.vh[h], self.fh[h], nu1, nu2, h)

        # no problemobject, change do_newton_cycle?
        self.vh[h] = self.do_newton_cycle('prob', nu1, nu2, nu0)

        return self.vh[h]

