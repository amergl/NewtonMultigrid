import scipy.sparse.linalg as sLA
import numpy as np
import math

#from pymg.multigrid_base import MultigridBase
from project.jacobi import specialJacobi
from scipy.sparse import *


from pymg.multigrid_base import MultigridBase
from project.linear_transfer2D import LinearTransfer2D
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

        
    def single_push(self, ndofs, v,level):
        mgrid = MyMultigrid(ndofs,int(np.log2(ndofs+1)))
    	mgrid.attach_transfer(LinearTransfer2D)
    	
    	# restrict upwards for incompatibility problems concerning the operator
        for i in range(0, self.nlevels-level):
	    	v = mgrid.trans[i].restrict(v)
    	return v
        
    def push(self, prob, v, rhs, level):
        return self.single_push(prob.ndofs,v,level),self.single_push(prob.ndofs,rhs,level)

    def single_pull(self, ndofs, v,level):
        #approximate error using linear multigrid
        mgrid = MyMultigrid(ndofs,int(np.log2(ndofs+1)))
    	mgrid.attach_transfer(LinearTransfer2D)
    	
    	# prolongate downwards for incompatibility problems concerning the operator
        for i in range(0, self.nlevels-level):
	    	v = mgrid.trans[i].prolong(v)
    	return v
    	
    def pull(self, prob, v, rhs, level):
        return self.single_pull(prob.ndofs,v,level),self.single_pull(prob.ndofs,rhs,level)


    def do_newton_lu_cycle(self,prob,max_outer=20,eps=1e-10):
        """Newton iteration using sparse lu decomposition"""
        #compatibility for linear problems
        v=np.ones(prob.rhs.shape)
        F = prob.A
        if type(prob.A) is csc_matrix:
            F = lambda x: prob.A.dot(x)
            
        r=prob.rhs - F(v)
        while max_outer > 0 and np.linalg.norm(r,np.inf) > eps:
            Jv = specialJacobi(prob.ndofs, v, F)
            r=prob.rhs-F(v)
            e=sLA.splu(csc_matrix(Jv)).solve(r)
            v+=e
            max_outer -= 1
        return v

    def do_newton_cycle2(self, prob, nu1, nu2, n_v_cycles=1):
        return self.do_newton_cycle(prob,np.ones(prob.rhs.shape),prob.rhs,nu1,nu2,0,n_v_cycles)

    #nu0 No. pre smoothing
    #nu1 No. post smoothing
    def do_newton_cycle(self, prob, v0, rhs0, nu1, nu2, level, n_v_cycles,max_outer=20, eps=1e-10):
        # level: current level

        #approximate error using linear multigrid
        mgrid = MyMultigrid(prob.ndofs,int(np.log2(prob.ndofs+1)))
    	mgrid.attach_transfer(LinearTransfer2D)
    	
        v = v0
        rhs = rhs0

        #compatibility for linear problems
        F = prob.A
        if type(prob.A) is csc_matrix:
            #wenn prob.A nicht linear -> dann Funktion ; Ansonsten: Matrix Vektor Produkt
            F = lambda x: prob.A.dot(x)

		# restrict upwards for incompatibility problems concerning the operator
        v_rest, rhs_rest = self.push(prob,v,rhs,level)
	    	
        r=rhs_rest-F(v_rest)
        # prolongate downwards for incompatibility problems concerning the operator
        for i in range(0, self.nlevels-level):
	    	r = mgrid.trans[i].prolong(r)
        
        
        while max_outer > 0 and np.linalg.norm(r,np.inf) > eps:
            Jv = specialJacobi(r.shape[0],v, F)
            
            v_rest, rhs_rest = self.push(prob,v,rhs,level)
            r = rhs_rest - F(v_rest)
            # prolongate downwards 
            for i in range(0, self.nlevels-level):
	    		r = mgrid.trans[i].prolong(r)

            mgrid.attach_smoother(WeightedJacobi,Jv,omega=2.0/3.0)
            e=np.ones(r.shape[0])
            for i in range(n_v_cycles):
                e=mgrid.do_v_cycle(e,r,nu1,nu2,0)

            v += e
            max_outer -= 1
            
        return v

    def do_newton_fmg_cycle(self, prob, rhs, level, nu0, nu1, nu2, max_inner=1,max_outer=20):
        self.fh[0] = rhs
        current_ndofs = int(math.sqrt(rhs.shape[0]))		
        mgrid = MyMultigrid(current_ndofs,int(np.log2(current_ndofs+1)))
        mgrid.attach_transfer(LinearTransfer2D)
        ############ TODO F ans ende, push prob parameter rauswerfen
        M = specialJacobi(current_ndofs, np.ones(rhs.shape[0]), lambda x: rhs - self.push(prob.ndofs,prob.A(self.push(prob.ndofs,x,level))))
        mgrid.attach_smoother(WeightedJacobi,M,omega=2.0/3.0)

        if (level < self.nlevels - 2):
            self.fh[level + 1] = mgrid.trans[level].restrict(self.fh[level])
            # plt.plot(level, self.flevel[level])
	    print level
            self.vh[level + 1] = self.do_newton_fmg_cycle(prob,self.fh[level + 1], level + 1, nu0, nu1, nu2)
        else:
	    print M.todense(), M.shape, self.fh[-1].shape
            self.vh[-1] = self.do_newton_cycle(prob,self.vh[-1], self.fh[-1], nu1, nu2, level, n_v_cycles=1, max_outer=1) #sLA.spsolve(M, self.fh[-1])
            return self.vh[-1]

        for i in range(nu0):
            self.vh[level] = self.do_newton_cycle(prob,self.vh[level], self.fh[level], nu1, nu2, level, max_inner, max_outer)

        return self.vh[level]
