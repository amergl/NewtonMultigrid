#!/usr/bin/env python

from numpy import *

from project.poisson1d import Poisson1D
from project.newton import Newton
from project.nonlinear2d import Nonlinear2D
from project.nontrivial2d import Nontrivial2D

def test_newton(problem,ndofs=4,eps=1e-8):
    if problem is "Poisson":
        prob = Poisson1D(ndofs)
    elif problem is "PseudoNonLinear":
        prob = Nonlinear2D(ndofs,0)
    elif problem is "NonLinear":
        prob = Nonlinear2D(ndofs,1e3)
    elif problem is "NonTrivial":
        prob = Nontrivial2D(ndofs,1e3)

    levels=int(log2(ndofs+1))
    newton=Newton(ndofs,levels)
    v=ones(ndofs)
    level=0

    iterations=1
    x=newton.do_newton_lu_cycle(prob)
    #x=newton.do_newton_cycle(prob,v,prob.rhs,1,2,level,max_inner=iterations)

    error=linalg.norm(x-prob.u_exact)
    print "%-15s %e"%(problem,error)
    assert error < eps;

if __name__ == "__main__":
    ndofs=2**3
    test_newton("Poisson",ndofs)
    test_newton("PseudoNonLinear", ndofs)
    test_newton("NonLinear", ndofs)
    test_newton("NonTrivial", ndofs)
