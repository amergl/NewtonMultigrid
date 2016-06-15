#!/usr/bin/env python

from numpy import *

from project.poisson1d import Poisson1D
from project.newton import Newton
from project.nonlinear2d import Nonlinear2D
from project.nontrivial2d import Nontrivial2D

from time import time

def test_newton(problem,ndofs=4,eps=1e-8):
    if problem is "PseudoNonLinear":
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

    fstring="%-15s %-15s %e %.4f"
    
    #if False:
    nu0=1
    nu1=1
    nu2=1
    begin=time()
    x=newton.do_newton_fmg_cycle(prob, prob.rhs, level, nu0, nu1, nu2)
    duration=time()-begin
    error=linalg.norm(x-prob.u_exact)
    print fstring%(problem,"Newton-FMG",error,duration)
    assert error < eps
    quit()
    
    begin=time()
    x=newton.do_newton_lu_cycle(prob)
    duration=time()-begin
    error=linalg.norm(x-prob.u_exact)
    print fstring%(problem,"Newton",error,duration)
    assert error < eps

    if False:
        nu1=1
        nu2=1
        n_v_cycles=20
        begin=time()
        x=newton.do_newton_cycle(prob,nu1,nu2,n_v_cycles)
        duration=time()-begin
        error=linalg.norm(x-prob.u_exact)
        print fstring%(problem,"Newton-MG",error,duration)
        assert error < eps

    


if __name__ == "__main__":
    print "%-15s %-15s %-12s %-15s"%("Problem","Method","||e||","Time")
    print "---------------------------------------------------"
    k=3
    ndofs=2**k -1
    test_newton("PseudoNonLinear", ndofs)
    test_newton("NonLinear", ndofs)
    test_newton("NonTrivial", ndofs)
