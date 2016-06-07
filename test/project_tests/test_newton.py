#!/usr/bin/env python

from numpy import *
from project.poisson1d import Poisson1D
from project.newton import Newton

def test_newton(problem,ndofs=4,eps=1e-8):
    if problem is "Poisson":
        prob = Poisson1D(ndofs)
    levels=int(log2(ndofs+1))
    newton=Newton(ndofs,levels)

    iterations=5
    x=newton.do_newton_cycle(prob,1,2,iterations)

    assert linalg.norm(x-prob.u_exact) < eps;

if __name__ == "__main__":
    ndofs=4
    test_newton("Poisson",ndofs)
