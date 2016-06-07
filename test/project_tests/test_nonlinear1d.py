#!/usr/bin/env python
import numpy as np

from project.nonlinear1d import Nonlinear1D


def test_exact_solution():
    eps=1e-8
    it=10
    list_gamma=[0,-1,1]
    list_n=[2**i for i in [1,5,10]]

    for gamma in list_gamma:
        for ndofs in list_n:
            prob = Nonlinear1D(ndofs,gamma)

            print prob.rhs
            print prob.u_exact
            print prob.partial_A.todense()
            print prob.A(prob.u_exact)

            
            error=np.linalg.norm(prob.rhs-prob.A(prob.u_exact),np.inf)
            
            assert error < eps

if __name__ == "__main__":
    test_exact_solution()
