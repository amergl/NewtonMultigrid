#!/usr/bin/env python
import numpy as np

from project.nonlinear2d import Nonlinear2D


def test_exact_solution():
    eps=1e-8
    it=10
    list_gamma=[1e3]
    list_n=[2**i for i in range(1,10)]

    for gamma in list_gamma:
        for ndofs in list_n:
            prob = Nonlinear2D(ndofs, gamma)
            
            error=np.linalg.norm(prob.rhs-prob.A(prob.u_exact),np.inf)

            print error
            #assert error < eps

if __name__ == "__main__":
    test_exact_solution()
