#!/usr/bin/env python

from project.newton import Newton
from project.nontrivial2d import Nontrivial2D
from numpy import *

if __name__ == "__main__":
    #table 2: convergence factor
    print "Table 2:"
    #ndofs=127
    ndofs=15
    gammas=[0,1,10,100,1000,10000]
    its=[]
    factors=[]
    levels = int(log2(ndofs + 1))
    newton = Newton(ndofs, levels)
    old_v = 1
    for gamma in gammas:
        prob = Nontrivial2D(ndofs,gamma)
        #calc
        v, outer = newton.do_newton_lu_cycle(prob, max_outer=20, eps=1e-10)[0:2]
        outer = 20 - outer
        its.append(outer)
        factors.append(linalg.norm(v - prob.u_exact)/linalg.norm(old_v - prob.u_exact))
        old_v = v
    fstring="%-20s|"
    rhsstring="%5s"
    print fstring%"Gamma",

    for gamma in gammas:
        print rhsstring%gamma,
    print ""
    print "---------------------------------------------------------"
    print fstring%"Convergence Factor",
    for factor in factors:
        print rhsstring%factor,
    print ""
    print fstring%"Newton Iterations",
    for iteration in its:
        print rhsstring%iteration,
    print ""

    print "\n\n"
    print "Table 3:"

    #table 3: residuum lower than 1e-10
    fstring="%-10s|%-10s %-10s"
    print fstring%("Method","Outer","Inner")
    print "---------------------------"
    gamma=10
    prob=Nontrivial2D(ndofs,gamma)
    #newton (lu decomposition)
    v, outer = newton.do_newton_lu_cycle(prob, max_outer=20, eps=1e-10)[0:2]
    outer = 20 - outer
    inner="-"
    print fstring%("Newton",outer,inner)

    inner=[20,10,5,2,1]
    for iteration in inner:
        #set inner iterations
        v, outer = newton.newton_mg(prob, 1, 1, n_v_cycles=iteration)[0:2]
        outer = 20 - outer
        #outer=0#calc using inexact newton
        print fstring%("Newton-MG",outer,iteration)

    print "\n\n"
    print "Table 5:"
    fstring="%-20s %10s %10s"
    print fstring%("Cycle","||r||","||e||")
    print "--------------------------------------------"
    #start with a newton fmg cycle and further reduce the residuum
    v = newton.do_newton_fmg_cycle(prob, prob.rhs, 0, 0, 1, 1)
    error = linalg.norm(v - prob.u_exact)
    res = linalg.norm(prob.rhs - prob.A.dot(v))
    print fstring%("FMG-Newton-MG",res,error)
    for i in range(1,14):
        v = newton.do_newton_fmg_cycle(prob, prob.rhs, 0, i, 1, 1)[0]
        error = linalg.norm(v - prob.u_exact)
        res = linalg.norm(v - prob.u_exact) / linalg.norm(prob.u_exact)
        print fstring%("Newton-MG %d"%i,res,error)