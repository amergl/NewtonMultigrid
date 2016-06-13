#!/usr/bin/env python

#from project.newton import Newton
from project.nontrivial2d import Nontrivial2D

if __name__ == "__main__":
    #table 2: convergence factor
    print "Table 2:"
    ndofs=127
    gammas=[0,1,10,100,1000,10000]
    its=[]
    factors=[]
    for gamma in gammas:
        prob = Nontrivial2D(ndofs,gamma)
        #calc
        its.append(0)
        factors.append(0)
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
    ndofs=127
    gamma=10
    prob=Nontrivial2D(ndofs,gamma)
    #newton (lu decomposition)
    outer=0
    inner="-"
    print fstring%("Newton",outer,inner)

    inner=[20,10,2,1]
    for iteration in inner:
        #set inner iterations
        outer=0#calc using inexact newton
        print fstring%("Newton-MG",outer,iteration)

    print "\n\n"
    print "Table 5:"
    fstring="%-20s %10s %10s"
    print fstring%("Cycle","||r||","||e||")
    print "--------------------------------------------"
    #start with a newton fmg cycle and further reduce the residuum
    print fstring%("FMG-Newton-MG",0,0)
    for i in range(1,14):
        print fstring%("Newton-MG %d"%i,0,0)
    

    

