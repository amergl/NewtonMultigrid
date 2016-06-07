#!/usr/bin/env python

from numpy import *
from project.jacobi import generalJacobi,specificJacobi

def testGeneralJacobi(g,x0,solution,eps=1e-8,iterations=50):

    norm=lambda y: linalg.norm(y,inf)
    if type(x0) in [float64,float,int]:
        norm=lambda y: abs(y)

    x=copy(x0)
    fx=g(x)

    while iterations > 0: #fixpoint iteration
        dfx=generalJacobi(g,x)
        if dfx.shape[0] > 1:
            x-= dot(linalg.inv(dfx),fx)
        else:
            x-=fx/dfx[0]
        fx=g(x)
        iterations-=1
    return norm(x-solution) < eps

if __name__ == "__main__":
    """1D linear test"""
    f=lambda x: 2*x - 4
    x0=1.
    solution=2.
    assert testGeneralJacobi(f,x0,solution)

    """1D nonlinear test"""
    f=lambda x: x*x-2
    x0=1.
    solution=sqrt(2)
    assert testGeneralJacobi(f,x0,solution)

    """nD linear test"""
    n=5
    x0=array([i for i in range(n)],dtype=float64)
    H=array([[1./(i+j+1) for j in range(n)] for i in range(n)],dtype=float64)#hilbert matrix
    b=ones(n)
    f=lambda x: b-dot(H,x)
    solution=linalg.solve(H,b)
    assert testGeneralJacobi(f,x0,solution)

    
    """nD nonlinear test"""
    f=lambda x: array([exp(x[i]*x[i]) - 1 for i in range(3)],dtype=float64)
    x0=r_[1.,1.,1.]
    solution=zeros(3)
    assert testGeneralJacobi(f,x0,solution)

    nx=ny=4
    h=1e-1
    gamma=1
    u=ones((nx,ny))
    J=specificJacobi(nx,ny,h,gamma,u).todense()
    print J.shape
    for i in range(J.shape[0]):
        for j in range(J.shape[1]):
            print "%6s"%("%-0.1f"%J[i,j]),
        print ""
    
