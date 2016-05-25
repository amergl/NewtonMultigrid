#!/usr/bin/env python

from numpy import *

"""
calculates jacobi matrix from f using the list of x variables
"""
def jacobi(g,x,delta=1e-4):
    x=array(x,dtype=float64)
    fx=g(x)
    f=g
    m=0
    if type(fx) == ndarray:
        m=fx.shape[0]
    else:
        m=1

    if m==1:
        f=lambda x: array(g(x))
    n=x.shape[0]
    J=zeros((m,n))

    for i in range(m):
        for j in range(n):
            xtilde=copy(x)
            xtilde[j]+=delta
            J[i,j]+=f(xtilde)[i]
            xtilde[j]-=2*delta
            J[i,j]-=f(xtilde)[i]
            J[i,j]/=2.0*delta
    return J

if __name__ == "__main__":
    m=4
    grad=array([2,3,0.5,2,5,3],dtype=float64)
    f=lambda x: array([i*dot(grad,x) for i in range(1,m+1)])
    x=array(range(1,grad.shape[0]+1))
    J=jacobi(f,x)

    e=0
    for i in range(m):
        e+=linalg.norm((i+1)*grad - J[i,],inf)
    print "error is",e
