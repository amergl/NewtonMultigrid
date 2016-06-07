from numpy import *
from scipy.sparse import *

def generalJacobi(g,x,delta=1e-4):
    x=array(x,dtype=float64)
    fx=g(x)
    f=g
    m=0
    if type(fx) == ndarray:
        m=fx.shape[0]
    else:
        m=1
        f=lambda x: array(g(x))

    
    if len(x.shape) is 0:
        x=r_[float64(x)]
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
            
    if m == 1 and n == 1:
        J=array([J[0,0]],dtype=float64)
        
    return J


def specificJacobi(nx,ny,h,gamma,u):
    factor=-1/(h*h)
    diag=zeros(nx*ny)
    for j in range(ny):
        for i in range(nx):
            diag[j*ny+i]=4./(h*h) + gamma * u[i,j] * exp(u[i,j])
    data=[diag,[factor]*nx*ny,[factor]*nx*ny,[factor]*nx*ny,[factor]*nx*ny]
    offset=[0,-1,1,-nx,nx]
    return spdiags(data,offset,nx*ny,nx*ny,format='csc')
