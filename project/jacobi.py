from numpy import *
from scipy import sparse

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
        
    return sparse.csc_matrix(J)


def specificJacobi(ndofs,gamma,u):
    factor=-(ndofs+1)*(ndofs+1)#-1/h**2
    size=ndofs*ndofs
    diag=zeros(size)
    for j in range(ndofs):
        for i in range(ndofs):
            diag[j*ndofs+i]= gamma * u[j*ndofs+i] * exp(u[j*ndofs+i])
    diag+=-4*factor
    data=[diag,[factor]*size,[factor]*size,[factor]*size,[factor]*size]
    offset=[0,-1,1,-ndofs,ndofs]
    M = sparse.spdiags(data,offset,size,size,format='lil')
    for i in range(1,ndofs):
        M[i*ndofs,i*ndofs-1]=M[i*ndofs-1,i*ndofs]=0
    return sparse.csc_matrix(M)


#specificJacobi wiht generalJacobi-diag
def specialJacobi(ndofs, u, g, delta=1e-4):
    n = u.shape[0]
    m = n
    factor=-(ndofs+1)*(ndofs+1)#-1/h**2
    #diag from generalJacobi
    diag=zeros(ndofs*ndofs)
    for i in range(n):
        xtilde = copy(u)
        xtilde[i] += delta
        diag[i] += g(xtilde)[i]
        xtilde[i] -= 2 * delta
        diag[i] -= g(xtilde)[i]
        diag[i] /= 2.0 * delta
    data=[diag,[factor]*ndofs*ndofs,[factor]*ndofs*ndofs,[factor]*ndofs*ndofs,[factor]*ndofs*ndofs]
    offset=[0,-1,1,-ndofs,ndofs]
    M = sparse.spdiags(data,offset,ndofs*ndofs,ndofs*ndofs,format='lil')
    for i in range(1,ndofs):
        M[i*ndofs,i*ndofs-1]=M[i*ndofs-1,i*ndofs]=0
    return sparse.csc_matrix(M)
