# coding=utf-8
import numpy as np
import scipy.sparse as sp

class Nontrivial2D():
    def __init__(self, ndofs, gamma, *args, **kwargs):
        self.gamma = gamma
        self.ndofs = ndofs
        self.dx = 1.0 / (ndofs + 1)
        self.partial_A = self.__get_partial_matrix()
        self.rhs=self.__get_rhs()
        self.u_exact=self.__get_u_exact()

    def __get_partial_matrix(self):
        factor=1./(self.dx*self.dx)
        size=self.ndofs*self.ndofs
        diag=np.zeros(size)
        for j in range(self.ndofs):
            for i in range(self.ndofs):
                diag[j*self.ndofs+i]=4.*factor
        data=[diag,[-factor]*size,[-factor]*size,[-factor]*size,[-factor]*size]
        offset=[0,-1,1,-self.ndofs,self.ndofs]
        M=sp.spdiags(data,offset,size,size,format='lil')
        for i in range(1,self.ndofs):
            M[i*self.ndofs,i*self.ndofs-1]=M[i*self.ndofs-1,i*self.ndofs]=0
        return M

    def A(self,v):
        partial_v=self.partial_A.dot(v)
        exp_v=np.exp(v)
        v_new=np.array([partial_v[i] + self.gamma * v[i] * exp_v[i] for i in range(self.ndofs*self.ndofs)],dtype=np.float64)
        return v_new

    def __get_rhs(self):
        size=self.ndofs*self.ndofs
        b=np.zeros(size)
        xn=np.linspace(0,1,self.ndofs+2)[1:-1]
        yn=np.linspace(0,1,self.ndofs+2)[1:-1]
        f=lambda x,y: ((
                9*np.pi**2+ self.gamma*np.exp((x**2-x**3)*np.sin(3*np.pi*y))
            )
                       *(x**2-x**3)+6*x-2)*np.sin(3*np.pi*y)
        for j in range(self.ndofs):
            for i in range(self.ndofs):
                b[j*self.ndofs+i]=f(xn[i],yn[j])
        return b

    def __get_u_exact(self):
        size=self.ndofs*self.ndofs
        u=np.zeros(size)
        xn=np.linspace(0,1,self.ndofs+2)[1:-1]
        yn=np.linspace(0,1,self.ndofs+2)[1:-1]
        f=lambda x,y: (x**2-x**3)*np.sin(3*np.pi*y)
        for j in range(self.ndofs):
            for i in range(self.ndofs):
                u[j*self.ndofs+i]=f(xn[i],yn[j])
        return u

