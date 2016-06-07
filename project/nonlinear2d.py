# coding=utf-8
import numpy as np
import scipy.sparse as sp

class Nonlinear2D():
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
        return sp.spdiags(data,offset,size,size,format='csc')

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
        f=lambda x,y: 2*((x-x*x)+(y-y*y)) + self.gamma*(x-x*x)*(y-y*y)*np.exp((x-x*x)*(y-y*y))
        for j in range(self.ndofs):
            for i in range(self.ndofs):
                b[j*self.ndofs+i]=f(xn[i],yn[j])
        return b

    def __get_u_exact(self):
        size=self.ndofs*self.ndofs
        u=np.zeros(size)
        xn=np.linspace(0,1,self.ndofs+2)[1:-1]
        yn=np.linspace(0,1,self.ndofs+2)[1:-1]
        f=lambda x,y: (x-x*x)*(y-y*y)
        for j in range(self.ndofs):
            for i in range(self.ndofs):
                u[j*self.ndofs+i]=f(xn[i],yn[j])
        return u

