import scipy.sparse.linalg as sLA
import numpy as np

#from pymg.multigrid_base import MultigridBase
from project.jacobi import generalJacobi,specificJacobi
from scipy.sparse import *

class Newton():#MultigridBase):
    """Implementation of a newton multigrid solver
    """

    def __init__(self, ndofs, nlevels):
        """Initialization routine
        """
        assert np.log2(ndofs+1) >= nlevels
        #super(Newton, self).__init__(ndofs, nlevels)
        self.ndofs=ndofs

    #nu0 No. pre smoothing
    # nu1 No. post smoothing

    def do_newton_cycle(self, prob, nu0, nu1, newton_it=10):
        #attach prolongation operator
        #attach restriction operator
        #attach smoother -> weighted Jacobi(fundamental)

        x=np.copy(prob.rhs)

        #klassischer Newton...
        #F=lambda z: np.array(prob.rhs) - prob.A(z)
        #while newton_it > 0:
        #    J=generalJacobi(F,x)
        #    x-= np.dot(np.linalg.inv(J),F(x))
        #    newton_it -=1

        F = prob.A
        if type(prob.A) is csc_matrix:
            #wenn prob.A nicht linear -> dann Funktion ; Ansonsten: Matrix Vektor Produkt
            F = lambda x: prob.A.dot(x)

        #v entspricht xStart
        v = np.copy(prob.rhs)
        #generalJacobi vllt ersetzen durch "specificJacobi"

        # hier schleifenstart
        while newton_it > 0:
            Jv = generalJacobi(F, v)
            # r = Residium; prob.rhs = fj
            r = prob.rhs - F(v)



            # e = Fehler
            e = np.linalg.solve(Jv,r)
            v += e
            newton_it -= 1;


        return v
        
