import scipy.sparse.linalg as sLA
import numpy as np

#from pymg.multigrid_base import MultigridBase
from project.jacobi import generalJacobi,specificJacobi

class Newton():#MultigridBase):
    """Implementation of a newton multigrid solver
    """

    def __init__(self, ndofs, nlevels):
        """Initialization routine
        """
        assert np.log2(ndofs+1) >= nlevels
        #super(Newton, self).__init__(ndofs, nlevels)
        self.ndofs=ndofs


    def do_newton_cycle(self, prob, nu0, nu1, newton_it=10):
        #attach prolongation operator
        #attach restriction operator
        #attach smoother
        x=np.copy(prob.rhs)
        F=lambda z: np.array(prob.rhs) - prob.A(z)
        while newton_it > 0:
            J=generalJacobi(F,x)
            x-= np.dot(np.linalg.inv(J),F(x))
            newton_it -=1
        return x
        
