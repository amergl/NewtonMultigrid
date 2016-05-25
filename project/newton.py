import scipy.sparse.linalg as sLA
import numpy as np

from pymg.multigrid_base import MultigridBase


class Newton(MultigridBase):
    """Implementation of a newton multigrid solver
    """

    def __init__(self, ndofs, nlevels):
        """Initialization routine
        """
        assert np.log2(ndofs+1) >= nlevels
        super(Newton, self).__init__(ndofs, nlevels)

    def jacobi(self, g, x, delta=1e-4):
        x = array(x, dtype=float64)
        fx = g(x)
        f = g
        m = 0
        if type(fx) == ndarray:
            m = fx.shape[0]
        else:
            m = 1

        if m == 1:
            f = lambda x: array(g(x))
            n = x.shape[0]
            J = zeros((m, n))

        for i in range(m):
            for j in range(n):
                xtilde = copy(x)
                xtilde[j] += delta
                J[i, j] += f(xtilde)[i]
                xtilde[j] -= 2*delta
                J[i, j] -= f(xtilde)[i]
                J[i, j] /= 2.0*delta
        return J
