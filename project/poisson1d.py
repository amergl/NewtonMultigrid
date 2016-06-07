# coding=utf-8
import numpy as np
import scipy.sparse as sp

from pymg.problem_base import ProblemBase


class Poisson1D(ProblemBase):
    """Implementation of the 1D Poission problem.

    Here we define the 1D Poisson problem :math:`-\Delta u = 0` with
    Dirichlet-Zero boundary conditions. This is the homogeneous problem,
    derive from this class if you want to play around with different RHS.

    Attributes:
        dx (float): mesh size
    """
    def __init__(self, ndofs, *args, **kwargs):
        """Initialization routine for the Poisson1D problem

        Args:
            ndofs (int): number of degrees of freedom (see
                :attr:`pymg.problem_base.ProblemBase.ndofs`)
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        self.dx = 1.0 / (ndofs + 1)
        # compute system matrix A, scale by 1/dx^2
        A = 1.0 / (self.dx ** 2) * self.__get_system_matrix(ndofs)
        rhs = self.__get_rhs(ndofs)

        super(Poisson1D, self).__init__(ndofs, A, rhs, *args, **kwargs)

    @staticmethod
    def __get_system_matrix(ndofs):
        """Helper routine to get the system matrix discretizing :math:`-Delta` with second order FD

        Args:
            ndofs (int): number of inner grid points (no boundaries!)
        Returns:
            scipy.sparse.csc_matrix: sparse system matrix A
                of size :attr:`ndofs` x :attr:`ndofs`
        """
        data = np.array([[2]*ndofs, [-1]*ndofs, [-1]*ndofs])
        diags = np.array([0, -1, 1])
        return sp.spdiags(data, diags, ndofs, ndofs, format='csc')

    @staticmethod
    def __get_rhs(ndofs):
        """Helper routine to set the right-hand side

        Args:
            ndofs (int): number of inner grid points (no boundaries!)
        Returns:
            numpy.ndarray: the right-hand side vector of size :attr:`ndofs`
        """
        return np.ones(ndofs)

    @property
    def u_exact(self):
        """Routine to compute the exact solution

        Returns:
            numpy.ndarray: exact solution array of size :attr:`ndofs`
        """

        t=np.linspace(0,1,self.ndofs+2)[1:-1]
        u=lambda x: np.array([-0.5*x[i]*x[i] + 0.5*x[i] for i in range(x.shape[0])],dtype=np.float64)
        return u(t)
