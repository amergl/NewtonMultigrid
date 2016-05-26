# coding=utf-8
import numpy as np
import scipy.sparse as sp


class ProblemBase(object):
    """Base class for problems.

    Derive from this class to ensure consistent handling of problems throughout the code.

    """
    def __init__(self, ndofs, A, rhs, *args, **kwargs):
        """Initialization routine for a problem

        Args:
            ndofs (int): number of degrees of freedom
            A (scipy.sparse.csc_matrix): a sparse system matrix
            rhs (numpy.ndarray): right-hand side vector
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        # we use private attributes with getters and setters here to avoid overwriting by accident
        self._ndofs = None
        self._A = None
        self._rhs = None

        # now set the attributes using potential validation through the setters defined below
        self.ndofs = ndofs
        self.A = A
        self.rhs = rhs

    @property
    def ndofs(self):
        """int: Number of DOFs (degrees of freedom).
        """
        return self._ndofs

    @ndofs.setter
    def ndofs(self, ndofs):
        assert isinstance(ndofs, int), 'Please use only integer values for ndofs'
        assert ndofs > 0, 'Please use at least one DOF'
        self._ndofs = ndofs

    @property
    def A(self):
        """scipy.sparse.csc_matrix: System matrix A.
        """
        return self._A

    @A.setter
    def A(self, A):
        assert isinstance(A, sp.csc_matrix), 'Please use a matrix in the CSC format'
        self._A = A

    @property
    def rhs(self):
        """numpy.ndarray: Right-hand side (RHS) of the problem.
        """
        return self._rhs

    @rhs.setter
    def rhs(self, rhs):
        assert isinstance(rhs, np.ndarray), 'Please use only ndarrays as datatype'
        assert rhs.ndim == 1, 'Please use only 1-d arrays as RHS'
        assert rhs.shape[0] == self.ndofs, 'Please adhere to %i DOFs for the RHS' % self.ndofs
        self._rhs = rhs

    @property
    def u_exact(self, *args, **kwargs):
        """Dummy routine for a potential exact solution
        """
        return None
