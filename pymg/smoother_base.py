# coding=utf-8
import abc
from future.utils import with_metaclass
import scipy.sparse as sp


class SmootherBase(with_metaclass(abc.ABCMeta)):
    """Base class for smoothers

    Derive from this class to ensure consistent handling of smoothers throughout the code.

    """
    def __init__(self, A, *args, **kwargs):
        """Initialization routine for a smoother

        Args:
            A (scipy.sparse.csc_matrix): sparse matrix A of the system to solve
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        # we use private attributes with getters and setters here to avoid overwriting by accident
        self._A = None

        # now set the attributes using potential validation through the setters defined below
        self.A = A

    @abc.abstractmethod
    def smooth(self, rhs, u_old):
        """Abstract method to be overwritten by implementation
        """
        pass

    @property
    def A(self):
        """scipy.sparse.csc_matrix: system matrix A
        """
        return self._A

    @A.setter
    def A(self, A):
        assert isinstance(A, sp.csc_matrix), 'Please use a matrix in the CSC format'
        self._A = A
