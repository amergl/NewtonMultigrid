import numpy as np
import scipy.sparse as sp
from pymg.transfer_base import TransferBase


class LinearTransfer(TransferBase):
    """Implementation of the linear prolongation and restriction operators

    Attributes:
        I_2htoh (scipy.sparse.csc_matrix): prolongation matrix
        I_hto2h (scipy.sparse.csc_matrix): restriction matrix
    """

    def __init__(self, ndofs_fine, ndofs_coarse, *args, **kwargs):
        """Initialization routine for transfer operators

        Args:
            ndofs_fine (int): number of DOFs on the fine grid
            ndofs_coarse (int): number of DOFs on the coarse grid
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        # for this particular transfer class, we need to make a few assumptions
        assert isinstance(ndofs_fine, int), type(ndofs_fine)
        assert isinstance(ndofs_coarse, int)
        assert (ndofs_fine + 1) % 2 == 0
        assert ndofs_coarse == (ndofs_fine + 1) / 2 - 1

        super(LinearTransfer, self).__init__(ndofs_fine, ndofs_coarse, *args, **kwargs)

        # pre-compute prolongation and restriction matrices
        self.I_2htoh = self.__get_prolongation_matrix(ndofs_coarse, ndofs_fine)
        self.I_hto2h = self.__get_restriction_matrix()

    @staticmethod
    def __get_prolongation_matrix(ndofs_coarse, ndofs_fine):
        """Helper routine for the prolongation operator

        Args:
            ndofs_fine (int): number of DOFs on the fine grid
            ndofs_coarse (int): number of DOFs on the coarse grid

        Returns:
            scipy.sparse.csc_matrix: sparse prolongation matrix of size
                `ndofs_fine` x `ndofs_coarse`
        """

        # This is a workaround, since I am not aware of a suitable way to do
        # this directly with sparse matrices.
        P = np.zeros((ndofs_fine, ndofs_coarse))
        np.fill_diagonal(P[1::2, :], 1)
        np.fill_diagonal(P[0::2, :], 1.0/2.0)
        np.fill_diagonal(P[2::2, :], 1.0/2.0)
        return sp.csc_matrix(P)

    def __get_restriction_matrix(self):
        """Helper routine for the restriction operator

        Returns:
           scipy.sparse.csc_matrix: sparse restriction matrix of size
                `ndofs_coarse` x `ndofs_fine`
        """
        assert hasattr(self, 'I_2htoh')
        return 0.5 * sp.csc_matrix(self.I_2htoh.T)

    def restrict(self, u_coarse):
        """Routine to apply restriction

        Args:
            u_coarse (numpy.ndarray): vector on coarse grid, size `ndofs_coarse`
        Returns:
            numpy.ndarray: vector on fine grid, size `ndofs_fine`
        """
        return self.I_hto2h.dot(u_coarse)

    def prolong(self, u_fine):
        """Routine to apply prolongation

        Args:
            u_fine (numpy.ndarray): vector on fine grid, size `ndofs_fine`
        Returns:
            numpy.ndarray: vector on coarse grid, size `ndofs_coarse`
        """
        return self.I_2htoh.dot(u_fine)
