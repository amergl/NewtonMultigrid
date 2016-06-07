import numpy as np
import scipy.sparse as sp
import math
from pymg.transfer_base import TransferBase


class LinearTransfer2D(TransferBase):
    """Implementation of the linear prolongation and restriction operators for 2-dimensional grids

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
        assert math.sqrt(ndofs_coarse) == (math.sqrt(ndofs_fine) + 1) / 2 - 1

        super(LinearTransfer2D, self).__init__(ndofs_fine, ndofs_coarse, *args, **kwargs)

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

        n_coarse = int(math.sqrt(ndofs_coarse))
        n_fine = int(math.sqrt(ndofs_fine))

        # This is a workaround, since I am not aware of a suitable way to do
        # this directly with sparse matrices.
        P = np.zeros((ndofs_fine, ndofs_coarse))

        for line_block in range(n_fine):
            if (line_block % 2 == 1):
                np.fill_diagonal(P[line_block * n_fine + 1:line_block * n_fine + 5:2,
                                 line_block / 2 * n_coarse:], 1.0)
                np.fill_diagonal(P[line_block * n_fine + 0:line_block * n_fine + 5:2,
                                 line_block / 2 * n_coarse:], 1.0 / 2.0)
                np.fill_diagonal(P[line_block * n_fine + 2:line_block * n_fine + 5:2,
                                 line_block / 2 * n_coarse:], 1.0 / 2.0)

            else:
                np.fill_diagonal(P[line_block * n_fine + 1:line_block * n_fine + 5:2,
                                 (line_block / 2 - 1) * n_coarse:], 1.0 / 2.0)
                np.fill_diagonal(P[line_block * n_fine + 0:line_block * n_fine + 5:2,
                                 (line_block / 2 - 1) * n_coarse:], 1.0 / 4.0)
                np.fill_diagonal(P[line_block * n_fine + 2:line_block * n_fine + 5:2,
                                 (line_block / 2 - 1) * n_coarse:], 1.0 / 4.0)

                np.fill_diagonal(P[line_block * n_fine + 1:line_block * n_fine + 5:2,
                                 (line_block / 2) * n_coarse:], 1.0 / 2.0)
                np.fill_diagonal(P[line_block * n_fine + 0:line_block * n_fine + 5:2,
                                 (line_block / 2) * n_coarse:], 1.0 / 4.0)
                np.fill_diagonal(P[line_block * n_fine + 2:line_block * n_fine + 5:2,
                                 (line_block / 2) * n_coarse:], 1.0 / 4.0)

        return sp.csc_matrix(P)

    def __get_restriction_matrix(self):
        """Helper routine for the restriction operator

        Returns:
           scipy.sparse.csc_matrix: sparse restriction matrix of size
                `ndofs_coarse` x `ndofs_fine`
        """
        data = [ [1.0/16]*self.ndofs_fine, [1.0/8]*self.ndofs_fine, [1.0/16]*self.ndofs_fine,
                 [1.0/8] *self.ndofs_fine, [1.0/4]*self.ndofs_fine, [1.0/8] *self.ndofs_fine,
                 [1.0/16]*self.ndofs_fine, [1.0/8]*self.ndofs_fine, [1.0/16]*self.ndofs_fine ]
        diags = [ -self.ndofs_fine - 1, -self.ndofs_fine, -self.ndofs_fine + 1,
                 -1, 0, 1,
                 self.ndofs_fine - 1, self.ndofs_fine, self.ndofs_fine +1 ]
        
        # matrix nxn
        big_matrix = sp.spdiags( data, diags, self.ndofs_fine-1, self.ndofs_fine, format='csc' )
        # matrix containing only every second row
	#print big_matrix.shape, big_matrix[::2,:].shape
        return big_matrix[::2,:]

    def restrict(self, u_coarse):
        """Routine to apply restriction

        Args:
            u_coarse (numpy.ndarray): vector on coarse grid, size `ndofs_coarse`
        Returns:
            numpy.ndarray: vector on fine grid, size `ndofs_fine`
        """
        print self.I_hto2h.shape, u_coarse.shape
        return self.I_hto2h.dot(u_coarse)

    def prolong(self, u_fine):
        """Routine to apply prolongation

        Args:
            u_fine (numpy.ndarray): vector on fine grid, size `ndofs_fine`
        Returns:
            numpy.ndarray: vector on coarse grid, size `ndofs_coarse`
        """
        return self.I_2htoh.dot(u_fine)
