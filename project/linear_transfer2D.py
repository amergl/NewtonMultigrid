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
        #assert math.sqrt(ndofs_coarse) == (math.sqrt(ndofs_fine) + 1) / 2 - 1

        super(LinearTransfer2D, self).__init__(ndofs_fine, ndofs_coarse, *args, **kwargs)

        # pre-compute prolongation and restriction matrices
        self.I_2htoh = sp.csc_matrix(self.__get_prolongation_matrix(ndofs_coarse, ndofs_fine))
        self.I_hto2h = sp.csc_matrix(self.I_2htoh.T*0.25)

    @staticmethod
    def __get_prolongation_matrix(n_coarse, n_fine):
        """Helper routine for the prolongation operator

        Args:
            ndofs_fine (int): number of DOFs on the fine grid
            ndofs_coarse (int): number of DOFs on the coarse grid

        Returns:
            scipy.sparse.csc_matrix: sparse prolongation matrix of size
                `ndofs_fine` x `ndofs_coarse`
        """

<<<<<<< HEAD
        ndofs_coarse = n_coarse**2
        ndofs_fine = n_fine**2

        P = sp.dok_matrix((ndofs_fine, ndofs_coarse), dtype=float)

        block1 = np.zeros((n_fine, n_coarse))
        np.fill_diagonal(block1[1::2, :], 1.0)
        np.fill_diagonal(block1[0::2, :], 1.0 / 2.0)
        np.fill_diagonal(block1[2::2, :], 1.0 / 2.0)
        block1 = sp.csc_matrix(block1)

        block2 = np.zeros((n_fine, n_coarse))
        np.fill_diagonal(block2[1::2, :], 1.0 / 2.0)
        np.fill_diagonal(block2[0::2, :], 1.0 / 4.0)
        np.fill_diagonal(block2[2::2, :], 1.0 / 4.0)

        block2 = sp.csc_matrix(block2)
=======
        n_coarse = ndofs_coarse #int(math.sqrt(ndofs_coarse))
        n_fine = ndofs_fine #int(math.sqrt(ndofs_fine))

        # This is a workaround, since I am not aware of a suitable way to do
        # this directly with sparse matrices.
        P = np.zeros((ndofs_fine**2, ndofs_coarse**2))
>>>>>>> 7e4d6cb679c4b06538df64003304fac7c1f6b098

        for line_block in range(n_fine):
            if (line_block % 2 == 1):
                for line in range(n_fine):
                    for column in range(n_coarse):
                        P[line_block * n_fine + line, (line_block - 1) / 2 * n_coarse + column] = block1[line, column]

            else:
                for line in range(n_fine):
                    for column in range(n_coarse):
                        if (line_block == 0):
                            P[line_block * n_fine + line, line_block / 2 * n_coarse + column] = block2[line, column]

                        elif (line_block == n_fine -1):
                            P[line_block * n_fine + line, (line_block / 2 - 1) * n_coarse + column] = block2[line, column]

                        else:
                            P[line_block * n_fine + line, (line_block / 2 - 1) * n_coarse + column] = block2[line, column]
                            P[line_block * n_fine + line, (line_block / 2) * n_coarse + column] = block2[line, column]

        #print P.todense()
        return P.tocsc()

    def __get_restriction_matrix(self):
        """Helper routine for the restriction operator

        Returns:
           scipy.sparse.csc_matrix: sparse restriction matrix of size
                `ndofs_coarse` x `ndofs_fine`
        """

        return (1./4 * self.I_2htoh.transpose()).tocsc()


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

if __name__ == "__main__":
    ndofs_fine = 7
    ndofs_coarse = 3
    lintans2D = LinearTransfer2D(ndofs_fine, ndofs_coarse)
    print lintans2D.I_2htoh.todense()
    print lintans2D.I_hto2h.todense()

