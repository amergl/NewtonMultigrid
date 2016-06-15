import abc
from future.utils import with_metaclass
import numpy as np


class MultigridBase(with_metaclass(abc.ABCMeta)):
    """Base class for multigrid cycles

    This mainly includes the data structure required to cycle through the levels:
    the vectors vh and fh, a list of smoothers (i.e. the correct matrices) and
    a list of transfer operators

    Attributes:
        nlevels (int): number of levels in the MG hierarchy
        vh (list of numpy.ndarrays): data structure for the solution vectors
        fh (list of numpy.ndarrays): data structure for the rhs vectors
        trans (list of :class:`pymg.transfer_base.TransferBase`): list of transfer operators
        smoo (list of :class:`pymg.smoother_base.SmootherBase`): list of smoothers
        Acoarse (scipy.sparse.csc_matrix): system matrix on the coarsest level
    """

    def __init__(self, ndofs, nlevels, *args, **kwargs):
        """Initialization routine for a multigrid solver

        Note:
            instantiation of smoothers and transfer operators is separated to allow
            passing parameters more easily

        Args:
            ndofs (int): number of degrees of freedom (see
                :attr:`pymg.problem_base.ProblemBase.ndofs`)
            nlevels (int): number of levels in the hierarchy
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        assert isinstance(nlevels, int)
        assert 0 <= nlevels <= np.log2(ndofs+1)
        
        self.nlevels = nlevels
        
        self._ndofs_list = [int((ndofs + 1) / 2**l) - 1 for l in range(nlevels)]

        self.vh = [np.zeros(ndofs_h**2) for ndofs_h in self._ndofs_list]
        self.fh = [np.zeros(ndofs_h**2) for ndofs_h in self._ndofs_list]

        self.trans = []
        self.smoo = []
        self.Acoarse = None

    def reset_vectors(self, lstart):
        """Routine to (re)set the solution and rhs vectors to zero

        Args:
            lstart (int): level to start from (all below will be set to zero)
        """
        self.vh[lstart:] = [np.zeros(ndofs_h**2) for ndofs_h in self._ndofs_list[lstart:]]
        self.fh[lstart:] = [np.zeros(ndofs_h**2) for ndofs_h in self._ndofs_list[lstart:]]

    def attach_smoother(self, smoother_class, A, *args, **kwargs):
        """Routine to attach a smoother to each level (except for the coarsest)

        Args:
            smoother_class (see :class:`pymg.smoother_base.SmootherBase`): the class of smoothers
            A (scipy.sparse.csc_matrix): system matrix of the problem
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        # for the Galerkin approach: check if transfer operators are attached
        assert len(self.trans) == self.nlevels - 1
        # work your way through the hierarchy
        self.Acoarse = A
        for l in range(0, self.nlevels-1):
            self.smoo.append(smoother_class(self.Acoarse, *args, **kwargs))
            # here comes Galerkin
            self.Acoarse = self.trans[l].I_hto2h.dot(self.Acoarse.dot(self.trans[l].I_2htoh))
        # in case we want to do smoothing instead of solving on the coarsest level:
        self.smoo.append(smoother_class(self.Acoarse, *args, **kwargs))

    def attach_transfer(self, transfer_class, *args, **kwargs):
        """Routine to attach transfer operators to each level (except for the coarsest)

        Args:
            transfer_class (see :class:`pymg.transfer_base.TransferBase`): the class of transfer ops
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        for l in range(self.nlevels-1):
            self.trans.append(transfer_class(ndofs_fine=self._ndofs_list[l],
                                             ndofs_coarse=self._ndofs_list[l+1], *args, **kwargs))
