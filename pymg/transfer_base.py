# coding=utf-8
import abc
from future.utils import with_metaclass


class TransferBase(with_metaclass(abc.ABCMeta)):
    """Base class for restriction and prolongation operators

    Derive from this class to ensure consistent handling of transfers throughout the code.

    """
    def __init__(self, ndofs_fine, ndofs_coarse, *args, **kwargs):
        """Initialization routine for transfer operators

        Args:
            ndofs_fine (int): number of DOFs on the fine grid
            ndofs_coarse (int): number of DOFs on the coarse grid
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        self.ndofs_fine = ndofs_fine
        self.ndofs_coarse = ndofs_coarse

    @abc.abstractmethod
    def restrict(self, u_fine):
        """Abstract restriction method to be overwritten by implementation
        """
        pass

    @abc.abstractmethod
    def prolong(self, u_coarse):
        """Abstract prolongation method to be overwritten by implementation
        """
        pass
