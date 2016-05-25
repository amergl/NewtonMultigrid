import scipy.sparse.linalg as sLA
import numpy as np

from pymg.multigrid_base import MultigridBase


class MyMultigrid(MultigridBase):
    """Implementation of a multigrid solver with different cycle implementations
    """

    def __init__(self, ndofs, nlevels):
        """Initialization routine
        """
        assert np.log2(ndofs+1) >= nlevels
        super(MyMultigrid, self).__init__(ndofs, nlevels)

    def do_v_cycle(self, v0, rhs, nu1, nu2, lstart):
        """Straightforward implementation of a V-cycle

        This can also be used inside an FMG-cycle!

        Args:
            v0 (numpy.array): initial values on finest level
            rhs (numpy.array): right-hand side on finest level
            nu1 (int): number of downward smoothing steps
            nu2 (int): number of upward smoothing steps
            lstart (int): starting level

        Returns:
            numpy.array: solution vector on finest level
        """

        assert self.nlevels >= lstart >= 0
        assert v0.size == self.vh[lstart].size

        # set intial conditions (note: resetting vectors here is important!)
        self.reset_vectors(lstart)
        self.vh[lstart] = v0
        self.fh[lstart] = rhs

        # downward cycle
        for l in range(lstart, self.nlevels-1):
            # print('V-down: %i -> %i' %(l,l+1))
            # pre-smoothing
            for i in range(nu1):
                self.vh[l] = self.smoo[l].smooth(self.fh[l], self.vh[l])

            # restrict
            self.fh[l+1] = self.trans[l].restrict(self.fh[l] - self.smoo[l].A.dot(self.vh[l]))

        # solve on coarsest level
        self.vh[-1] = sLA.spsolve(self.Acoarse, self.fh[-1])

        # upward cycle
        for l in reversed(range(lstart, self.nlevels-1)):
            # print('V-up: %i -> %i' %(l+1,l))
            # correct
            self.vh[l] += self.trans[l].prolong(self.vh[l+1])

            # post-smoothing
            for i in range(nu2):
                self.vh[l] = self.smoo[l].smooth(self.fh[l], self.vh[l])

        return self.vh[lstart]

    def do_v_cycle_recursive(self, v0, rhs, nu1, nu2, level):
        """Recursive implementation of a V-cycle

        This can also be used inside an FMG-cycle!

        Args:
            v0 (numpy.array): initial values on finest level
            rhs (numpy.array): right-hand side on finest level
            nu1 (int): number of downward smoothing steps
            nu2 (int): number of upward smoothing steps
            level (int): current level

        Returns:
            numpy.array: solution vector on current level
        """

        assert self.nlevels > level >= 0
        assert v0.size == self.vh[level].size

        # set intial conditions
        self.vh[level] = v0
        self.fh[level] = rhs

        # downward cycle
        if level < self.nlevels-1:

            # pre-smoothing
            for i in range(nu1):
                self.vh[level] = self.smoo[level].smooth(self.fh[level], self.vh[level])

            # restrict
            self.fh[level+1] = self.trans[level].restrict(self.fh[level] -
                                                          self.smoo[level].A.dot(self.vh[level]))
            # recursive call to v-cycle
            self.vh[level+1] = self.do_v_cycle_recursive(np.zeros(self.vh[level+1].size),
                                                         self.fh[level+1], nu1, nu2, level+1)
            # on coarsest level
        else:

            # solve on coarsest level
            self.vh[level] = sLA.spsolve(self.Acoarse, self.fh[level])

            return self.vh[level]

        # correct
        self.vh[level] += self.trans[level].prolong(self.vh[level+1])

        # post-smoothing
        for i in range(nu2):
            self.vh[level] = self.smoo[level].smooth(self.fh[level], self.vh[level])

        return self.vh[level]

    def do_fmg_cycle(self, rhs, nu0, nu1, nu2, level):
        """Implementation of a FMG-cycle

        Args:
            v0 (numpy.array): initial values on finest level
            rhs (numpy.array): right-hand side on finest level
            nu1 (int): number of downward smoothing steps
            nu2 (int): number of upward smoothing steps
            level (int): current level

        Returns:
            numpy.array: solution vector on current level
        """

        self.fh[level] = rhs
        for i in range(level+1, self.nlevels):
            self.fh[i] = self.trans[i-1].restrict(self.fh[i-1] -
                                                  self.smoo[i-1].A.dot(self.vh[i-1]))

        self.vh[self.nlevels-2] += self.trans[self.nlevels-1].prolong(
            np.zeros(self.vh[self.nlevels-1]))
        for i in range(self.nlevels-2, level):
            do_v_cycle_recursive(self, self.vh[i], self.fh[i], nu1, nu2, i)

        return self.vh[0]
