import numpy as np
import scipy.sparse.linalg as spLA
import scipy.linalg as LA
import unittest

from project.mymultigrid import MyMultigrid
from project.poisson1d import Poisson1D
from project.linear_transfer2D import LinearTransfer2D
from project.weighted_jacobi import WeightedJacobi


class test_mymultigrid2D(unittest.TestCase):

    def setUp(self):
        ndofs = 31
        nlevels = int(np.log2(ndofs+1))
        self.prob = Poisson1D(ndofs=ndofs)

        self.mymg = MyMultigrid(ndofs, nlevels=nlevels)
        self.mymg.attach_transfer(LinearTransfer2D)
        self.mymg.attach_smoother(WeightedJacobi, self.prob.A, omega=2.0/3.0)

        k = 6
        xvalues = np.array([(i+1) * self.prob.dx for i in range(self.prob.ndofs)])
        self.u = np.sin(np.pi*k*xvalues)

        self.nu0 = 1
        self.nu1 = 2
        self.nu2 = 2

    def test_can_solve_homogeneous_problem_vcycle(self):
        res = 1
        u = self.u
        while res > 1E-10:
            u = self.mymg.do_v_cycle(u, self.prob.rhs, nu1=self.nu1, nu2=self.nu2, lstart=0)
            res = np.linalg.norm(self.prob.A.dot(u)-self.prob.rhs, np.inf)

        err = np.linalg.norm(u, np.inf)

        assert err < 1E-12, 'V-cycles do not bring solution down far enough'

    def test_converges_for_inhomogeneous_problem_vcycle(self):
        k = 6
        xvalues = np.array([(i+1) * self.prob.dx for i in range(self.prob.ndofs)])
        uex = np.sin(np.pi*k*xvalues)
        self.prob.rhs = (np.pi*k)**2 * uex

        u = np.zeros(self.prob.ndofs)
        for i in range(20):
            u = self.mymg.do_v_cycle(u, self.prob.rhs, nu1=self.nu1, nu2=self.nu2, lstart=0)
        res = np.linalg.norm(self.prob.A.dot(u)-self.prob.rhs, np.inf)

        assert res < 1E-12, 'V-cycles do not bring residual down far enough' + str(res)

    def test_recursion_is_equal_to_vcycle(self):
        u_rec = self.mymg.do_v_cycle_recursive(self.u, self.prob.rhs,
                                               nu1=self.nu1, nu2=self.nu2, level=0)
        u_old = self.mymg.do_v_cycle(self.u, self.prob.rhs, nu1=self.nu1, nu2=self.nu2, lstart=0)

        assert np.linalg.norm(u_old-u_rec, np.inf) == 0, np.linalg.norm(u_old-u_rec, np.inf)

    def test_exact_is_fixpoint_of_vcycle(self):
        k = 6
        xvalues = np.array([(i+1) * self.prob.dx for i in range(self.prob.ndofs)])
        self.prob.rhs = (np.pi*k)**2 * np.sin(np.pi*k*xvalues)
        uex = spLA.spsolve(self.prob.A, self.prob.rhs)

        u = uex
        for i in range(10):
            u = self.mymg.do_v_cycle(u, self.prob.rhs, nu1=self.nu1, nu2=self.nu2, lstart=0)

        err = np.linalg.norm(u - uex, np.inf)

        assert err <= 1E-14, 'Exact solution is not a fixpoint of the V-cycle iteration!' + str(err)
