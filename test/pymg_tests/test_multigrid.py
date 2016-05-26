import numpy as np
import scipy.sparse as sp
import nose

from pymg.multigrid_base import MultigridBase
from test.test_helpers import get_derived_from_in_package

from project.linear_transfer import LinearTransfer
from project.weighted_jacobi import WeightedJacobi
from project.poisson1d import Poisson1D


list = []
ndofs = None
nlevels = None


def setup():
    global list, ndofs, nlevels
    list = get_derived_from_in_package(MultigridBase, 'project')
    ndofs = 15
    nlevels = 4


@nose.tools.with_setup(setup)
def test_is_derived_from_base_class():
    for m_class in list:
        yield is_derived_from_base_class, m_class


def is_derived_from_base_class(p_class):
    assert issubclass(p_class, MultigridBase), 'Make sure you inherit from ProblemBaseClass'


@nose.tools.with_setup(setup)
def test_reset_vectors_works():
    for m_class in list:
        mult = m_class(ndofs, nlevels)
        yield reset_vectors_works, mult


def reset_vectors_works(mult):
    mult.vh[0] = np.eye(ndofs)
    mult.fh[0] = np.eye(ndofs)
    mult.reset_vectors(0)
    assert (mult.vh[0] == 0).all() and (mult.fh[0] == 0).all(), \
        'reset_vectors does not seem to work'


@nose.tools.with_setup(setup)
def test_attach_transfer_works():
    for m_class in list:
        mult = m_class(ndofs, nlevels)
        yield attach_transfer_works, mult


def attach_transfer_works(mult):
    mult.attach_transfer(LinearTransfer)
    assert all([isinstance(mult.trans[l], LinearTransfer) for l in range(nlevels-1)]), \
        'attach_transfer does not work'


@nose.tools.with_setup(setup)
def test_attach_smoother_works():
    for m_class in list:
        mult = m_class(ndofs, nlevels)
        yield attach_smoother_works, mult


def attach_smoother_works(mult):
    prob = Poisson1D(ndofs=ndofs)
    mult.attach_transfer(LinearTransfer)
    mult.attach_smoother(WeightedJacobi, prob.A, omega=2.0/3.0)
    assert isinstance(mult.Acoarse, sp.csc_matrix), type(mult.Acoarse)
