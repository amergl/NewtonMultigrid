# coding=utf-8
import numpy as np
import scipy.sparse.linalg as spLA
import nose

from project.poisson1d import Poisson1D
from pymg.smoother_base import SmootherBase
from test.test_helpers import get_derived_from_in_package


list = []
ndofs = None
prob = None


def setup():
    global list, ndofs, prob
    list = get_derived_from_in_package(SmootherBase, 'project')
    ndofs = 10
    prob = Poisson1D(ndofs)


@nose.tools.with_setup(setup)
def test_is_derived_from_base_class():
    for smoo_class in list:
        yield is_derived_from_base_class, smoo_class


def is_derived_from_base_class(smoo_class):
    assert issubclass(smoo_class, SmootherBase), 'Make sure you inherit from SmootherBaseClass'


@nose.tools.with_setup(setup)
def test_has_all_required_attributes():
    for smoo_class in list:
        smoo = smoo_class(prob.A, 2.0/3.0)
        yield has_all_required_attributes, smoo


def has_all_required_attributes(smoo):
    assert hasattr(smoo, 'A'), 'Need to specify system matrix A'
    assert hasattr(smoo, 'smooth') and callable(getattr(smoo, 'smooth')), \
        'Need smooth functionality'


@nose.tools.with_setup(setup)
def test_can_call_smoothing():
    for smoo_class in list:
        smoo = smoo_class(prob.A, 2.0/3.0)
        yield can_call_smoothing, smoo, ndofs


def can_call_smoothing(smoo, ndofs):
    rhs = np.ones(ndofs)
    u = smoo.smooth(rhs=rhs, u_old=rhs)
    assert isinstance(u, np.ndarray), 'Something is wrong in do_smoothing()'


@nose.tools.with_setup(setup)
def test_exact_is_fixpoint():
    for smoo_class in list:
        yield exact_is_fixpoint, smoo_class


def exact_is_fixpoint(smoo_class):
    ndofs = 127
    prob = Poisson1D(ndofs)
    smoo = smoo_class(prob.A, 2.0/3.0)
    k = 6

    xvalues = np.array([(i + 1) * prob.dx for i in range(prob.ndofs)])
    rhs = np.sin(np.pi * k * xvalues)
    uex = spLA.spsolve(smoo.A, rhs)

    u = uex
    for i in range(10):
        u = smoo.smooth(rhs, u)

    err = np.linalg.norm(u - uex, np.inf)

    assert err <= 1E-14, 'Exact solution is not a fixpoint of the iteration!'
