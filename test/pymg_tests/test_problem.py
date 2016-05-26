# coding=utf-8
import numpy as np
import nose

from pymg.problem_base import ProblemBase
from test.test_helpers import get_derived_from_in_package

list = []
ndofs = None


def setup():
    global list, ndofs
    list = get_derived_from_in_package(ProblemBase, 'project')
    ndofs = 10


@nose.tools.with_setup(setup)
def test_is_derived_from_base_class():
    for p_class in list:
        yield is_derived_from_base_class, p_class


def is_derived_from_base_class(p_class):
    assert issubclass(p_class, ProblemBase), 'Make sure you inherit from ProblemBaseClass'


@nose.tools.with_setup(setup)
def test_has_all_required_attributes():
    for p_class in list:
        prob = p_class(ndofs=ndofs)
        yield has_all_required_attributes, prob


def has_all_required_attributes(prob):
    assert hasattr(prob, 'A'), 'Need system matrix A'
    assert hasattr(prob, 'rhs'), 'Need RHS vector'
    assert hasattr(prob, 'ndofs'), 'Need number of DOFs'


@nose.tools.with_setup(setup)
def test_can_do_basic_operations():
    for p_class in list:
        prob = p_class(ndofs=ndofs)
        yield can_do_basic_operations, prob, ndofs


def can_do_basic_operations(prob, ndofs):
    u = np.ones(ndofs)

    prob.rhs = np.ones(ndofs)

    res = prob.A.dot(u) - prob.A.dot(prob.rhs)

    if prob.u_exact is not None:
        assert isinstance(prob.u_exact, np.ndarray), 'function u_exact does not return an ndarray'
        assert prob.u_exact.size == ndofs, 'u_exact does not have the correct size'

    assert (res == 0).all(), 'The residual should be 0 here...'
