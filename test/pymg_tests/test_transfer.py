# coding=utf-8
import numpy as np
import nose

from pymg.transfer_base import TransferBase
from test.test_helpers import get_derived_from_in_package


list = []
ndofs_fine = None
ndofs_coarse = None


def setup():
    global list, ndofs_fine, ndofs_coarse
    list = get_derived_from_in_package(TransferBase, 'project')
    ndofs_fine = 15
    ndofs_coarse = 7


@nose.tools.with_setup(setup)
def test_is_derived_from_base_class():
    for t_class in list:
        yield is_derived_from_base_class, t_class


def is_derived_from_base_class(t_class):
    assert issubclass(t_class, TransferBase), 'Make sure you inherit from SmootherBaseClass'


@nose.tools.with_setup(setup)
def test_has_all_required_attributes():
    for t_class in list:
        trans = t_class(ndofs_fine=ndofs_fine, ndofs_coarse=ndofs_coarse)
        yield has_all_required_attributes, trans


def has_all_required_attributes(trans):
    assert hasattr(trans, 'prolong') and callable(getattr(trans, 'prolong')), \
        'Need prolong functionality'
    assert hasattr(trans, 'restrict') and callable(getattr(trans, 'restrict')), \
        'Need restrict functionality'


@nose.tools.with_setup(setup)
def test_can_call_prolong():
    for t_class in list:
        trans = t_class(ndofs_fine=ndofs_fine, ndofs_coarse=ndofs_coarse)
        yield can_call_prolong, trans, ndofs_coarse, ndofs_fine


def can_call_prolong(trans, ndofs_coarse, ndofs_fine):
    uc = np.ones(ndofs_coarse)
    uf = trans.prolong(uc)
    assert isinstance(uf, np.ndarray), 'Prolong does not produce ndarray'
    assert uf.size == ndofs_fine, 'Prolong does not return correct ndofs'


@nose.tools.with_setup(setup)
def test_can_call_restrict():
    for t_class in list:
        trans = t_class(ndofs_fine=ndofs_fine, ndofs_coarse=ndofs_coarse)
        yield can_call_restrict, trans, ndofs_fine, ndofs_coarse


def can_call_restrict(trans, ndofs_fine, ndofs_coarse):
    uf = np.ones(ndofs_fine)
    uc = trans.restrict(uf)
    assert isinstance(uf, np.ndarray), 'Restrict does not produce ndarray'
    assert uc.size == ndofs_coarse, 'Restrict does not return correct ndofs'
