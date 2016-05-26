import numpy as np

from project.poisson1d import Poisson1D


def test_has_spatial_order_of_accuracy():
    expected_order = 2

    k = 4
    ntests = 6

    ndofs = []
    err_list = []
    for i in range(ntests):
        ndofs.append(2 ** (i + 4) - 1)

        prob = Poisson1D(ndofs[-1])

        xvalues = np.array([(i + 1) * prob.dx for i in range(prob.ndofs)])
        uinit = np.sin(np.pi * k * xvalues)
        uexact = (np.pi * k) ** 2 * uinit
        ucomp = prob.A.dot(uinit)

        err_list.append(np.linalg.norm(uexact - ucomp, np.inf) / np.linalg.norm(uexact, np.inf))

    order = []
    for i in range(1, len(err_list)):
        order.append(np.log(err_list[i - 1] / err_list[i]) / np.log(ndofs[i] / ndofs[i - 1]))

    order = np.array(order)

    assert (order > expected_order * 0.9).all() and (order < expected_order * 1.1).all(), \
        'Order of accuracy of the spatial discretization is not ' + str(expected_order)
