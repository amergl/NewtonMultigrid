import numpy as np

from project.linear_transfer import LinearTransfer


def test_prolong_has_expected_order_of_accuracy():
    expected_order = 2

    k = 4
    ntests = 6

    ndofs = []
    err_list = []
    for i in range(ntests):
        ndofs.append(int(2 ** (i + 5) - 1))

        ndofs_fine = ndofs[-1]
        ndofs_coarse = int((ndofs_fine + 1) / 2 - 1)

        trans = LinearTransfer(ndofs_fine=ndofs_fine, ndofs_coarse=ndofs_coarse)

        dx_coarse = 1.0/(ndofs_coarse+1)
        x_coarse = np.array([(i + 1) * dx_coarse for i in range(ndofs_coarse)])
        u_coarse = np.sin(np.pi * k * x_coarse)
        dx_fine = 1.0/(ndofs_fine+1)
        x_fine = np.array([(i + 1) * dx_fine for i in range(ndofs_fine)])
        u_fine_exact = np.sin(np.pi * k * x_fine)

        u_fine_comp = trans.prolong(u_coarse)

        err_list.append(np.linalg.norm(u_fine_exact - u_fine_comp, np.inf) /
                        np.linalg.norm(u_fine_exact, np.inf))

    order = []
    for i in range(1, len(err_list)):
        order.append(np.log(err_list[i - 1] / err_list[i]) / np.log(ndofs[i] / ndofs[i - 1]))

    order = np.array(order)

    assert (order > expected_order * 0.9).all() and (order < expected_order * 1.1).all(), \
        'Order of accuracy of the prolongation is not ' + str(expected_order)


def test_restrict_has_expected_order_of_accuracy():
    expected_order = 2

    k = 4
    ntests = 6

    ndofs = []
    err_list = []
    for i in range(ntests):
        ndofs.append(int(2 ** (i + 5) - 1))

        ndofs_fine = ndofs[-1]
        ndofs_coarse = int((ndofs_fine + 1) / 2 - 1)

        trans = LinearTransfer(ndofs_fine=ndofs_fine, ndofs_coarse=ndofs_coarse)

        dx_coarse = 1.0/(ndofs_coarse+1)
        x_coarse = np.array([(i + 1) * dx_coarse for i in range(ndofs_coarse)])
        u_coarse_exact = np.sin(np.pi * k * x_coarse)
        dx_fine = 1.0/(ndofs_fine+1)
        x_fine = np.array([(i + 1) * dx_fine for i in range(ndofs_fine)])
        u_fine = np.sin(np.pi * k * x_fine)

        u_coarse_comp = trans.restrict(u_fine)

        err_list.append(np.linalg.norm(u_coarse_exact - u_coarse_comp, np.inf) /
                        np.linalg.norm(u_coarse_exact, np.inf))

    order = []
    for i in range(1, len(err_list)):
        order.append(np.log(err_list[i - 1] / err_list[i]) / np.log(ndofs[i] / ndofs[i - 1]))

    order = np.array(order)

    assert (order > expected_order * 0.9).all() and (order < expected_order * 1.1).all(), \
        'Order of accuracy of the restriction is not ' + str(expected_order)
