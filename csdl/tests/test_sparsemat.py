import numpy as np
import pytest


def test_sparse_mat_mat(backend):
    from csdl.examples.valid.ex_sparsemat_mat_mat import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    np.testing.assert_almost_equal(
        sim['out'],
        np.array([
            [13., 13., 13., 13.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [2., 2., 2., 2.],
        ]))

    partials_error = sim.check_partials(includes=['out'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)
