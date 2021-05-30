import numpy as np
import pytest


def test_matrix_transpose(backend):
    from csdl.examples.valid.ex_transpose_matrix import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    # MATRIX TRANSPOSE
    val = np.arange(4 * 2).reshape((4, 2))
    desired_output = np.transpose(val)

    np.testing.assert_almost_equal(sim['matrix_transpose'], desired_output)

    partials_error = sim.check_partials(includes=['comp_matrix_transpose'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_tensor_transpose(backend):
    from csdl.examples.valid.ex_transpose_tensor import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    # TENSOR TRANSPOSE
    val = np.arange(4 * 3 * 5 * 2).reshape((4, 3, 5, 2))
    desired_output = np.transpose(val)

    np.testing.assert_almost_equal(sim['tensor_transpose'], desired_output)

    partials_error = sim.check_partials(includes=['comp_tensor_transpose'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)
