import numpy as np
import pytest


def test_reorder_axes_matrix(backend):
    from csdl.examples.valid.ex_reorder_axes_matrix import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    val = np.arange(4 * 2).reshape((4, 2))
    desired_output = np.transpose(val)

    np.testing.assert_almost_equal(sim['axes_reordered_matrix'],
                                   desired_output)

    partials_error = sim.check_partials(
        includes=['comp_axes_reordered_matrix'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_reorder_axes_tensor(backend):
    from csdl.examples.valid.ex_reorder_axes_tensor import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    val = np.arange(4 * 3 * 5 * 2).reshape((4, 3, 5, 2))
    desired_output = np.transpose(val, [3, 1, 2, 0])

    np.testing.assert_almost_equal(sim['axes_reordered_tensor'],
                                   desired_output)

    partials_error = sim.check_partials(
        includes=['comp_axes_reordered_tensor'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)
