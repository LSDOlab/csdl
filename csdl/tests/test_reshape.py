import numpy as np
import pytest


def test_reshape_tensor2vector(backend):
    from csdl.examples.valid.ex_reshape_tensor2_vector import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    i = 2
    j = 3
    k = 4
    l = 5
    shape = (i, j, k, l)

    tensor = np.arange(np.prod(shape)).reshape(shape)
    vector = np.arange(np.prod(shape))

    # TENSOR TO VECTOR
    desired_output = vector
    np.testing.assert_almost_equal(sim['reshape_tensor2vector'],
                                   desired_output)

    partials_error = sim.check_partials(
        includes=['comp_reshape_tensor2vector'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_reshape_vector2tensor(backend):
    from csdl.examples.valid.ex_reshape_vector2_tensor import example
    exec('from {} import Simulator'.format(backend))
    sim, rep = example(eval('Simulator'))

    i = 2
    j = 3
    k = 4
    l = 5
    shape = (i, j, k, l)

    tensor = np.arange(np.prod(shape)).reshape(shape)
    vector = np.arange(np.prod(shape))

    # VECTOR TO TENSOR
    desired_output = tensor

    np.testing.assert_almost_equal(sim['reshape_vector2tensor'],
                                   desired_output)

    partials_error = sim.check_partials(
        includes=['comp_reshape_vector2tensor'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-5, rtol=1.e-5)
