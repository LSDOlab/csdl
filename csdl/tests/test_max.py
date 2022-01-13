import numpy as np
import pytest


def test_max_scalar(backend):
    from csdl.examples.valid.ex_max_scalar import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    m = 2
    n = 3
    o = 4
    p = 5
    q = 6

    tensor_shape = (m, n, o, p, q)
    num_of_elements = np.prod(tensor_shape)
    tensor = np.arange(num_of_elements).reshape(tensor_shape)

    # SCALAR MIN
    desired_output = np.max(tensor)
    np.testing.assert_almost_equal(sim['ScalarMax'], desired_output)
    assert sim['ScalarMax'].shape == (1, ), sim['ScalarMax'].shape

    partials_error = sim.check_partials(includes=['comp_ScalarMax'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_max_axiswise(backend):
    from csdl.examples.valid.ex_max_axiswise import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    m = 2
    n = 3
    o = 4
    p = 5
    q = 6

    tensor_shape = (m, n, o, p, q)
    num_of_elements = np.prod(tensor_shape)
    tensor = np.arange(num_of_elements).reshape(tensor_shape)

    # AXISWISE MIN
    desired_output = np.amax(tensor, axis=1)
    np.testing.assert_almost_equal(sim['AxiswiseMax'], desired_output)
    assert sim['AxiswiseMax'].shape == (m, o, p,
                                        q), sim['AxiswiseMax'].shape

    partials_error = sim.check_partials(includes=['comp_AxiswiseMax'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_max_elementwise(backend):
    from csdl.examples.valid.ex_max_elementwise import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    tensor1 = np.array([[1, 5, -8], [10, -3, -5]])
    tensor2 = np.array([[2, 6, 9], [-1, 2, 4]])

    desired_output = np.maximum(tensor1, tensor2)
    np.testing.assert_almost_equal(sim['ElementwiseMax'],
                                   desired_output)
    assert sim['ElementwiseMax'].shape == (
        2, 3), sim['ElementwiseMax'].shape

    partials_error = sim.check_partials(
        includes=['comp_ElementwiseMax'],
        out_stream=None,
        compact_print=True,
        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_max_multi_inputs_and_axis(backend):
    exec('from {} import Simulator'.format(backend))
    from csdl.examples.invalid.ex_max_multi_inputs_and_axis import example
    with pytest.raises(Exception):
        example(eval('Simulator'))


def test_max_inputs_not_same_size(backend):
    exec('from {} import Simulator'.format(backend))
    from csdl.examples.invalid.ex_max_inputs_not_same_size import example
    with pytest.raises(Exception):
        example(eval('Simulator'))


def test_max_scalar_random(backend):
    from csdl.examples.valid.ex_max_scalar_random import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    m = 2
    n = 3
    o = 4
    p = 5
    q = 6
    np.random.seed(0)

    tensor_shape = (m, n, o, p, q)
    num_of_elements = np.prod(tensor_shape)
    tensor = np.random.rand(num_of_elements).reshape(tensor_shape)

    # SCALAR MIN
    desired_output = np.max(tensor)
    np.testing.assert_almost_equal(sim['ScalarMax'], desired_output)
    assert sim['ScalarMax'].shape == (1, ), sim['ScalarMax'].shape

    partials_error = sim.check_partials(includes=['comp_ScalarMax'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_max_axiswise_random(backend):
    from csdl.examples.valid.ex_max_axiswise_random import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    m = 2
    n = 3
    o = 4
    p = 5
    q = 6
    np.random.seed(0)

    tensor_shape = (m, n, o, p, q)
    num_of_elements = np.prod(tensor_shape)
    tensor = np.random.rand(num_of_elements).reshape(tensor_shape)

    # AXISWISE MIN
    desired_output = np.amax(tensor, axis=1)
    np.testing.assert_almost_equal(sim['AxiswiseMax'], desired_output)

    partials_error = sim.check_partials(includes=['comp_AxiswiseMax'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)
    assert sim['AxiswiseMax'].shape == (m, o, p,
                                        q), sim['AxiswiseMax'].shape
