import numpy as np
import pytest


def test_max_scalar(name):
    from csdl.examples.valid.ex_max_scalar import example
    exec('from {} import Simulator'.format(name))
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
    np.testing.assert_almost_equal(sim['ScalarMin'], desired_output)

    partials_error = sim.check_partials(includes=['comp_ScalarMin'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_max_axiswise(name):
    from csdl.examples.valid.ex_max_axiswise import example
    exec('from {} import Simulator'.format(name))
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
    np.testing.assert_almost_equal(sim['AxiswiseMin'], desired_output)

    partials_error = sim.check_partials(includes=['comp_AxiswiseMin'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_max_elementwise(name):
    from csdl.examples.valid.ex_max_elementwise import example
    exec('from {} import Simulator'.format(name))
    sim = example(eval('Simulator'))

    tensor1 = np.array([[1, 5, -8], [10, -3, -5]])
    tensor2 = np.array([[2, 6, 9], [-1, 2, 4]])

    desired_output = np.maximum(tensor1, tensor2)
    np.testing.assert_almost_equal(sim['ElementwiseMin'], desired_output)

    partials_error = sim.check_partials(includes=['comp_ElementwiseMin'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_max_multi_inputs_and_axis(name):
    with pytest.raises(Exception):
        from csdl.examples.invalid.ex_max_multi_inputs_and_axis import example
    exec('from {} import Simulator'.format(name))
    sim = example(eval('Simulator'))


def test_max_inputs_not_same_size(name):
    with pytest.raises(Exception):
        from csdl.examples.invalid.ex_max_inputs_not_same_size import example
    exec('from {} import Simulator'.format(name))
    sim = example(eval('Simulator'))
