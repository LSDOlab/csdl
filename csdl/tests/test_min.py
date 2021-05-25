from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_min_scalar(name):
    from csdl.examples.valid.ex_min_scalar import example
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
    desired_output = np.min(tensor)
    np.testing.assert_almost_equal(sim['ScalarMin'], desired_output)

    partials_error = sim.check_partials(includes=['comp_ScalarMin'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_min_axiswise(name):
    from csdl.examples.valid.ex_min_axiswise import example
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
    desired_output = np.amin(tensor, axis=1)
    np.testing.assert_almost_equal(sim['AxiswiseMin'], desired_output)

    partials_error = sim.check_partials(includes=['comp_AxiswiseMin'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_min_elementwise(name):
    from csdl.examples.valid.ex_min_elementwise import example
    exec('from {} import Simulator'.format(name))
    sim = example(eval('Simulator'))

    tensor1 = np.array([[1, 5, -8], [10, -3, -5]])
    tensor2 = np.array([[2, 6, 9], [-1, 2, 4]])

    desired_output = np.minimum(tensor1, tensor2)
    np.testing.assert_almost_equal(sim['ElementwiseMin'], desired_output)

    partials_error = sim.check_partials(includes=['comp_ElementwiseMin'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_min_multi_inputs_and_axis(name):
    with pytest.raises(Exception):
        from csdl.examples.invalid.ex_min_multi_inputs_and_axis import example
    exec('from {} import Simulator'.format(name))
    sim = example(eval('Simulator'))


def test_min_inputs_not_same_size(name):
    with pytest.raises(Exception):
        from csdl.examples.invalid.ex_min_inputs_not_same_size import example
    exec('from {} import Simulator'.format(name))
    sim = example(eval('Simulator'))
