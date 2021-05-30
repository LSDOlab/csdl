import numpy as np
import pytest


def test_vector_vector_dot(backend):
    from csdl.examples.valid.ex_dot_vector_vector import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    m = 3

    # Values for the two vectors
    vec1 = np.arange(m)
    vec2 = np.arange(m, 2 * m)

    # VECTOR VECTOR
    desired_output = np.dot(vec1, vec2)
    np.testing.assert_almost_equal(sim['VecVecDot'], desired_output)

    partials_error = sim.check_partials(includes=['comp_VecVecDot'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_tensor_tensor_first_dot(backend):
    from csdl.examples.valid.ex_dot_tensor_tensor_first import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    m = 3
    n = 4
    p = 5

    # Shape of the tensors
    ten_shape = (m, n, p)

    # Number of elements in the tensors
    num_ten_elements = np.prod(ten_shape)

    # Values for the two tensors
    ten1 = np.arange(num_ten_elements).reshape(ten_shape)
    ten2 = np.arange(num_ten_elements, 2 * num_ten_elements).reshape(ten_shape)

    # TENSOR TENSOR
    desired_output = np.sum(ten1 * ten2, axis=0)
    np.testing.assert_almost_equal(sim['TenTenDotFirst'], desired_output)

    partials_error = sim.check_partials(includes=['comp_TenTenDotFirst'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-4, rtol=1.e-4)


def test_tensor_tensor_last_dot(backend):
    from csdl.examples.valid.ex_dot_tensor_tensor_last import example
    exec('from {} import Simulator'.format(backend))
    sim = example(eval('Simulator'))

    m = 2
    n = 4
    p = 3

    # Shape of the tensors
    ten_shape = (m, n, p)

    # Number of elements in the tensors
    num_ten_elements = np.prod(ten_shape)

    # Values for the two tensors
    ten1 = np.arange(num_ten_elements).reshape(ten_shape)
    ten2 = np.arange(num_ten_elements, 2 * num_ten_elements).reshape(ten_shape)

    # TENSOR TENSOR
    desired_output = np.sum(ten1 * ten2, axis=2)
    np.testing.assert_almost_equal(sim['TenTenDotLast'], desired_output)

    partials_error = sim.check_partials(includes=['comp_TenTenDotLast'],
                                        out_stream=None,
                                        compact_print=True,
                                        method='cs')
    sim.assert_check_partials(partials_error, atol=1.e-4, rtol=1.e-4)


def test_dot_vec_different_shapes(backend):
    exec('from {} import Simulator'.format(backend))
    from csdl.examples.invalid.ex_dot_vec_different_shapes import example
    with pytest.raises(ValueError):
        example(eval('Simulator'))


def test_dot_ten_different_shapes(backend):
    exec('from {} import Simulator'.format(backend))
    from csdl.examples.invalid.ex_dot_ten_different_shapes import example
    with pytest.raises(ValueError):
        example(eval('Simulator'))


def test_dot_wrong_axis(backend):
    exec('from {} import Simulator'.format(backend))
    from csdl.examples.invalid.ex_dot_wrong_axis import example
    with pytest.raises(ValueError):
        example(eval('Simulator'))
